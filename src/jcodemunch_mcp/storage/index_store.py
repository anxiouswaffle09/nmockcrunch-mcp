"""Index storage with save/load, byte-offset content retrieval, and incremental indexing."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..parser.symbols import Symbol

# Bump this when the index schema changes in an incompatible way.
INDEX_VERSION = 2

# ---------------------------------------------------------------------------
# Process-level index cache: path -> (mtime, CodeIndex)
# Invalidated on any write via _invalidate_index_cache().
# ---------------------------------------------------------------------------
_index_cache: dict[str, tuple[float, "CodeIndex"]] = {}
_cache_lock = threading.Lock()

# Windows fallback for merge_refs locking (process-local only)
_refs_threading_lock = threading.Lock()


def _invalidate_index_cache(index_path: Path) -> None:
    cache_key = str(index_path)
    with _cache_lock:
        _index_cache.pop(cache_key, None)


def _file_hash(content: str) -> str:
    """SHA-256 hash of file content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _get_git_head(repo_path: Path) -> Optional[str]:
    """Get current HEAD commit hash for a git repo, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _make_file_meta(path: Path, content: str) -> dict:
    """Build file metadata dict with sha256 + mtime + size."""
    stat = path.stat()
    return {
        "sha256": _file_hash(content),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def _detect_changes_git(
    source_path: Path,
    stored_head: str,
    stored_file_metas: dict,
) -> tuple[set[str], set[str], str]:
    """Use git to get the set of modified and deleted files since last index.

    Returns (modified_set, deleted_set, current_head).
    Falls back to empty sets (triggering mtime fallback) on any git error.
    """
    current_head = _get_git_head(source_path) or ""
    modified: set[str] = set()
    deleted: set[str] = set()

    # Committed changes since last index
    if stored_head and current_head and current_head != stored_head:
        try:
            result = subprocess.run(
                ["git", "-C", str(source_path), "diff", "--name-only",
                 stored_head, current_head, "--"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    f = line.strip()
                    if f:
                        modified.add(f)
        except Exception:
            pass

    # Uncommitted working tree changes
    try:
        result = subprocess.run(
            ["git", "-C", str(source_path), "status", "--porcelain",
             "--untracked-files=all", "--"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if len(line) < 4:
                    continue
                xy = line[:2]
                path_part = line[3:]
                if "R" in xy and " -> " in path_part:
                    old, new = path_part.split(" -> ", 1)
                    deleted.add(old.strip())
                    modified.add(new.strip())
                elif xy.strip() == "D" or line[0] == "D":
                    deleted.add(path_part.strip())
                else:
                    modified.add(path_part.strip())
    except Exception:
        pass

    return modified, deleted, current_head


@dataclass
class CodeIndex:
    """Index for a repository's source code."""
    repo: str                    # "owner/repo"
    owner: str
    name: str
    indexed_at: str              # ISO timestamp
    source_files: list[str]      # All indexed file paths
    languages: dict[str, int]    # Language -> file count
    symbols: list[dict]          # Serialized Symbol dicts (without source content)
    index_version: int = INDEX_VERSION
    # file_hashes: file_path -> {sha256, mtime, size} or bare str (old format)
    file_hashes: dict = field(default_factory=dict)
    git_head: str = ""           # HEAD commit hash at index time (for git repos)
    file_summaries: dict[str, str] = field(default_factory=dict)  # file_path -> summary

    def get_symbol(self, symbol_id: str) -> Optional[dict]:
        """Find a symbol by ID."""
        for sym in self.symbols:
            if sym.get("id") == symbol_id:
                return sym
        return None

    def search(
        self,
        query: str,
        kind: Optional[str] = None,
        file_pattern: Optional[str] = None,
        language: Optional[str] = None,
    ) -> list[dict]:
        """Search symbols with weighted scoring.

        Returns dicts with an injected "score" field so callers don't need
        to re-score the results.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for sym in self.symbols:
            # Apply filters (language filter here avoids post-search filtering)
            if kind and sym.get("kind") != kind:
                continue
            if file_pattern and not self._match_pattern(sym.get("file", ""), file_pattern):
                continue
            if language and sym.get("language") != language:
                continue

            # Score symbol
            score = self._score_symbol(sym, query_lower, query_words)
            if score > 0:
                scored.append((score, sym))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": score, **sym} for score, sym in scored]

    def _match_pattern(self, file_path: str, pattern: str) -> bool:
        """Match file path against glob pattern."""
        import fnmatch
        return fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(file_path, f"*/{pattern}")

    def _score_symbol(self, sym: dict, query_lower: str, query_words: set) -> int:
        """Calculate search score for a symbol."""
        score = 0

        # 1. Exact name match (highest weight)
        name_lower = sym.get("name", "").lower()
        if query_lower == name_lower:
            score += 20
        elif query_lower in name_lower:
            score += 10

        # 2. Name word overlap
        for word in query_words:
            if word in name_lower:
                score += 5

        # 3. Signature match
        sig_lower = sym.get("signature", "").lower()
        if query_lower in sig_lower:
            score += 8
        for word in query_words:
            if word in sig_lower:
                score += 2

        # 4. Summary match
        summary_lower = sym.get("summary", "").lower()
        if query_lower in summary_lower:
            score += 5
        for word in query_words:
            if word in summary_lower:
                score += 1

        # 5. Keyword match
        keywords = set(sym.get("keywords", []))
        matching_keywords = query_words & keywords
        score += len(matching_keywords) * 3

        # 6. Docstring match
        doc_lower = sym.get("docstring", "").lower()
        for word in query_words:
            if word in doc_lower:
                score += 1

        return score


class IndexStore:
    """Storage for code indexes with byte-offset content retrieval."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize store.

        Args:
            base_path: Base directory for storage. Defaults to ~/.code-index/
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path.home() / ".code-index"

        self.base_path.mkdir(parents=True, exist_ok=True)

    def _safe_repo_component(self, value: str, field_name: str) -> str:
        """Validate owner/name components used in on-disk cache paths."""
        import re

        if not value or value in {".", ".."}:
            raise ValueError(f"Invalid {field_name}: {value!r}")
        if "/" in value or "\\" in value:
            raise ValueError(f"Invalid {field_name}: {value!r}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(f"Invalid {field_name}: {value!r}")
        return value

    def _repo_slug(self, owner: str, name: str) -> str:
        """Stable and safe slug used for index/content file paths."""
        safe_owner = self._safe_repo_component(owner, "owner")
        safe_name = self._safe_repo_component(name, "name")
        return f"{safe_owner}-{safe_name}"

    def _index_path(self, owner: str, name: str) -> Path:
        """Path to index JSON file."""
        return self.base_path / f"{self._repo_slug(owner, name)}.json"

    def _content_dir(self, owner: str, name: str) -> Path:
        """Path to raw content directory."""
        return self.base_path / self._repo_slug(owner, name)

    def _safe_content_path(self, content_dir: Path, relative_path: str) -> Optional[Path]:
        """Resolve a content path and ensure it stays within content_dir.

        Prevents path traversal when writing/reading cached raw files from
        untrusted repository paths.
        """
        try:
            base = content_dir.resolve()
            candidate = (content_dir / relative_path).resolve()
            if os.path.commonpath([str(base), str(candidate)]) != str(base):
                return None
            return candidate
        except (OSError, ValueError):
            return None

    def save_index(
        self,
        owner: str,
        name: str,
        source_files: list[str],
        symbols: list[Symbol],
        raw_files: dict[str, str],
        languages: dict[str, int],
        file_hashes: Optional[dict] = None,
        git_head: str = "",
        folder_path: Optional[Path] = None,
    ) -> "CodeIndex":
        """Save index and raw files to storage.

        Args:
            owner: Repository owner.
            name: Repository name.
            source_files: List of indexed file paths.
            symbols: List of Symbol objects.
            raw_files: Dict mapping file path to raw content.
            languages: Dict mapping language to file count.
            file_hashes: Optional precomputed file metadata map.
                         Values may be bare sha256 strings (old format) or
                         {sha256, mtime, size} dicts (new format).
            git_head: Optional HEAD commit hash at index time.
            folder_path: Root folder path (used to build mtime/size metadata
                         when file_hashes is None and files are local).

        Returns:
            CodeIndex object.
        """
        # Compute file metadata if not provided
        if file_hashes is None:
            if folder_path is not None:
                file_hashes = {}
                for fp, content in raw_files.items():
                    abs_path = folder_path / fp
                    try:
                        file_hashes[fp] = _make_file_meta(abs_path, content)
                    except OSError:
                        file_hashes[fp] = _file_hash(content)
            else:
                file_hashes = {fp: _file_hash(content) for fp, content in raw_files.items()}

        # Create index
        index = CodeIndex(
            repo=f"{owner}/{name}",
            owner=owner,
            name=name,
            indexed_at=datetime.now().isoformat(),
            source_files=source_files,
            languages=languages,
            symbols=[self._symbol_to_dict(s) for s in symbols],
            index_version=INDEX_VERSION,
            file_hashes=file_hashes,
            git_head=git_head,
        )

        # Save index JSON atomically: write to temp then rename
        index_path = self._index_path(owner, name)
        tmp_path = index_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._index_to_dict(index), f, indent=2)
        # Atomic rename (on POSIX; best-effort on Windows)
        tmp_path.replace(index_path)
        _invalidate_index_cache(index_path)

        # Save raw files
        content_dir = self._content_dir(owner, name)
        content_dir.mkdir(parents=True, exist_ok=True)

        for file_path, content in raw_files.items():
            file_dest = self._safe_content_path(content_dir, file_path)
            if not file_dest:
                raise ValueError(f"Unsafe file path in raw_files: {file_path}")
            file_dest.parent.mkdir(parents=True, exist_ok=True)
            with open(file_dest, "w", encoding="utf-8") as f:
                f.write(content)

        return index

    def load_index(self, owner: str, name: str) -> Optional[CodeIndex]:
        """Load index from storage. Rejects incompatible versions.

        Uses a process-level mtime-gated cache to avoid repeated JSON parses
        when nothing has changed on disk.
        """
        index_path = self._index_path(owner, name)

        if not index_path.exists():
            return None

        try:
            current_mtime = index_path.stat().st_mtime
        except OSError:
            return None

        cache_key = str(index_path)
        with _cache_lock:
            if cache_key in _index_cache:
                cached_mtime, cached_index = _index_cache[cache_key]
                if cached_mtime == current_mtime:
                    return cached_index

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        # Version check
        stored_version = data.get("index_version", 1)
        if stored_version > INDEX_VERSION:
            return None  # Future version we can't read

        index = CodeIndex(
            repo=data["repo"],
            owner=data["owner"],
            name=data["name"],
            indexed_at=data["indexed_at"],
            source_files=data["source_files"],
            languages=data["languages"],
            symbols=data["symbols"],
            index_version=stored_version,
            file_hashes=data.get("file_hashes", {}),
            git_head=data.get("git_head", ""),
            file_summaries=data.get("file_summaries", {}),
        )

        with _cache_lock:
            _index_cache[cache_key] = (current_mtime, index)

        return index

    def get_symbol_content(
        self,
        owner: str,
        name: str,
        symbol_id: str,
        index: Optional["CodeIndex"] = None,
    ) -> Optional[str]:
        """Read symbol source using stored byte offsets.

        This is O(1) - no re-parsing, just seek + read.

        Args:
            index: Pre-loaded CodeIndex to avoid a second load_index call.
                   If None, loads from disk.
        """
        if index is None:
            index = self.load_index(owner, name)
        if not index:
            return None

        symbol = index.get_symbol(symbol_id)
        if not symbol:
            return None

        file_path = self._safe_content_path(self._content_dir(owner, name), symbol["file"])
        if not file_path:
            return None

        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            f.seek(symbol["byte_offset"])
            source_bytes = f.read(symbol["byte_length"])

        return source_bytes.decode("utf-8", errors="replace")

    def detect_changes_fast(
        self,
        owner: str,
        name: str,
        folder_path: Path,
        current_discovered: list[Path],
        source_path: Optional[Path] = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Two-phase change detection: mtime first, SHA-256 only for suspects.

        Also uses git (when available) as an even faster first layer.

        Args:
            owner: Repository owner.
            name: Repository name.
            folder_path: Resolved root folder.
            current_discovered: Paths from discover_local_files.
            source_path: Path to the git repo root (usually same as folder_path).

        Returns:
            Tuple of (changed_files, new_files, deleted_files) — relative paths.
        """
        index = self.load_index(owner, name)
        if not index:
            return [], [p.relative_to(folder_path).as_posix() for p in current_discovered], []

        old_meta = index.file_hashes
        current_rel = {
            p.relative_to(folder_path).as_posix(): p
            for p in current_discovered
        }

        old_set = set(old_meta.keys())
        new_set = set(current_rel.keys())

        deleted = list(old_set - new_set)
        added = list(new_set - old_set)
        possibly_changed: list[str] = []

        # Try git-accelerated detection first
        git_modified: set[str] = set()
        git_deleted: set[str] = set()
        used_git = False
        if source_path is not None or folder_path is not None:
            git_root = source_path or folder_path
            git_modified, git_deleted, _ = _detect_changes_git(
                git_root, index.git_head, old_meta
            )
            used_git = bool(git_modified or git_deleted or index.git_head)

        for rel_path in old_set & new_set:
            meta = old_meta[rel_path]
            if isinstance(meta, str):
                # Old format — no mtime/size, must SHA-256 verify
                possibly_changed.append(rel_path)
                continue
            # If git told us this file changed, skip mtime check
            if used_git and rel_path in git_modified:
                possibly_changed.append(rel_path)
                continue
            # mtime + size fast check
            abs_path = current_rel[rel_path]
            try:
                stat = abs_path.stat()
                if stat.st_mtime != meta.get("mtime") or stat.st_size != meta.get("size"):
                    possibly_changed.append(rel_path)
            except OSError:
                deleted.append(rel_path)

        # Phase 2: SHA-256 only for suspects
        changed: list[str] = []
        for rel_path in possibly_changed:
            abs_path = current_rel.get(rel_path)
            if abs_path is None:
                continue
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
                new_hash = _file_hash(content)
                meta = old_meta[rel_path]
                old_hash = meta if isinstance(meta, str) else meta.get("sha256", "")
                if new_hash != old_hash:
                    changed.append(rel_path)
            except OSError:
                deleted.append(rel_path)

        return changed, added, deleted

    def detect_changes(
        self,
        owner: str,
        name: str,
        current_files: dict[str, str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Detect changed, new, and deleted files by comparing hashes.

        Args:
            owner: Repository owner.
            name: Repository name.
            current_files: Dict mapping file_path -> content for current state.

        Returns:
            Tuple of (changed_files, new_files, deleted_files).
        """
        index = self.load_index(owner, name)
        if not index:
            # No existing index: all files are new
            return [], list(current_files.keys()), []

        old_hashes = index.file_hashes
        current_hashes = {fp: _file_hash(content) for fp, content in current_files.items()}

        old_set = set(old_hashes.keys())
        new_set = set(current_hashes.keys())

        new_files = list(new_set - old_set)
        deleted_files = list(old_set - new_set)
        changed_files = [
            fp for fp in (old_set & new_set)
            if old_hashes[fp] != current_hashes[fp]
        ]

        return changed_files, new_files, deleted_files

    def incremental_save(
        self,
        owner: str,
        name: str,
        changed_files: list[str],
        new_files: list[str],
        deleted_files: list[str],
        new_symbols: list[Symbol],
        raw_files: dict[str, str],
        languages: dict[str, int],
        git_head: str = "",
        folder_path: Optional[Path] = None,
        file_hashes_override: Optional[dict] = None,
    ) -> Optional[CodeIndex]:
        """Incrementally update an existing index.

        Removes symbols for deleted/changed files, adds new symbols,
        updates raw content, and saves atomically.

        Args:
            owner: Repository owner.
            name: Repository name.
            changed_files: Files that changed (symbols will be replaced).
            new_files: New files (symbols will be added).
            deleted_files: Deleted files (symbols will be removed).
            new_symbols: Symbols extracted from changed + new files.
            raw_files: Raw content for changed + new files.
            languages: Legacy language counts (ignored when symbols are present).
            git_head: Current HEAD commit hash.

        Returns:
            Updated CodeIndex, or None if no existing index.
        """
        index = self.load_index(owner, name)
        if not index:
            return None

        # Remove symbols for deleted and changed files
        files_to_remove = set(deleted_files) | set(changed_files)
        kept_symbols = [s for s in index.symbols if s.get("file") not in files_to_remove]

        # Add new symbols
        all_symbols_dicts = kept_symbols + [self._symbol_to_dict(s) for s in new_symbols]
        recomputed_languages = self._languages_from_symbols(all_symbols_dicts)
        if not recomputed_languages and languages:
            recomputed_languages = languages

        # Update source files list
        old_files = set(index.source_files)
        for f in deleted_files:
            old_files.discard(f)
        for f in new_files:
            old_files.add(f)
        for f in changed_files:
            old_files.add(f)

        # Update file hashes
        if file_hashes_override is not None:
            # Caller supplies the complete post-update hashes (e.g. remote blob SHAs)
            file_hashes = file_hashes_override
        else:
            # Compute from content, using mtime+size metadata when folder_path is available
            file_hashes = dict(index.file_hashes)
            for f in deleted_files:
                file_hashes.pop(f, None)
            for fp, content in raw_files.items():
                if folder_path is not None:
                    abs_path = folder_path / fp
                    try:
                        file_hashes[fp] = _make_file_meta(abs_path, content)
                    except OSError:
                        file_hashes[fp] = _file_hash(content)
                else:
                    file_hashes[fp] = _file_hash(content)

        # Build updated index
        updated = CodeIndex(
            repo=f"{owner}/{name}",
            owner=owner,
            name=name,
            indexed_at=datetime.now().isoformat(),
            source_files=sorted(old_files),
            languages=recomputed_languages,
            symbols=all_symbols_dicts,
            index_version=INDEX_VERSION,
            file_hashes=file_hashes,
            git_head=git_head,
        )

        # Save atomically
        index_path = self._index_path(owner, name)
        tmp_path = index_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._index_to_dict(updated), f, indent=2)
        tmp_path.replace(index_path)
        _invalidate_index_cache(index_path)

        # Update raw files
        content_dir = self._content_dir(owner, name)
        content_dir.mkdir(parents=True, exist_ok=True)

        # Remove deleted files from content dir
        for fp in deleted_files:
            dead = self._safe_content_path(content_dir, fp)
            if not dead:
                continue
            if dead.exists():
                dead.unlink()

        # Write changed + new files
        for fp, content in raw_files.items():
            dest = self._safe_content_path(content_dir, fp)
            if not dest:
                raise ValueError(f"Unsafe file path in raw_files: {fp}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(content)

        return updated

    def _languages_from_symbols(self, symbols: list[dict]) -> dict[str, int]:
        """Compute language->file_count from serialized symbols."""
        file_languages: dict[str, str] = {}
        for sym in symbols:
            file_path = sym.get("file")
            language = sym.get("language")
            if not file_path or not language:
                continue
            file_languages.setdefault(file_path, language)

        counts: dict[str, int] = {}
        for language in file_languages.values():
            counts[language] = counts.get(language, 0) + 1
        return counts

    def list_repos(self) -> list[dict]:
        """List all indexed repositories."""
        repos = []

        for index_file in self.base_path.glob("*.json"):
            if index_file.name.endswith("-refs.json"):
                continue
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                repos.append({
                    "repo": data["repo"],
                    "indexed_at": data["indexed_at"],
                    "symbol_count": len(data["symbols"]),
                    "file_count": len(data["source_files"]),
                    "languages": data["languages"],
                    "index_version": data.get("index_version", 1),
                })
            except Exception:
                continue

        return repos

    def delete_index(self, owner: str, name: str) -> bool:
        """Delete an index and its raw files."""
        index_path = self._index_path(owner, name)
        refs_path = self._refs_path(owner, name)
        content_dir = self._content_dir(owner, name)

        deleted = False

        if index_path.exists():
            index_path.unlink()
            deleted = True

        if refs_path.exists():
            refs_path.unlink()
            deleted = True

        if content_dir.exists():
            shutil.rmtree(content_dir)
            deleted = True

        return deleted

    # ------------------------------------------------------------------
    # Cross-reference storage (refs.json — separate from main index)
    # ------------------------------------------------------------------

    def _refs_path(self, owner: str, name: str) -> Path:
        """Path to refs JSON file."""
        return self.base_path / f"{self._repo_slug(owner, name)}-refs.json"

    def save_refs(self, owner: str, name: str, refs: list[dict]) -> None:
        """Save cross-reference table atomically."""
        refs_path = self._refs_path(owner, name)
        tmp_path = refs_path.with_suffix(".json.tmp")
        payload = {"repo": f"{owner}/{name}", "ref_count": len(refs), "refs": refs}
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp_path.replace(refs_path)

    def load_refs(self, owner: str, name: str) -> Optional[list[dict]]:
        """Load cross-reference table, or None if not yet built."""
        refs_path = self._refs_path(owner, name)
        if not refs_path.exists():
            return None
        try:
            with open(refs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("refs", [])
        except Exception:
            return None

    def merge_refs(self, owner: str, name: str, new_refs: list[dict], removed_files: set[str]) -> None:
        """Merge new refs into existing table, removing stale entries for changed/deleted files.

        Uses file-level locking to prevent concurrent merge_refs calls from racing.
        """
        refs_path = self._refs_path(owner, name)
        refs_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = refs_path.with_suffix(".lock")

        if sys.platform == "win32":
            # On Windows, use a threading lock (process-local)
            with _refs_threading_lock:
                existing = self.load_refs(owner, name) or []
                kept = [r for r in existing if r.get("caller_file") not in removed_files]
                self.save_refs(owner, name, kept + new_refs)
        else:
            import fcntl
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    existing = self.load_refs(owner, name) or []
                    kept = [r for r in existing if r.get("caller_file") not in removed_files]
                    self.save_refs(owner, name, kept + new_refs)
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _symbol_to_dict(self, symbol: Symbol) -> dict:
        """Convert Symbol to dict (without source content)."""
        return {
            "id": symbol.id,
            "file": symbol.file,
            "name": symbol.name,
            "qualified_name": symbol.qualified_name,
            "kind": symbol.kind,
            "language": symbol.language,
            "signature": symbol.signature,
            "docstring": symbol.docstring,
            "summary": symbol.summary,
            "decorators": symbol.decorators,
            "keywords": symbol.keywords,
            "parent": symbol.parent,
            "line": symbol.line,
            "end_line": symbol.end_line,
            "byte_offset": symbol.byte_offset,
            "byte_length": symbol.byte_length,
            "content_hash": symbol.content_hash,
        }

    def _index_to_dict(self, index: CodeIndex) -> dict:
        """Convert CodeIndex to dict."""
        return {
            "repo": index.repo,
            "owner": index.owner,
            "name": index.name,
            "indexed_at": index.indexed_at,
            "source_files": index.source_files,
            "languages": index.languages,
            "symbols": index.symbols,
            "index_version": index.index_version,
            "file_hashes": index.file_hashes,
            "git_head": index.git_head,
            "file_summaries": index.file_summaries,
        }

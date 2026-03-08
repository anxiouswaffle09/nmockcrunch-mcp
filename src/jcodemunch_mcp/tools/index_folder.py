"""Index local folder tool - walk, parse, summarize, save."""

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pathspec

logger = logging.getLogger(__name__)

from ..parser import parse_file, extract_refs, LANGUAGE_EXTENSIONS
from ..security import (
    validate_path,
    should_exclude_file,
    DEFAULT_MAX_FILE_SIZE,
    get_max_index_files,
)
from ..storage import IndexStore
from ..summarizer import summarize_symbols
from ._utils import invalidate_repo_name_cache


# File patterns to skip (sync with index_repo.py)
SKIP_PATTERNS = [
    "node_modules/", "vendor/", "venv/", ".venv/", "__pycache__/",
    "dist/", "build/", ".git/", ".tox/", ".mypy_cache/",
    "target/",
    ".gradle/",
    "test_data/", "testdata/", "fixtures/", "snapshots/",
    "migrations/",
    ".min.js", ".min.ts", ".bundle.js",
    "package-lock.json", "yarn.lock", "go.sum",
    "generated/", "proto/",
]


def should_skip_file(path: str) -> bool:
    """Check if file should be skipped based on path patterns."""
    # Normalize path separators for matching
    normalized = path.replace("\\", "/")
    for pattern in SKIP_PATTERNS:
        if pattern in normalized:
            return True
    return False


def _load_gitignore(folder_path: Path) -> Optional[pathspec.PathSpec]:
    """Load .gitignore from the folder root if it exists."""
    gitignore_path = folder_path / ".gitignore"
    if gitignore_path.is_file():
        try:
            content = gitignore_path.read_text(encoding="utf-8", errors="replace")
            return pathspec.PathSpec.from_lines("gitignore", content.splitlines())
        except Exception:
            pass
    return None


def discover_local_files(
    folder_path: Path,
    max_files: Optional[int] = None,
    max_size: int = DEFAULT_MAX_FILE_SIZE,
    extra_ignore_patterns: Optional[list[str]] = None,
    follow_symlinks: bool = False,
) -> tuple[list[Path], list[str], dict[str, int]]:
    """Discover source files in a local folder with security filtering.

    Args:
        folder_path: Root folder to scan (must be resolved).
        max_files: Maximum number of files to index.
        max_size: Maximum file size in bytes.
        extra_ignore_patterns: Additional gitignore-style patterns to exclude.
        follow_symlinks: Whether to follow symlinks (default False for safety).

    Returns:
        Tuple of (list of Path objects for source files, list of warning strings).
    """
    max_files = get_max_index_files(max_files)
    files = []
    warnings = []
    root = folder_path.resolve()

    skip_counts: dict[str, int] = {
        "symlink": 0,
        "symlink_escape": 0,
        "path_traversal": 0,
        "skip_pattern": 0,
        "gitignore": 0,
        "extra_ignore": 0,
        "secret": 0,
        "wrong_extension": 0,
        "too_large": 0,
        "unreadable": 0,
        "binary": 0,
        "file_limit": 0,
    }
    visited_dirs: set[str] = set()

    # Load .gitignore
    gitignore_spec = _load_gitignore(root)

    # Build extra ignore spec if provided
    extra_spec = None
    if extra_ignore_patterns:
        try:
            extra_spec = pathspec.PathSpec.from_lines("gitignore", extra_ignore_patterns)
        except Exception:
            pass

    # Map should_exclude_file() reason strings to skip_counts keys
    _REASON_TO_KEY: dict[str, str] = {
        "symlink_escape": "symlink_escape",
        "path_traversal": "path_traversal",
        "outside_root": "path_traversal",
        "secret_file": "secret",
        "file_too_large": "too_large",
        "unreadable": "unreadable",
        "binary": "binary",
    }

    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=follow_symlinks):
        dir_path = Path(dirpath)
        try:
            real_dir = str(dir_path.resolve())
        except OSError:
            dirnames.clear()
            skip_counts["unreadable"] += 1
            continue
        if real_dir in visited_dirs:
            dirnames.clear()
            continue
        visited_dirs.add(real_dir)
        try:
            dir_rel = dir_path.relative_to(root).as_posix()
        except ValueError:
            dirnames.clear()
            continue

        # Prune directories in-place so os.walk won't descend into them
        pruned_dirnames = []
        for d in dirnames:
            rel_dir = (f"{dir_rel}/{d}/" if dir_rel != "." else f"{d}/")
            if should_skip_file(rel_dir.lstrip("./")):
                continue
            if gitignore_spec and gitignore_spec.match_file(rel_dir):
                continue
            if extra_spec and extra_spec.match_file(rel_dir):
                continue
            try:
                child_real = str((dir_path / d).resolve())
            except OSError:
                skip_counts["unreadable"] += 1
                continue
            if child_real in visited_dirs:
                continue
            pruned_dirnames.append(d)
        dirnames[:] = pruned_dirnames

        for filename in filenames:
            file_path = dir_path / filename

            # Symlink: skip all symlinks when follow_symlinks is False
            if not follow_symlinks and file_path.is_symlink():
                skip_counts["symlink"] += 1
                logger.debug("SKIP symlink: %s", file_path)
                continue

            # Get relative path for gitignore / skip-pattern matching
            try:
                rel_path = file_path.relative_to(root).as_posix()
            except ValueError:
                skip_counts["path_traversal"] += 1
                logger.debug("SKIP relative_to_failed: %s", file_path)
                continue

            # Skip patterns (file-level check)
            if should_skip_file(rel_path):
                skip_counts["skip_pattern"] += 1
                logger.debug("SKIP skip_pattern: %s", rel_path)
                continue

            # .gitignore matching
            if gitignore_spec and gitignore_spec.match_file(rel_path):
                skip_counts["gitignore"] += 1
                logger.debug("SKIP gitignore: %s", rel_path)
                continue

            # Extra ignore patterns
            if extra_spec and extra_spec.match_file(rel_path):
                skip_counts["extra_ignore"] += 1
                logger.debug("SKIP extra_ignore: %s", rel_path)
                continue

            # Extension filter
            ext = file_path.suffix
            if ext not in LANGUAGE_EXTENSIONS:
                skip_counts["wrong_extension"] += 1
                logger.debug("SKIP wrong_extension: %s", rel_path)
                continue

            # Security checks: path traversal, symlink escape, secret,
            # file size, and binary detection — all via one call
            reason = should_exclude_file(file_path, root, max_file_size=max_size)
            if reason is not None:
                key = _REASON_TO_KEY.get(reason, reason)
                skip_counts[key] = skip_counts.get(key, 0) + 1
                if reason in ("symlink_escape", "binary"):
                    warnings.append(f"Skipped {reason.replace('_', ' ')}: {rel_path}")
                elif reason == "secret_file":
                    warnings.append(f"Skipped secret file: {rel_path}")
                logger.debug("SKIP %s: %s", reason, rel_path)
                continue

            logger.debug("ACCEPT: %s", rel_path)
            files.append(file_path)

    logger.info(
        "Discovery complete — accepted: %d, skipped by reason: %s",
        len(files),
        skip_counts,
    )

    # File count limit with prioritization
    if len(files) > max_files:
        skip_counts["file_limit"] = len(files) - max_files
        # Prioritize: src/, lib/, pkg/, cmd/, internal/ first
        priority_dirs = ["src/", "lib/", "pkg/", "cmd/", "internal/"]

        def priority_key(file_path: Path) -> tuple:
            try:
                rel_path = file_path.relative_to(root).as_posix()
            except ValueError:
                return (999, 999, str(file_path))

            # Check if in priority dir
            for i, prefix in enumerate(priority_dirs):
                if rel_path.startswith(prefix):
                    return (i, rel_path.count("/"), rel_path)
            # Not in priority dir - sort after
            return (len(priority_dirs), rel_path.count("/"), rel_path)

        files.sort(key=priority_key)
        files = files[:max_files]

    return files, warnings, skip_counts


def index_folder(
    path: str,
    use_ai_summaries: bool = True,
    storage_path: Optional[str] = None,
    extra_ignore_patterns: Optional[list[str]] = None,
    follow_symlinks: bool = False,
    incremental: bool = False,
) -> dict:
    """Index a local folder containing source code.

    Args:
        path: Path to local folder (absolute or relative).
        use_ai_summaries: Whether to use AI for symbol summaries.
        storage_path: Custom storage path (default: ~/.code-index/).
        extra_ignore_patterns: Additional gitignore-style patterns to exclude.
        follow_symlinks: Whether to follow symlinks (default False for safety).
        incremental: When True and an existing index exists, only re-index changed files.

    Returns:
        Dict with indexing results.
    """
    # Resolve folder path
    folder_path = Path(path).expanduser().resolve()

    if not folder_path.exists():
        return {"success": False, "error": f"Folder not found: {path}"}

    if not folder_path.is_dir():
        return {"success": False, "error": f"Path is not a directory: {path}"}

    warnings = []
    max_files = get_max_index_files()

    try:
        # ── Git pre-check: skip the expensive file-tree walk when nothing
        #    changed since the last index.  Only applies to incremental mode
        #    inside a git repo with an existing index.
        if incremental:
            from ..storage.index_store import _detect_changes_git, _get_git_head

            repo_name = folder_path.name
            owner = "local"
            store = IndexStore(base_path=storage_path)
            existing = store.load_index(owner, repo_name)

            if existing is not None:
                git_modified, git_deleted, current_head = _detect_changes_git(
                    folder_path, existing.git_head, existing.file_hashes,
                )

                if not git_modified and not git_deleted:
                    # Git says nothing is dirty and HEAD hasn't changed.
                    # No need to walk the file tree at all.
                    if current_head and current_head == existing.git_head:
                        logger.debug("git pre-check: clean, skipping file walk")
                        return {
                            "success": True,
                            "message": "No changes detected",
                            "repo": f"{owner}/{repo_name}",
                            "folder_path": str(folder_path),
                            "changed": 0, "new": 0, "deleted": 0,
                        }
                    # HEAD changed but no dirty files — e.g. someone did
                    # `git stash` or `git checkout`.  Fall through to the
                    # full path so detect_changes_fast picks up committed
                    # diffs and new/deleted files.
                elif not git_deleted:
                    # Files are dirty but nothing was deleted (no new/removed
                    # files possible from git's perspective).  Spot-check the
                    # dirty files' metadata against the index — if timestamps
                    # match, we already indexed this state.
                    all_match = True
                    for rel_path in git_modified:
                        meta = existing.file_hashes.get(rel_path)
                        if not meta or isinstance(meta, str):
                            all_match = False
                            break
                        abs_path = folder_path / rel_path
                        try:
                            st = abs_path.stat()
                        except OSError:
                            all_match = False
                            break
                        if (
                            st.st_mtime_ns != meta.get("mtime_ns")
                            or st.st_size != meta.get("size")
                        ):
                            all_match = False
                            break
                    if all_match:
                        logger.debug(
                            "git pre-check: %d dirty files already indexed, "
                            "skipping file walk", len(git_modified),
                        )
                        return {
                            "success": True,
                            "message": "No changes detected",
                            "repo": f"{owner}/{repo_name}",
                            "folder_path": str(folder_path),
                            "changed": 0, "new": 0, "deleted": 0,
                        }
                # If we reach here, something genuinely changed — fall through
                # to the full discover + detect_changes_fast path.

        # Discover source files (with security filtering)
        source_files, discover_warnings, skip_counts = discover_local_files(
            folder_path,
            max_files=max_files,
            extra_ignore_patterns=extra_ignore_patterns,
            follow_symlinks=follow_symlinks,
        )
        warnings.extend(discover_warnings)
        logger.info("Discovery skip counts: %s", skip_counts)

        if not source_files:
            return {"success": False, "error": "No source files found"}

        # Create repo identifier from folder path
        repo_name = folder_path.name
        owner = "local"
        store = IndexStore(base_path=storage_path)

        # Incremental path: detect changes, then read only what changed
        if incremental and store.load_index(owner, repo_name) is not None:
            changed, new, deleted = store.detect_changes_fast(
                owner, repo_name, folder_path, source_files, source_path=folder_path
            )

            if not changed and not new and not deleted:
                return {
                    "success": True,
                    "message": "No changes detected",
                    "repo": f"{owner}/{repo_name}",
                    "folder_path": str(folder_path),
                    "changed": 0, "new": 0, "deleted": 0,
                }

            # Read ONLY changed + new files
            files_to_parse = set(changed) | set(new)
            parsed_content: dict[str, str] = {}
            for file_path in source_files:
                if not validate_path(folder_path, file_path):
                    continue
                try:
                    rel_path = file_path.relative_to(folder_path).as_posix()
                except ValueError:
                    continue
                if rel_path not in files_to_parse:
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    warnings.append(f"Failed to read {file_path}: {e}")
                    continue
                ext = file_path.suffix
                if ext not in LANGUAGE_EXTENSIONS:
                    continue
                parsed_content[rel_path] = content

            new_symbols = []
            raw_files_subset: dict[str, str] = {}

            incremental_no_symbols: list[str] = []
            for rel_path in files_to_parse:
                content = parsed_content.get(rel_path)
                if content is None:
                    continue  # file disappeared between discover and read
                raw_files_subset[rel_path] = content
                ext = os.path.splitext(rel_path)[1]
                language = LANGUAGE_EXTENSIONS.get(ext)
                if not language:
                    continue
                try:
                    symbols = parse_file(content, rel_path, language)
                    if symbols:
                        new_symbols.extend(symbols)
                    else:
                        incremental_no_symbols.append(rel_path)
                        logger.debug("NO SYMBOLS (incremental): %s", rel_path)
                except Exception as e:
                    warnings.append(f"Failed to parse {rel_path}: {e}")
                    logger.debug("PARSE ERROR (incremental): %s — %s", rel_path, e)

            logger.info(
                "Incremental parsing — with symbols: %d, no symbols: %d",
                len(new_symbols),
                len(incremental_no_symbols),
            )

            new_symbols = summarize_symbols(new_symbols, use_ai=use_ai_summaries)

            from ..storage.index_store import _get_git_head
            git_head = _get_git_head(folder_path) or ""

            # Only pass files we successfully read to incremental_save.
            # Files that failed to read are excluded so their existing symbols
            # are not stripped from the index — same pattern as index_repo.py.
            successfully_read = set(parsed_content.keys())
            actual_changed = [f for f in changed if f in successfully_read]
            actual_new = [f for f in new if f in successfully_read]

            # Check before incremental_save — safe either way since incremental_save
            # never reads or writes the refs file. The check must happen before we
            # save so we know whether to run a full backfill or a partial merge.
            # Note: index_repo.py checks this after incremental_save — both positions
            # are equivalent; the ordering difference is cosmetic, not a bug.
            needs_full_backfill = store.load_refs(owner, repo_name) is None
            updated = store.incremental_save(
                owner=owner, name=repo_name,
                changed_files=actual_changed, new_files=actual_new, deleted_files=deleted,
                new_symbols=new_symbols, raw_files=raw_files_subset,
                # languages={} is intentional — incremental_save recomputes language
                # counts from the merged symbol list via _languages_from_symbols.
                # This parameter is a legacy fallback only used when symbols are absent.
                languages={}, git_head=git_head,
                folder_path=folder_path,
            )

            if needs_full_backfill:
                # refs.json missing — backfill refs for ALL current files.
                # Reuse already-read content; read remaining files from disk on demand.
                #
                # updated.symbols is list[dict] (serialised from disk by incremental_save),
                # not list[Symbol]. We use SimpleNamespace proxies below so extract_refs
                # gets the same duck-typed interface it expects from Symbol objects.
                all_sym_dicts = updated.symbols if updated else []
                all_refs: list[dict] = []
                for file_path in source_files:
                    if not validate_path(folder_path, file_path):
                        continue  # discovery already filters path-traversal; unreachable in practice
                    try:
                        rel_path = file_path.relative_to(folder_path).as_posix()
                    except ValueError:
                        continue  # same — discovery guarantees all source_files are under folder_path
                    ext = os.path.splitext(rel_path)[1]
                    language = LANGUAGE_EXTENSIONS.get(ext)
                    if not language:
                        continue  # discovery's extension filter already covers this; defensive only
                    content = parsed_content.get(rel_path)
                    if content is None:
                        try:
                            content = file_path.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            continue  # file disappeared or became unreadable between parse and backfill; skip silently
                    try:
                        # The four dict keys (line/end_line/id/file) are always written
                        # by _symbol_to_dict — no KeyError risk. The try/except here
                        # guards extract_refs (tree-sitter parse on arbitrary content).
                        proxies = [
                            SimpleNamespace(line=s["line"], end_line=s["end_line"],
                                            id=s["id"], file=s["file"])
                            for s in all_sym_dicts if s.get("file") == rel_path
                        ]
                        all_refs.extend(extract_refs(content, rel_path, language, proxies))
                    except Exception:
                        pass
                store.save_refs(owner, repo_name, all_refs)
            else:
                # Update cross-references for changed/new files only.
                # new_symbols is list[Symbol] (freshly parsed this run), unlike
                # all_sym_dicts above which is list[dict] loaded from disk. The two
                # paths use different types because they have different sources: the
                # backfill needs ALL symbols (including untouched files) which only
                # exist as serialised dicts; this path only needs this run's symbols.
                incremental_refs: list[dict] = []
                for rel_path in files_to_parse:
                    content = parsed_content.get(rel_path)
                    if content is None:
                        continue
                    ext = os.path.splitext(rel_path)[1]
                    language = LANGUAGE_EXTENSIONS.get(ext)
                    if not language:
                        continue
                    try:
                        file_symbols = [s for s in new_symbols if s.file == rel_path]
                        incremental_refs.extend(extract_refs(content, rel_path, language, file_symbols))
                    except Exception:
                        pass
                # Use actual_changed (not changed) so refs for unread files are
                # also preserved — consistent with symbol preservation above.
                removed = set(actual_changed) | set(deleted)
                store.merge_refs(owner, repo_name, incremental_refs, removed)

            # Incremental result deliberately omits file_count, languages, and files
            # (the 20-file sample) that the full-index result includes. Those require
            # loading the entire index or re-scanning all files — too expensive for a
            # diff-only run. Callers that need them should trigger a full index.
            result = {
                "success": True,
                "repo": f"{owner}/{repo_name}",
                "folder_path": str(folder_path),
                "incremental": True,
                "changed": len(actual_changed), "new": len(actual_new), "deleted": len(deleted),
                "symbol_count": len(updated.symbols) if updated else 0,
                "indexed_at": updated.indexed_at if updated else "",
                "ref_count": store.get_ref_count(owner, repo_name),
                "discovery_skip_counts": skip_counts,
                "no_symbols_count": len(incremental_no_symbols),
                "no_symbols_files": incremental_no_symbols[:50],
            }
            if warnings:
                result["warnings"] = warnings
            return result

        # Full index path
        current_files: dict[str, str] = {}
        for file_path in source_files:
            if not validate_path(folder_path, file_path):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                warnings.append(f"Failed to read {file_path}: {e}")
                continue
            try:
                rel_path = file_path.relative_to(folder_path).as_posix()
            except ValueError:
                continue
            ext = file_path.suffix
            if ext not in LANGUAGE_EXTENSIONS:
                continue
            current_files[rel_path] = content

        all_symbols = []
        languages = {}
        raw_files = {}
        parsed_files = []

        no_symbols_files: list[str] = []
        for rel_path, content in current_files.items():
            ext = os.path.splitext(rel_path)[1]
            language = LANGUAGE_EXTENSIONS.get(ext)
            if not language:
                continue
            try:
                symbols = parse_file(content, rel_path, language)
                raw_files[rel_path] = content
                if symbols:
                    all_symbols.extend(symbols)
                    file_language = symbols[0].language or language
                    languages[file_language] = languages.get(file_language, 0) + 1
                    parsed_files.append(rel_path)
                else:
                    no_symbols_files.append(rel_path)
                    logger.debug("NO SYMBOLS: %s", rel_path)
            except Exception as e:
                warnings.append(f"Failed to parse {rel_path}: {e}")
                logger.debug("PARSE ERROR: %s — %s", rel_path, e)
                continue

        logger.info(
            "Parsing complete — with symbols: %d, no symbols: %d",
            len(parsed_files),
            len(no_symbols_files),
        )

        if not all_symbols:
            return {"success": False, "error": "No symbols extracted from files"}

        # Generate summaries
        all_symbols = summarize_symbols(all_symbols, use_ai=use_ai_summaries)

        # Save index — let save_index build mtime+size metadata from disk
        from ..storage.index_store import _get_git_head as _ggh
        git_head_full = _ggh(folder_path) or ""
        saved_index = store.save_index(
            owner=owner,
            name=repo_name,
            source_files=sorted(current_files),
            symbols=all_symbols,
            raw_files=raw_files,
            languages=languages,
            folder_path=folder_path,
            git_head=git_head_full,
        )

        invalidate_repo_name_cache()

        # Extract and save cross-references
        all_refs = []
        for rel_path, content in current_files.items():
            ext = os.path.splitext(rel_path)[1]
            language = LANGUAGE_EXTENSIONS.get(ext)
            if not language:
                continue
            try:
                file_symbols = [s for s in all_symbols if s.file == rel_path]
                all_refs.extend(extract_refs(content, rel_path, language, file_symbols))
            except Exception:
                pass
        store.save_refs(owner, repo_name, all_refs)

        result = {
            "success": True,
            "repo": f"{owner}/{repo_name}",
            "folder_path": str(folder_path),
            "indexed_at": saved_index.indexed_at,
            "file_count": len(parsed_files),
            "symbol_count": len(all_symbols),
            "ref_count": len(all_refs),
            "languages": languages,
            "files": parsed_files[:20],  # Limit files in response
            "discovery_skip_counts": skip_counts,
            "no_symbols_count": len(no_symbols_files),
            "no_symbols_files": no_symbols_files[:50],  # Show up to 50 for inspection
        }

        if warnings:
            result["warnings"] = warnings

        if skip_counts.get("file_limit", 0) > 0:
            result["note"] = f"Folder has many files; indexed first {max_files}"

        return result

    except Exception as e:
        return {"success": False, "error": f"Indexing failed: {str(e)}"}

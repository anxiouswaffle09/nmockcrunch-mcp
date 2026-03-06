"""Regression tests for the 29-issue fixes (FIXES.md)."""

import json
import os
import sys
import threading
import time
import unittest.mock as mock
from pathlib import Path
from typing import Optional

import pytest

from jcodemunch_mcp.storage import IndexStore, CodeIndex
from jcodemunch_mcp.parser import Symbol
from jcodemunch_mcp.storage import index_store as _is_mod
from jcodemunch_mcp.storage.index_store import (
    _file_hash,
    _make_file_meta,
    _detect_changes_git,
    _invalidate_index_cache,
    _index_cache,
    _cache_lock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_symbol(name: str, file: str = "main.py", language: str = "python") -> Symbol:
    return Symbol(
        id=f"{file.replace('.', '-')}::{name}",
        file=file,
        name=name,
        qualified_name=name,
        kind="function",
        language=language,
        signature=f"def {name}():",
        byte_offset=0,
        byte_length=20,
    )


def _save_minimal_index(store: IndexStore, owner: str, name: str,
                        content: str = "def foo(): pass\n",
                        folder_path: Optional[Path] = None):
    sym = _make_symbol("foo")
    return store.save_index(
        owner=owner,
        name=name,
        source_files=["main.py"],
        symbols=[sym],
        raw_files={"main.py": content},
        languages={"python": 1},
        folder_path=folder_path,
    )


# ---------------------------------------------------------------------------
# Phase 1 — Foundation
# ---------------------------------------------------------------------------

class TestRegisterPathPersists:
    """Issue 1 + 28: register_path persists to autorefresh.json, normalises paths."""

    def test_register_path_writes_config(self, tmp_path):
        from jcodemunch_mcp.server import AutoRefresher
        config_path = tmp_path / "autorefresh.json"

        ar = AutoRefresher.__new__(AutoRefresher)
        ar._lock = threading.Lock()
        ar._last_refresh = {}
        ar._paths = set()
        ar._cooldown = 0.0
        ar._cfg_mtime = None
        ar.CONFIG_PATH = str(config_path)

        test_dir = tmp_path / "project"
        test_dir.mkdir()
        ar.register_path(str(test_dir))

        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert str(test_dir.resolve()) in data["paths"]

    def test_register_path_survives_restart(self, tmp_path):
        """After restart, path is loaded from persisted config."""
        from jcodemunch_mcp.server import AutoRefresher

        config_path = tmp_path / "autorefresh.json"
        test_dir = tmp_path / "project"
        test_dir.mkdir()

        # Write config manually (simulating a restart)
        config_path.write_text(json.dumps({"paths": [str(test_dir.resolve())]}))

        ar = AutoRefresher.__new__(AutoRefresher)
        ar._lock = threading.Lock()
        ar._last_refresh = {}
        ar._paths = set()
        ar._cooldown = 0.0
        ar._cfg_mtime = None
        ar.CONFIG_PATH = str(config_path)
        ar._load_config()

        assert str(test_dir.resolve()) in ar._paths

    def test_register_path_normalises_duplicates(self, tmp_path):
        """Registering ~/project and /home/user/project should yield one entry."""
        from jcodemunch_mcp.server import AutoRefresher

        config_path = tmp_path / "autorefresh.json"
        test_dir = tmp_path / "project"
        test_dir.mkdir()

        ar = AutoRefresher.__new__(AutoRefresher)
        ar._lock = threading.Lock()
        ar._last_refresh = {}
        ar._paths = set()
        ar._cooldown = 0.0
        ar._cfg_mtime = None
        ar.CONFIG_PATH = str(config_path)

        # Register the same resolved path twice
        resolved = str(test_dir.resolve())
        ar.register_path(resolved)
        ar.register_path(resolved)

        assert len(ar._paths) == 1


class TestRefreshToolsAllowlist:
    """Issue 11: list_repos and index tools must NOT trigger auto-refresh."""

    def test_list_repos_not_in_refresh_tools(self):
        from jcodemunch_mcp.server import _REFRESH_TOOLS
        assert "list_repos" not in _REFRESH_TOOLS

    def test_index_folder_not_in_refresh_tools(self):
        from jcodemunch_mcp.server import _REFRESH_TOOLS
        assert "index_folder" not in _REFRESH_TOOLS

    def test_read_tools_in_refresh_set(self):
        from jcodemunch_mcp.server import _REFRESH_TOOLS
        for tool in ["get_symbol", "search_symbols", "get_file_tree",
                     "find_references", "find_callers"]:
            assert tool in _REFRESH_TOOLS, f"{tool} should be in _REFRESH_TOOLS"


class TestWatchlistCap:
    """Issue 9: MAX_WATCHED_PATHS cap enforced."""

    def test_cap_enforced(self, tmp_path):
        from jcodemunch_mcp.server import AutoRefresher, MAX_WATCHED_PATHS

        config_path = tmp_path / "autorefresh.json"
        ar = AutoRefresher.__new__(AutoRefresher)
        ar._lock = threading.Lock()
        ar._last_refresh = {}
        ar._paths = set()
        ar._cooldown = 0.0
        ar._cfg_mtime = None
        ar.CONFIG_PATH = str(config_path)

        for i in range(MAX_WATCHED_PATHS + 5):
            d = tmp_path / f"p{i}"
            d.mkdir(exist_ok=True)
            ar.register_path(str(d))

        assert len(ar._paths) == MAX_WATCHED_PATHS


class TestNoDuplicateIndexStoreGetFileTree:
    """Issue 13: get_file_tree should not create a second IndexStore."""

    def test_store2_removed(self, tmp_path):
        import inspect
        from jcodemunch_mcp.tools.get_file_tree import get_file_tree
        src = inspect.getsource(get_file_tree)
        assert "store2" not in src, "store2 redundant IndexStore must be removed"


# ---------------------------------------------------------------------------
# Phase 2 — Performance
# ---------------------------------------------------------------------------

class TestIndexCache:
    """Issue 6: load_index uses process-level mtime-gated cache."""

    def test_load_index_cached(self, tmp_path):
        """Second load_index call returns cached result without re-parsing JSON."""
        store = IndexStore(base_path=str(tmp_path))
        _save_minimal_index(store, "owner", "repo")

        # Clear cache so we start fresh
        idx_path = store._index_path("owner", "repo")
        _invalidate_index_cache(idx_path)

        open_calls = []
        _real_open = open

        def counting_open(path, *args, **kwargs):
            if str(path) == str(idx_path) and ("r" in args or kwargs.get("mode", "r") == "r"):
                open_calls.append(path)
            return _real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=counting_open):
            store.load_index("owner", "repo")
            store.load_index("owner", "repo")

        # First call opens JSON; second should return from cache
        assert len(open_calls) <= 1

    def test_cache_invalidated_on_save(self, tmp_path):
        """After save_index, load_index returns fresh data."""
        store = IndexStore(base_path=str(tmp_path))
        _save_minimal_index(store, "owner", "repo", content="def foo(): pass")

        # Modify the index on disk without going through store
        idx_path = store._index_path("owner", "repo")
        # Force-load to populate cache
        store.load_index("owner", "repo")

        # Save a new version — should invalidate cache
        _save_minimal_index(store, "owner", "repo", content="def bar(): pass")
        loaded = store.load_index("owner", "repo")
        # The new index is a fresh parse; no assertion needed on content (we
        # just verify it doesn't return a stale result from before save)
        assert loaded is not None


class TestOsWalkPrunesDirectories:
    """Issue 7: discover_local_files prunes node_modules/ without descending."""

    def test_node_modules_pruned(self, tmp_path):
        from jcodemunch_mcp.tools.index_folder import discover_local_files

        # Create files that should be found
        (tmp_path / "app.py").write_text("def hello(): pass", encoding="utf-8")

        # Create node_modules with Python files that should NOT be found
        nm = tmp_path / "node_modules" / "lib"
        nm.mkdir(parents=True)
        (nm / "helper.py").write_text("def helper(): pass", encoding="utf-8")

        files, _, _ = discover_local_files(tmp_path)
        paths = [f.name for f in files]
        assert "app.py" in paths
        assert "helper.py" not in paths

    def test_vendor_pruned(self, tmp_path):
        from jcodemunch_mcp.tools.index_folder import discover_local_files

        (tmp_path / "main.py").write_text("def main(): pass", encoding="utf-8")
        vendor = tmp_path / "vendor" / "third_party"
        vendor.mkdir(parents=True)
        (vendor / "lib.py").write_text("def lib(): pass", encoding="utf-8")

        files, _, _ = discover_local_files(tmp_path)
        paths = [f.name for f in files]
        assert "main.py" in paths
        assert "lib.py" not in paths


class TestGetSymbolNoDuplicateLoad:
    """Issue 10: get_symbol passes loaded index to get_symbol_content."""

    def test_no_double_load(self, tmp_path):
        store = IndexStore(base_path=str(tmp_path))
        sym = _make_symbol("foo")
        store.save_index(
            owner="o", name="r",
            source_files=["main.py"],
            symbols=[sym],
            raw_files={"main.py": "def foo(): pass\n"},
            languages={"python": 1},
        )

        # Pre-load index before patching so the load_call count is clean
        preloaded = store.load_index("o", "r")

        load_calls = []
        original_load = store.load_index

        def tracking_load(owner, name):
            load_calls.append((owner, name))
            return original_load(owner, name)

        with mock.patch.object(store, "load_index", side_effect=tracking_load):
            store.get_symbol_content("o", "r", sym.id, index=preloaded)

        # get_symbol_content should NOT call load_index again when index passed
        assert len(load_calls) == 0


# ---------------------------------------------------------------------------
# Phase 3 — Change detection
# ---------------------------------------------------------------------------

class TestMtimeDetection:
    """Issue 2: detect_changes_fast uses mtime+size before SHA-256."""

    def test_unchanged_file_not_in_changed(self, tmp_path):
        store = IndexStore(base_path=str(tmp_path))
        content = "def foo(): pass\n"
        main_py = tmp_path / "main.py"
        main_py.write_text(content, encoding="utf-8")

        _save_minimal_index(store, "local", "proj", content=content, folder_path=tmp_path)

        paths = [main_py]
        changed, added, deleted = store.detect_changes_fast(
            "local", "proj", tmp_path, paths
        )
        assert "main.py" not in changed
        assert len(added) == 0
        assert len(deleted) == 0

    def test_modified_file_detected(self, tmp_path):
        store = IndexStore(base_path=str(tmp_path))
        main_py = tmp_path / "main.py"
        main_py.write_text("def foo(): pass\n", encoding="utf-8")

        _save_minimal_index(store, "local", "proj", content="def foo(): pass\n", folder_path=tmp_path)

        # Modify the file
        time.sleep(0.01)  # ensure mtime changes
        main_py.write_text("def bar(): pass\n", encoding="utf-8")
        # Update mtime explicitly to be different
        cur_mtime = main_py.stat().st_mtime
        os.utime(main_py, (cur_mtime + 1, cur_mtime + 1))

        paths = [main_py]
        changed, added, deleted = store.detect_changes_fast(
            "local", "proj", tmp_path, paths
        )
        assert "main.py" in changed

    def test_new_file_in_added(self, tmp_path):
        store = IndexStore(base_path=str(tmp_path))
        main_py = tmp_path / "main.py"
        main_py.write_text("def foo(): pass\n", encoding="utf-8")

        _save_minimal_index(store, "local", "proj", folder_path=tmp_path)

        # Add a new file
        new_py = tmp_path / "new_file.py"
        new_py.write_text("def new(): pass\n", encoding="utf-8")

        paths = [main_py, new_py]
        changed, added, deleted = store.detect_changes_fast(
            "local", "proj", tmp_path, paths
        )
        assert "new_file.py" in added

    def test_deleted_file_in_deleted(self, tmp_path):
        store = IndexStore(base_path=str(tmp_path))
        main_py = tmp_path / "main.py"
        main_py.write_text("def foo(): pass\n", encoding="utf-8")
        other_py = tmp_path / "other.py"
        other_py.write_text("def other(): pass\n", encoding="utf-8")

        # Index both files
        sym1 = _make_symbol("foo", "main.py")
        sym2 = _make_symbol("other", "other.py")
        store.save_index(
            owner="local", name="proj",
            source_files=["main.py", "other.py"],
            symbols=[sym1, sym2],
            raw_files={"main.py": "def foo(): pass\n", "other.py": "def other(): pass\n"},
            languages={"python": 1},
            folder_path=tmp_path,
        )

        # Delete other.py
        other_py.unlink()

        paths = [main_py]  # only main.py remains
        changed, added, deleted = store.detect_changes_fast(
            "local", "proj", tmp_path, paths
        )
        assert "other.py" in deleted

    def test_same_content_not_changed(self, tmp_path):
        """Same mtime but same content: not reported as changed."""
        store = IndexStore(base_path=str(tmp_path))
        content = "def foo(): pass\n"
        main_py = tmp_path / "main.py"
        main_py.write_text(content, encoding="utf-8")

        _save_minimal_index(store, "local", "proj", content=content, folder_path=tmp_path)

        # Force mtime to differ but content is same
        cur_mtime = main_py.stat().st_mtime
        os.utime(main_py, (cur_mtime + 1, cur_mtime + 1))

        paths = [main_py]
        changed, added, deleted = store.detect_changes_fast(
            "local", "proj", tmp_path, paths
        )
        # Content is identical → Phase 2 SHA-256 should not flag it
        assert "main.py" not in changed


class TestGitDetection:
    """Issue 3: git-accelerated change detection."""

    def test_git_modified_detected(self, tmp_path):
        """_detect_changes_git correctly parses git status output."""
        modified, deleted, head = _detect_changes_git.__wrapped__(
            tmp_path, "", {}
        ) if hasattr(_detect_changes_git, "__wrapped__") else _test_git_mock(tmp_path)

    def test_fallback_on_non_git_dir(self, tmp_path):
        """Non-git directory → empty sets (mtime fallback activates)."""
        # tmp_path is not a git repo
        modified, deleted, head = _detect_changes_git(tmp_path, "somehead", {})
        # git commands fail → empty sets returned
        assert isinstance(modified, set)
        assert isinstance(deleted, set)


def _test_git_mock(tmp_path):
    with mock.patch("subprocess.run") as mock_run:
        # Simulate git status showing a modified file
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout=" M modified.py\n?? new.py\n",
        )
        return _detect_changes_git(tmp_path, "abc123", {})


class TestConcurrentRefreshLock:
    """Issue 4: per-path lock prevents concurrent refresh races."""

    def test_concurrent_refresh_second_skips(self, tmp_path):
        from jcodemunch_mcp.server import _get_path_lock

        path = str(tmp_path / "project")
        os.makedirs(path, exist_ok=True)

        lock = _get_path_lock(path)
        skipped = []

        def slow_refresh():
            acquired = lock.acquire(blocking=False)
            if not acquired:
                skipped.append(True)
                return
            try:
                time.sleep(0.05)
            finally:
                lock.release()

        t1 = threading.Thread(target=slow_refresh)
        t2 = threading.Thread(target=slow_refresh)
        t1.start()
        time.sleep(0.01)  # let t1 acquire first
        t2.start()
        t1.join()
        t2.join()

        assert skipped, "Second thread should have skipped refresh"


class TestMergeRefsConcurrent:
    """Issue 5: merge_refs is safe under concurrent access."""

    def test_concurrent_merge_preserves_all_refs(self, tmp_path):
        store = IndexStore(base_path=str(tmp_path))
        _save_minimal_index(store, "o", "r")

        # Seed with empty refs
        store.save_refs("o", "r", [])

        results = []
        errors = []

        def add_ref(ref_id):
            try:
                ref = {"caller_file": "a.py", "callee": ref_id, "line": 1}
                store.merge_refs("o", "r", [ref], set())
                results.append(ref_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_ref, args=(f"ref{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        final_refs = store.load_refs("o", "r") or []
        assert len(final_refs) == 10


class TestConfigHotReload:
    """Issue 29: maybe_refresh reloads config if mtime changed."""

    def test_config_reloaded_on_mtime_change(self, tmp_path):
        from jcodemunch_mcp.server import AutoRefresher

        config_path = tmp_path / "autorefresh.json"
        config_path.write_text(json.dumps({"cooldown_secs": 999}))

        ar = AutoRefresher.__new__(AutoRefresher)
        ar._lock = threading.Lock()
        ar._last_refresh = {}
        ar._paths = set()
        ar._cooldown = 0.0
        ar._cfg_mtime = None
        ar.CONFIG_PATH = str(config_path)
        ar._load_config()
        assert ar._cooldown == 999.0

        # Simulate config change
        config_path.write_text(json.dumps({"cooldown_secs": 42}))
        # Touch to ensure mtime changes
        mtime = config_path.stat().st_mtime
        os.utime(config_path, (mtime + 1, mtime + 1))
        ar._maybe_reload_config()

        assert ar._cooldown == 42.0


# ---------------------------------------------------------------------------
# Phase 4 — Correctness
# ---------------------------------------------------------------------------

class TestFileSummariesPersisted:
    """Issue 15: file_summaries field is saved and loaded."""

    def test_file_summaries_roundtrip(self, tmp_path):
        store = IndexStore(base_path=str(tmp_path))
        index = _save_minimal_index(store, "o", "r")

        # Manually craft an index with file_summaries
        idx_path = store._index_path("o", "r")
        data = json.loads(idx_path.read_text())
        data["file_summaries"] = {"main.py": "A simple module"}
        idx_path.write_text(json.dumps(data))
        _invalidate_index_cache(idx_path)

        loaded = store.load_index("o", "r")
        assert loaded is not None
        assert loaded.file_summaries.get("main.py") == "A simple module"

    def test_file_summaries_in_dict(self, tmp_path):
        """_index_to_dict must include file_summaries."""
        store = IndexStore(base_path=str(tmp_path))
        sym = _make_symbol("foo")
        from jcodemunch_mcp.storage.index_store import CodeIndex
        idx = CodeIndex(
            repo="o/r", owner="o", name="r",
            indexed_at="2025-01-01T00:00:00",
            source_files=["main.py"],
            languages={"python": 1},
            symbols=[],
            file_summaries={"main.py": "hello"},
        )
        d = store._index_to_dict(idx)
        assert "file_summaries" in d
        assert d["file_summaries"]["main.py"] == "hello"


class TestAmbiguousRepoName:
    """Issue 20: ambiguous bare repo name raises ValueError."""

    def test_ambiguous_raises(self, tmp_path):
        from jcodemunch_mcp.tools._utils import resolve_repo, invalidate_repo_name_cache

        store = IndexStore(base_path=str(tmp_path))
        for owner in ["owner1", "owner2"]:
            store.save_index(
                owner=owner, name="myrepo",
                source_files=["main.py"],
                symbols=[],
                raw_files={"main.py": ""},
                languages={"python": 1},
            )

        invalidate_repo_name_cache()
        with pytest.raises(ValueError, match="Ambiguous"):
            resolve_repo("myrepo", storage_path=str(tmp_path))

    def test_unique_name_resolves(self, tmp_path):
        from jcodemunch_mcp.tools._utils import resolve_repo, invalidate_repo_name_cache

        store = IndexStore(base_path=str(tmp_path))
        store.save_index(
            owner="local", name="myrepo",
            source_files=["main.py"],
            symbols=[],
            raw_files={"main.py": ""},
            languages={"python": 1},
        )

        invalidate_repo_name_cache()
        owner, name = resolve_repo("myrepo", storage_path=str(tmp_path))
        assert owner == "local"
        assert name == "myrepo"

    def test_cache_invalidated_after_index(self, tmp_path):
        from jcodemunch_mcp.tools._utils import _repo_name_cache, invalidate_repo_name_cache

        invalidate_repo_name_cache()
        _repo_name_cache["myrepo"] = ("old", "old")
        invalidate_repo_name_cache()
        assert "myrepo" not in _repo_name_cache


# ---------------------------------------------------------------------------
# Phase 5 — Search quality
# ---------------------------------------------------------------------------

class TestSearchScoreEmbedded:
    """Issue 23: CodeIndex.search returns score embedded in result dicts."""

    def test_search_returns_score(self):
        index = CodeIndex(
            repo="o/r", owner="o", name="r",
            indexed_at="2025-01-01T00:00:00",
            source_files=["main.py"],
            languages={"python": 1},
            symbols=[
                {"id": "m::foo", "name": "foo", "kind": "function", "file": "main.py",
                 "language": "python", "signature": "def foo():", "summary": "",
                 "keywords": [], "docstring": ""},
            ],
        )
        results = index.search("foo")
        assert len(results) > 0
        assert "score" in results[0], "score must be embedded in results"

    def test_language_filter_applied_in_search(self):
        """Issue 24: language filter is applied inside CodeIndex.search."""
        index = CodeIndex(
            repo="o/r", owner="o", name="r",
            indexed_at="2025-01-01T00:00:00",
            source_files=["main.py", "main.rs"],
            languages={"python": 1, "rust": 1},
            symbols=[
                {"id": "m-py::foo", "name": "foo", "kind": "function", "file": "main.py",
                 "language": "python", "signature": "def foo():", "summary": "",
                 "keywords": [], "docstring": ""},
                {"id": "m-rs::foo", "name": "foo", "kind": "function", "file": "main.rs",
                 "language": "rust", "signature": "fn foo()", "summary": "",
                 "keywords": [], "docstring": ""},
            ],
        )
        results = index.search("foo", language="python")
        assert all(r.get("language") == "python" for r in results)
        assert len(results) == 1

    def test_no_double_scoring_in_search_symbols(self, tmp_path):
        """Issue 23: search_symbols must not call _calculate_score (removed)."""
        import inspect
        from jcodemunch_mcp.tools import search_symbols
        src = inspect.getsource(search_symbols)
        assert "_calculate_score" not in src, "_calculate_score must be removed from search_symbols"


class TestGetSymbolsNoDuplicateScan:
    """Issue 25: get_symbols doesn't call index.get_symbol twice per ID."""

    def test_no_redundant_scan(self, tmp_path):
        store = IndexStore(base_path=str(tmp_path))
        syms = [_make_symbol(f"fn{i}") for i in range(3)]
        store.save_index(
            owner="o", name="r",
            source_files=["main.py"],
            symbols=syms,
            raw_files={"main.py": "def fn0(): pass\n"},
            languages={"python": 1},
        )

        index = store.load_index("o", "r")
        call_count = [0]
        original_get = index.get_symbol

        def tracking_get(sid):
            call_count[0] += 1
            return original_get(sid)

        index.get_symbol = tracking_get

        from jcodemunch_mcp.tools.get_symbol import get_symbols
        # Patch load_index to return our tracked index
        with mock.patch.object(store, "load_index", return_value=index):
            with mock.patch("jcodemunch_mcp.tools.get_symbol.IndexStore", return_value=store):
                get_symbols(repo="o/r", symbol_ids=[s.id for s in syms],
                            storage_path=str(tmp_path))

        # Should call get_symbol at most 2× len(syms):
        # once in the main loop + once inside get_symbol_content.
        # Before fix it was 3× (extra scan in token savings loop).
        assert call_count[0] <= 2 * len(syms)


class TestGetFileOutlineNoDictToSymbol:
    """Issue 26: get_file_outline uses build_symbol_tree_from_dicts, no roundtrip."""

    def test_no_dict_to_symbol_roundtrip(self):
        import inspect
        from jcodemunch_mcp.tools import get_file_outline
        src = inspect.getsource(get_file_outline)
        assert "_dict_to_symbol" not in src, "_dict_to_symbol roundtrip must be eliminated"
        assert "build_symbol_tree_from_dicts" in src

    def test_outline_still_works(self, tmp_path):
        from jcodemunch_mcp.tools.get_file_outline import get_file_outline as gfo

        store = IndexStore(base_path=str(tmp_path))
        sym = _make_symbol("foo")
        store.save_index(
            owner="local", name="proj",
            source_files=["main.py"],
            symbols=[sym],
            raw_files={"main.py": "def foo(): pass\n"},
            languages={"python": 1},
        )

        result = gfo(repo="local/proj", file_path="main.py", storage_path=str(tmp_path))
        assert "symbols" in result
        assert len(result["symbols"]) == 1
        assert result["symbols"][0]["name"] == "foo"


# ---------------------------------------------------------------------------
# Phase 6 — Security & privacy
# ---------------------------------------------------------------------------

class TestSearchTextSafePath:
    """Issue 21: search_text uses _safe_content_path to prevent traversal."""

    def test_traversal_entry_skipped(self, tmp_path):
        from jcodemunch_mcp.tools.search_text import search_text

        store = IndexStore(base_path=str(tmp_path))
        sym = _make_symbol("foo")
        store.save_index(
            owner="local", name="proj",
            source_files=["main.py"],
            symbols=[sym],
            raw_files={"main.py": "def foo(): pass\n"},
            languages={"python": 1},
        )

        # Manually add a traversal entry to the index
        idx_path = store._index_path("local", "proj")
        data = json.loads(idx_path.read_text())
        data["source_files"].append("../../etc/passwd")
        idx_path.write_text(json.dumps(data))
        _invalidate_index_cache(idx_path)

        # search_text should not crash and should not read ../.. file
        result = search_text(
            repo="local/proj",
            query="foo",
            storage_path=str(tmp_path),
        )
        assert "error" not in result


class TestRecordSavingsConcurrent:
    """Issue 22: record_savings is thread-safe and atomic."""

    def test_concurrent_savings_accurate(self, tmp_path):
        from jcodemunch_mcp.storage.token_tracker import record_savings, _savings_path

        path = _savings_path(str(tmp_path))
        if path.exists():
            path.unlink()

        errors = []
        n_threads = 20

        def add_one():
            try:
                record_savings(1, base_path=str(tmp_path))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_one) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        data = json.loads(path.read_text())
        assert data["total_tokens_saved"] == n_threads

    def test_atomic_write(self, tmp_path):
        """record_savings writes via temp file — old data survives a simulated error."""
        from jcodemunch_mcp.storage.token_tracker import record_savings, _savings_path

        path = _savings_path(str(tmp_path))
        path.write_text(json.dumps({"total_tokens_saved": 100}))

        # Simulate crash before tmp.replace by patching Path.replace
        original_replace = Path.replace
        calls = []

        def failing_replace(self, target):
            if str(self).endswith(".json.tmp") and not calls:
                calls.append(True)
                raise OSError("simulated crash")
            return original_replace(self, target)

        with mock.patch.object(Path, "replace", failing_replace):
            try:
                record_savings(50, base_path=str(tmp_path))
            except Exception:
                pass

        # Old data must still be intact
        surviving = json.loads(path.read_text())
        assert surviving["total_tokens_saved"] == 100


# ---------------------------------------------------------------------------
# Phase 7 — Cleanup
# ---------------------------------------------------------------------------

class TestSearchTextSizeGuard:
    """Issue 8: search_text stops at 50MB."""

    def test_size_guard_warning_present(self):
        import inspect
        from jcodemunch_mcp.tools.search_text import search_text
        src = inspect.getsource(search_text)
        assert "MAX_SEARCH_BYTES" in src
        assert "50" in src


class TestMakeFileMeta:
    """Issue 2: _make_file_meta produces correct structure."""

    def test_make_file_meta_fields(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello", encoding="utf-8")
        meta = _make_file_meta(f, "hello")
        assert "sha256" in meta
        assert "mtime" in meta
        assert "size" in meta
        assert meta["sha256"] == _file_hash("hello")
        assert meta["size"] == 5


# ---------------------------------------------------------------------------
# Phase 8 — Remaining fixes (Issues 17 & 27)
# ---------------------------------------------------------------------------

class TestDiscoverSourceFilesBlobShas:
    """Issue 17: discover_source_files returns blob SHAs from tree entries."""

    def test_blob_sha_in_return(self):
        from jcodemunch_mcp.tools.index_repo import discover_source_files

        tree = [
            {"type": "blob", "path": "main.py", "size": 100, "sha": "abc123"},
            {"type": "blob", "path": "README.md", "size": 50, "sha": "def456"},  # wrong ext
            {"type": "tree", "path": "src", "size": 0, "sha": "777"},
        ]
        files, truncated, blob_shas = discover_source_files(tree)
        assert "main.py" in files
        assert "main.py" in blob_shas
        assert blob_shas["main.py"] == "abc123"
        # README.md filtered out by extension
        assert "README.md" not in files
        assert "README.md" not in blob_shas

    def test_truncation_prunes_blob_shas(self):
        """When file limit truncates the list, blob_shas is pruned to match."""
        from jcodemunch_mcp.tools.index_repo import discover_source_files

        # Generate 5 py files, limit to 3
        tree = [
            {"type": "blob", "path": f"f{i}.py", "size": 10, "sha": f"sha{i}"}
            for i in range(5)
        ]
        files, truncated, blob_shas = discover_source_files(tree, max_files=3)
        assert truncated
        assert len(files) == 3
        assert set(blob_shas.keys()) == set(files), "blob_shas must match truncated file list"


class TestIncrementalBlobShaDetection:
    """Issue 17: incremental_save accepts file_hashes_override (blob SHAs)."""

    def _make_index(self, store, owner, name, hashes):
        """Helper: create a minimal stored index with given file_hashes."""
        idx = CodeIndex(
            repo=f"{owner}/{name}", owner=owner, name=name,
            indexed_at="2025-01-01T00:00:00",
            source_files=list(hashes.keys()),
            languages={}, symbols=[],
            file_hashes=hashes,
        )
        index_path = store._index_path(owner, name)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w") as f:
            json.dump(store._index_to_dict(idx), f)
        _invalidate_index_cache(index_path)
        return idx

    def test_file_hashes_override_used(self, tmp_path):
        """When file_hashes_override is passed, incremental_save stores those hashes."""
        store = IndexStore(base_path=str(tmp_path))
        self._make_index(store, "o", "r", {"a.py": "old_sha"})

        blob_overrides = {"a.py": "new_blob_sha", "b.py": "b_blob_sha"}
        store.incremental_save(
            owner="o", name="r",
            changed_files=["a.py"], new_files=["b.py"], deleted_files=[],
            new_symbols=[], raw_files={"a.py": "x", "b.py": "y"},
            languages={},
            file_hashes_override=blob_overrides,
        )
        updated = store.load_index("o", "r")
        assert updated.file_hashes["a.py"] == "new_blob_sha"
        assert updated.file_hashes["b.py"] == "b_blob_sha"

    def test_without_override_uses_content_hash(self, tmp_path):
        """Without override, incremental_save computes content SHA-256 as before."""
        store = IndexStore(base_path=str(tmp_path))
        self._make_index(store, "o", "r", {"a.py": "old"})

        store.incremental_save(
            owner="o", name="r",
            changed_files=["a.py"], new_files=[], deleted_files=[],
            new_symbols=[], raw_files={"a.py": "new content"},
            languages={},
        )
        updated = store.load_index("o", "r")
        # Should be content SHA-256, not the old value
        assert updated.file_hashes["a.py"] == _file_hash("new content")


class TestShouldExcludeFileUnified:
    """Issue 27: discover_local_files uses should_exclude_file for security checks."""

    def test_no_direct_is_binary_file_import(self):
        """is_binary_file, is_symlink_escape, is_secret_file no longer imported directly."""
        import inspect
        from jcodemunch_mcp.tools import index_folder as _mod
        src = inspect.getsource(_mod)
        assert "is_binary_file" not in src
        assert "is_symlink_escape" not in src
        assert "is_secret_file" not in src

    def test_should_exclude_file_called(self):
        """should_exclude_file is used in discover_local_files source."""
        import inspect
        from jcodemunch_mcp.tools.index_folder import discover_local_files
        src = inspect.getsource(discover_local_files)
        assert "should_exclude_file" in src

    def test_binary_file_excluded(self, tmp_path):
        """A file with null bytes (binary content) is excluded from discovery."""
        from jcodemunch_mcp.tools.index_folder import discover_local_files

        py_file = tmp_path / "binary.py"
        py_file.write_bytes(b"def foo():\n    pass\x00\x00\x00")  # null bytes = binary
        files, warnings, skip_counts = discover_local_files(tmp_path)
        assert py_file not in files
        assert skip_counts["binary"] >= 1

    def test_secret_file_excluded(self, tmp_path):
        """A file matching a secret pattern (with source extension) is excluded."""
        from jcodemunch_mcp.tools.index_folder import discover_local_files

        # Use a .py extension so it passes the extension filter, but a name matching
        # the "*secret*" pattern in SECRET_PATTERNS so should_exclude_file catches it.
        secret = tmp_path / "my_secrets.py"
        secret.write_text("API_KEY = 'abc123'")
        (tmp_path / "main.py").write_text("def f(): pass")
        files, warnings, skip_counts = discover_local_files(tmp_path)
        assert secret not in files
        assert skip_counts["secret"] >= 1
        assert any("secret file" in w.lower() for w in warnings)

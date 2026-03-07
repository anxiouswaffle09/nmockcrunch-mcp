"""Tests for incremental indexing via index_folder."""

import pytest
from pathlib import Path

from jcodemunch_mcp.tools.index_folder import index_folder
from jcodemunch_mcp.storage import IndexStore


def _write_py(d: Path, name: str, content: str) -> Path:
    """Write a Python file into directory d."""
    p = d / name
    p.write_text(content, encoding="utf-8")
    return p


def _write_file(d: Path, name: str, content: str) -> Path:
    """Write an arbitrary source file into directory d."""
    p = d / name
    p.write_text(content, encoding="utf-8")
    return p


class TestIncrementalIndexFolder:
    """Test incremental indexing through index_folder."""

    def test_full_index_then_incremental_no_changes(self, tmp_path):
        """Incremental re-index with no changes returns early."""
        src = tmp_path / "src"
        src.mkdir()
        store = tmp_path / "store"

        _write_py(src, "hello.py", "def hello():\n    return 'hi'\n")
        _write_py(src, "world.py", "def world():\n    return 'earth'\n")

        result = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
        assert result["success"] is True
        assert result["symbol_count"] == 2

        # Incremental with no changes
        result2 = index_folder(
            str(src), use_ai_summaries=False, storage_path=str(store), incremental=True
        )
        assert result2["success"] is True
        assert result2["message"] == "No changes detected"
        assert result2["changed"] == 0
        assert result2["new"] == 0
        assert result2["deleted"] == 0

    def test_incremental_detects_modified_file(self, tmp_path):
        """Incremental re-index detects a modified file."""
        src = tmp_path / "src"
        src.mkdir()
        store = tmp_path / "store"

        _write_py(src, "calc.py", "def add(a, b):\n    return a + b\n")
        _write_py(src, "util.py", "def noop():\n    pass\n")

        result = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
        assert result["success"] is True
        original_count = result["symbol_count"]

        # Modify one file: change body and add a function
        _write_py(src, "calc.py", "def add(a, b):\n    return a + b + 1\n\ndef sub(a, b):\n    return a - b\n")

        result2 = index_folder(
            str(src), use_ai_summaries=False, storage_path=str(store), incremental=True
        )
        assert result2["success"] is True
        assert result2["incremental"] is True
        assert result2["changed"] == 1
        assert result2["new"] == 0
        assert result2["deleted"] == 0
        # Should have original symbols + 1 new (sub added)
        assert result2["symbol_count"] == original_count + 1

    def test_incremental_detects_new_file(self, tmp_path):
        """Incremental re-index detects a new file."""
        src = tmp_path / "src"
        src.mkdir()
        store = tmp_path / "store"

        _write_py(src, "a.py", "def func_a():\n    pass\n")

        index_folder(str(src), use_ai_summaries=False, storage_path=str(store))

        # Add a new file
        _write_py(src, "b.py", "def func_b():\n    return 42\n")

        result = index_folder(
            str(src), use_ai_summaries=False, storage_path=str(store), incremental=True
        )
        assert result["success"] is True
        assert result["new"] == 1
        assert result["symbol_count"] == 2

    def test_incremental_detects_deleted_file(self, tmp_path):
        """Incremental re-index detects a deleted file."""
        src = tmp_path / "src"
        src.mkdir()
        store = tmp_path / "store"

        _write_py(src, "keep.py", "def keep():\n    pass\n")
        _write_py(src, "remove.py", "def remove():\n    pass\n")

        index_folder(str(src), use_ai_summaries=False, storage_path=str(store))

        # Delete one file
        (src / "remove.py").unlink()

        result = index_folder(
            str(src), use_ai_summaries=False, storage_path=str(store), incremental=True
        )
        assert result["success"] is True
        assert result["deleted"] == 1
        assert result["symbol_count"] == 1

    def test_incremental_false_does_full_reindex(self, tmp_path):
        """With incremental=False (default), a full re-index is performed."""
        src = tmp_path / "src"
        src.mkdir()
        store = tmp_path / "store"

        _write_py(src, "mod.py", "def original():\n    pass\n")

        result = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
        assert result["success"] is True

        # Full re-index (default) should not have incremental key
        result2 = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
        assert result2["success"] is True
        assert "incremental" not in result2

    def test_incremental_reclassifies_h_language_from_c_to_cpp(self, tmp_path):
        """Changing .h from C-like to C++-like should update persisted language counts."""
        src = tmp_path / "src"
        src.mkdir()
        store = tmp_path / "store"

        _write_file(src, "api.h", "int only_c(void) { int v[] = (int[]){1,2,3}; return v[0]; }\n")

        full = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
        assert full["success"] is True
        assert full["languages"] == {"c": 1}

        _write_file(
            src,
            "api.h",
            "namespace demo { class Widget { public: int Get() const; }; }\n",
        )

        inc = index_folder(str(src), use_ai_summaries=False, storage_path=str(store), incremental=True)
        assert inc["success"] is True
        assert inc["changed"] == 1

        idx = IndexStore(base_path=str(store)).load_index("local", src.name)
        assert idx is not None
        assert idx.languages.get("cpp") == 1
        assert "c" not in idx.languages

    def test_incremental_delete_and_readd_h_updates_language_counts(self, tmp_path):
        """Deleting and re-adding .h with different style should keep language counts correct."""
        src = tmp_path / "src"
        src.mkdir()
        store = tmp_path / "store"

        _write_file(src, "api.h", "int only_c(void) { int v[] = (int[]){1,2,3}; return v[0]; }\n")
        _write_file(src, "main.cpp", "namespace demo { int add(int a, int b) { return a + b; } }\n")

        full = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
        assert full["success"] is True
        assert full["languages"] == {"c": 1, "cpp": 1}

        (src / "api.h").unlink()
        inc_delete = index_folder(str(src), use_ai_summaries=False, storage_path=str(store), incremental=True)
        assert inc_delete["success"] is True
        assert inc_delete["deleted"] == 1

        idx_after_delete = IndexStore(base_path=str(store)).load_index("local", src.name)
        assert idx_after_delete is not None
        assert idx_after_delete.languages == {"cpp": 1}

        _write_file(src, "api.h", "namespace demo { class Readded { public: int Go() const; }; }\n")
        inc_add = index_folder(str(src), use_ai_summaries=False, storage_path=str(store), incremental=True)
        assert inc_add["success"] is True
        assert inc_add["new"] == 1

        idx_after_add = IndexStore(base_path=str(store)).load_index("local", src.name)
        assert idx_after_add is not None
        assert idx_after_add.languages == {"cpp": 2}

    def test_incremental_no_symbol_file_not_repeatedly_new(self, tmp_path):
        """No-symbol files should not be repeatedly reported as new across incremental runs."""
        src = tmp_path / "src"
        src.mkdir()
        store = tmp_path / "store"

        _write_file(src, "main.cpp", "int main() { return 0; }\n")
        _write_file(src, "no_symbols.h", "/* no symbols here */\n")

        full = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
        assert full["success"] is True
        assert full["no_symbols_count"] >= 1

        # No changes should be clean on first incremental run.
        inc1 = index_folder(str(src), use_ai_summaries=False, storage_path=str(store), incremental=True)
        assert inc1["success"] is True
        assert inc1["message"] == "No changes detected"
        assert inc1["changed"] == 0
        assert inc1["new"] == 0
        assert inc1["deleted"] == 0

        # Change the no-symbol file once.
        _write_file(src, "no_symbols.h", "/* still no symbols, but changed */\n")
        inc2 = index_folder(str(src), use_ai_summaries=False, storage_path=str(store), incremental=True)
        assert inc2["success"] is True
        assert inc2["changed"] == 1
        assert inc2["new"] == 0
        assert inc2["deleted"] == 0

        # Next incremental should be clean again (no repeated churn).
        inc3 = index_folder(str(src), use_ai_summaries=False, storage_path=str(store), incremental=True)
        assert inc3["success"] is True
        assert inc3["message"] == "No changes detected"
        assert inc3["changed"] == 0
        assert inc3["new"] == 0
        assert inc3["deleted"] == 0

    def test_incremental_preserves_symbols_when_read_text_fails(self, tmp_path):
        """If read_text fails during the parse loop, existing symbols must not be stripped.

        Simulates a file that passes binary-check (discovery) but fails the full
        content read in the incremental parse loop (e.g. network FS dropping mid-read).
        The binary check uses open()+read(8192), while the parse loop uses read_text(),
        so patching Path.read_text isolates exactly the D1 failure mode.
        """
        from unittest.mock import patch

        src = tmp_path / "src"
        src.mkdir()
        store_path = tmp_path / "store"

        _write_py(src, "a.py", "def foo():\n    pass\n\ndef bar():\n    pass\n")
        _write_py(src, "b.py", "def baz():\n    pass\n")

        # Full index — a.py contributes foo and bar
        result = index_folder(str(src), use_ai_summaries=False, storage_path=str(store_path))
        assert result["success"] is True

        # Write new content so detect_changes_fast marks a.py as changed
        _write_py(src, "a.py", "def foo():\n    return 1\n\ndef bar():\n    pass\n")

        # Patch Path.read_text so it raises for a.py during the parse loop only.
        # detect_changes_fast Phase 2 also calls read_text for SHA verification —
        # that first call must succeed so a.py lands in `changed`, not `deleted`.
        # The second call is from the incremental parse loop: that one we fail.
        original_read_text = Path.read_text
        read_call_count: dict[str, int] = {}

        def failing_read_text(self, *args, **kwargs):
            if self.name == "a.py":
                n = read_call_count.get("a.py", 0) + 1
                read_call_count["a.py"] = n
                if n > 1:  # second call = parse loop; raise to simulate failure
                    raise PermissionError("Simulated read failure for a.py")
            return original_read_text(self, *args, **kwargs)

        with patch.object(Path, "read_text", failing_read_text):
            result2 = index_folder(
                str(src), use_ai_summaries=False, storage_path=str(store_path), incremental=True
            )

        assert result2["success"] is True
        assert result2["incremental"] is True
        # Read failed so a.py was excluded from actual_changed
        assert result2["changed"] == 0
        # A warning must be emitted for the read failure
        assert "warnings" in result2

        # a.py's original symbols (foo, bar) must still be in the index
        store = IndexStore(base_path=str(store_path))
        index = store.load_index("local", src.name)
        assert index is not None
        sym_names = {s["name"] for s in index.symbols}
        assert "foo" in sym_names, "foo was stripped despite read failure"
        assert "bar" in sym_names, "bar was stripped despite read failure"
        assert "baz" in sym_names

    def test_incremental_rebuilds_refs_when_refs_json_missing(self, tmp_path):
        """If refs.json is deleted, the next incremental run backfills it from all current files."""
        src = tmp_path / "src"
        src.mkdir()
        store_path = tmp_path / "store"

        _write_py(src, "a.py", "def foo():\n    pass\n\ndef bar():\n    foo()\n")
        _write_py(src, "b.py", "def baz():\n    pass\n")

        # Full index
        result = index_folder(str(src), use_ai_summaries=False, storage_path=str(store_path))
        assert result["success"] is True

        # Delete refs.json to simulate a missing refs file
        from jcodemunch_mcp.storage import IndexStore
        store = IndexStore(base_path=str(store_path))
        refs_path = store._refs_path("local", src.name)
        refs_path.unlink()
        assert store.load_refs("local", src.name) is None

        # Modify one file to trigger an actual incremental run (not early-exit)
        _write_py(src, "a.py", "def foo():\n    return 1\n\ndef bar():\n    foo()\n")

        result2 = index_folder(
            str(src), use_ai_summaries=False, storage_path=str(store_path), incremental=True
        )
        assert result2["success"] is True
        assert result2["incremental"] is True
        assert result2["changed"] == 1

        # refs.json must have been rebuilt
        rebuilt = store.load_refs("local", src.name)
        assert rebuilt is not None
        assert len(rebuilt) > 0

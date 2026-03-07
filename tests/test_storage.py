"""Tests for storage module."""

import os
import pytest
import json
import time
from pathlib import Path

from jcodemunch_mcp.storage import IndexStore, CodeIndex
from jcodemunch_mcp.parser import Symbol


def test_save_and_load_index(tmp_path):
    """Test saving and loading an index."""
    store = IndexStore(base_path=str(tmp_path))
    
    symbols = [
        Symbol(
            id="test-py::foo",
            file="test.py",
            name="foo",
            qualified_name="foo",
            kind="function",
            language="python",
            signature="def foo():",
            summary="Does foo",
            byte_offset=0,
            byte_length=100,
        )
    ]
    
    index = store.save_index(
        owner="testowner",
        name="testrepo",
        source_files=["test.py"],
        symbols=symbols,
        raw_files={"test.py": "def foo(): pass"},
        languages={"python": 1}
    )
    
    assert index.repo == "testowner/testrepo"
    assert len(index.symbols) == 1
    
    # Load and verify
    loaded = store.load_index("testowner", "testrepo")
    assert loaded is not None
    assert loaded.repo == "testowner/testrepo"
    assert len(loaded.symbols) == 1


def test_byte_offset_retrieval(tmp_path):
    """Test byte-offset content retrieval."""
    store = IndexStore(base_path=str(tmp_path))
    
    content = "line1\nline2\ndef foo():\n    pass\n"
    
    symbols = [
        Symbol(
            id="test-py::foo",
            file="test.py",
            name="foo",
            qualified_name="foo",
            kind="function",
            language="python",
            signature="def foo():",
            byte_offset=12,  # Start of "def foo():"
            byte_length=19,  # Length of "def foo():\n    pass"
        )
    ]
    
    store.save_index(
        owner="testowner",
        name="testrepo",
        source_files=["test.py"],
        symbols=symbols,
        raw_files={"test.py": content},
        languages={"python": 1}
    )
    
    # Retrieve symbol content
    source = store.get_symbol_content("testowner", "testrepo", "test-py::foo")
    assert source is not None
    assert "def foo():" in source


def test_list_repos(tmp_path):
    """Test listing indexed repositories."""
    store = IndexStore(base_path=str(tmp_path))
    
    # Create two indexes
    for owner, name in [("owner1", "repo1"), ("owner2", "repo2")]:
        store.save_index(
            owner=owner,
            name=name,
            source_files=["main.py"],
            symbols=[],
            raw_files={"main.py": ""},
            languages={"python": 1}
        )
    
    repos = store.list_repos()
    assert len(repos) == 2


def test_delete_index(tmp_path):
    """Test deleting an index."""
    store = IndexStore(base_path=str(tmp_path))
    
    store.save_index(
        owner="test",
        name="repo",
        source_files=["main.py"],
        symbols=[],
        raw_files={"main.py": ""},
        languages={"python": 1}
    )
    
    assert store.load_index("test", "repo") is not None
    
    store.delete_index("test", "repo")
    
    assert store.load_index("test", "repo") is None


def test_save_index_rejects_path_traversal_in_raw_files(tmp_path):
    """Raw file cache writes must not escape content dir."""
    store = IndexStore(base_path=str(tmp_path))

    with pytest.raises(ValueError, match="Unsafe file path"):
        store.save_index(
            owner="evil",
            name="repo",
            source_files=["../../escape.py"],
            symbols=[],
            raw_files={"../../escape.py": "print('x')"},
            languages={"python": 1},
        )

    assert store.load_index("evil", "repo") is None


def test_incremental_save_rolls_back_on_invalid_raw_file_path(tmp_path):
    """incremental_save should keep the previous index/content on validation failure."""
    store = IndexStore(base_path=str(tmp_path))

    original = Symbol(
        id="main.py::foo#function",
        file="main.py",
        name="foo",
        qualified_name="foo",
        kind="function",
        language="python",
        signature="def foo():",
        byte_offset=0,
        byte_length=len("def foo():\n    return 1\n".encode("utf-8")),
    )
    store.save_index(
        owner="demo",
        name="repo",
        source_files=["main.py"],
        symbols=[original],
        raw_files={"main.py": "def foo():\n    return 1\n"},
        languages={"python": 1},
    )

    replacement = Symbol(
        id="main.py::bar#function",
        file="main.py",
        name="bar",
        qualified_name="bar",
        kind="function",
        language="python",
        signature="def bar():",
        byte_offset=0,
        byte_length=10,
    )

    with pytest.raises(ValueError, match="Unsafe file path"):
        store.incremental_save(
            owner="demo",
            name="repo",
            changed_files=["main.py"],
            new_files=[],
            deleted_files=[],
            new_symbols=[replacement],
            raw_files={"../escape.py": "def nope():\n    pass\n"},
            languages={},
        )

    loaded = store.load_index("demo", "repo")
    assert loaded is not None
    assert loaded.symbols[0]["name"] == "foo"
    source = store.get_symbol_content("demo", "repo", "main.py::foo#function")
    assert source is not None
    assert "return 1" in source


def test_detect_changes_fast_catches_same_size_edits_with_preserved_mtime(tmp_path):
    """Same-size rewrites with a restored mtime should still be treated as changed."""
    store = IndexStore(base_path=str(tmp_path))
    main_py = tmp_path / "main.py"
    original = "def foo():\n    return 1\n"
    updated = "def bar():\n    return 2\n"
    assert len(original) == len(updated)

    main_py.write_text(original, encoding="utf-8")
    store.save_index(
        owner="local",
        name="proj",
        source_files=["main.py"],
        symbols=[
            Symbol(
                id="main.py::foo#function",
                file="main.py",
                name="foo",
                qualified_name="foo",
                kind="function",
                language="python",
                signature="def foo():",
                byte_offset=0,
                byte_length=len(original.encode("utf-8")),
            )
        ],
        raw_files={"main.py": original},
        languages={"python": 1},
        folder_path=tmp_path,
    )

    original_mtime_ns = main_py.stat().st_mtime_ns
    time.sleep(0.01)
    main_py.write_text(updated, encoding="utf-8")
    os.utime(main_py, ns=(original_mtime_ns, original_mtime_ns))

    changed, added, deleted = store.detect_changes_fast(
        "local", "proj", tmp_path, [main_py], source_path=None
    )
    assert changed == ["main.py"]
    assert added == []
    assert deleted == []


def test_get_symbol_content_rejects_traversal_symbol_file(tmp_path):
    """Traversal in stored symbol file path should be blocked on read."""
    store = IndexStore(base_path=str(tmp_path))
    content_dir = tmp_path / "owner-repo"
    content_dir.mkdir(parents=True)
    (content_dir / "safe.py").write_text("def ok():\n    return 1\n", encoding="utf-8")

    # Write index manually to simulate malicious/corrupt symbol metadata.
    index_path = tmp_path / "owner-repo.json"
    index_path.write_text(
        json.dumps({
            "repo": "owner/repo",
            "owner": "owner",
            "name": "repo",
            "indexed_at": "2025-01-01T00:00:00",
            "source_files": ["safe.py"],
            "languages": {"python": 1},
            "symbols": [{
                "id": "bad::sym",
                "file": "../../etc/passwd",
                "name": "bad",
                "qualified_name": "bad",
                "kind": "function",
                "language": "python",
                "signature": "def bad():",
                "docstring": "",
                "summary": "",
                "decorators": [],
                "keywords": [],
                "parent": "",
                "line": 1,
                "end_line": 1,
                "byte_offset": 0,
                "byte_length": 10,
                "content_hash": "",
            }],
            "index_version": 2,
            "file_hashes": {"safe.py": "abc"},
            "git_head": "",
        }),
        encoding="utf-8",
    )

    assert store.get_symbol_content("owner", "repo", "bad::sym") is None


def test_save_index_rejects_invalid_owner_component(tmp_path):
    """Owner/name inputs should be safe for filesystem paths."""
    store = IndexStore(base_path=str(tmp_path))

    with pytest.raises(ValueError, match="Invalid owner"):
        store.save_index(
            owner="../escape",
            name="repo",
            source_files=["main.py"],
            symbols=[],
            raw_files={"main.py": ""},
            languages={"python": 1},
        )


def test_save_index_rejects_invalid_name_component(tmp_path):
    """Reject names with path separators or unsafe characters."""
    store = IndexStore(base_path=str(tmp_path))

    with pytest.raises(ValueError, match="Invalid name"):
        store.save_index(
            owner="owner",
            name="repo/evil",
            source_files=["main.py"],
            symbols=[],
            raw_files={"main.py": ""},
            languages={"python": 1},
        )


def test_codeindex_get_symbol():
    """Test getting a symbol by ID from CodeIndex."""
    index = CodeIndex(
        repo="test/repo",
        owner="test",
        name="repo",
        indexed_at="2025-01-15T10:00:00",
        source_files=["main.py"],
        languages={"python": 1},
        symbols=[
            {"id": "main-py::foo", "name": "foo", "kind": "function"},
            {"id": "main-py::bar", "name": "bar", "kind": "function"},
        ]
    )
    
    sym = index.get_symbol("main-py::foo")
    assert sym is not None
    assert sym["name"] == "foo"
    
    assert index.get_symbol("nonexistent") is None


def test_codeindex_search():
    """Test searching symbols."""
    index = CodeIndex(
        repo="test/repo",
        owner="test",
        name="repo",
        indexed_at="2025-01-15T10:00:00",
        source_files=["main.py"],
        languages={"python": 1},
        symbols=[
            {"id": "main-py::authenticate", "name": "authenticate", "kind": "function", "signature": "def authenticate(user)", "summary": "Auth user", "keywords": ["auth"]},
            {"id": "main-py::login", "name": "login", "kind": "function", "signature": "def login()", "summary": "Login user", "keywords": []},
            {"id": "main-py::MyClass", "name": "MyClass", "kind": "class", "signature": "class MyClass", "summary": "A class", "keywords": []},
        ]
    )
    
    # Search by name
    results = index.search("authenticate")
    assert len(results) > 0
    assert results[0]["name"] == "authenticate"
    
    # Search by kind filter
    results = index.search("login", kind="class")
    assert len(results) == 0  # login is a function
    
    results = index.search("login", kind="function")
    assert len(results) > 0


def test_incremental_save_recomputes_languages_from_merged_symbols(tmp_path):
    """incremental_save should derive languages from merged symbols, not caller counts."""
    store = IndexStore(base_path=str(tmp_path))

    py_sym = Symbol(
        id="app-py::run#function",
        file="app.py",
        name="run",
        qualified_name="run",
        kind="function",
        language="python",
        signature="def run():",
        byte_offset=0,
        byte_length=10,
    )
    c_sym = Symbol(
        id="api-h::only_c#function",
        file="api.h",
        name="only_c",
        qualified_name="only_c",
        kind="function",
        language="c",
        signature="int only_c(void)",
        byte_offset=0,
        byte_length=20,
    )

    store.save_index(
        owner="lang",
        name="demo",
        source_files=["app.py", "api.h"],
        symbols=[py_sym, c_sym],
        raw_files={"app.py": "def run():\n    pass\n", "api.h": "int only_c(void) { return 0; }\n"},
        languages={"python": 1, "c": 1},
    )

    cpp_sym = Symbol(
        id="api-h::Widget#class",
        file="api.h",
        name="Widget",
        qualified_name="Widget",
        kind="class",
        language="cpp",
        signature="class Widget",
        byte_offset=0,
        byte_length=12,
    )

    updated = store.incremental_save(
        owner="lang",
        name="demo",
        changed_files=["api.h"],
        new_files=[],
        deleted_files=[],
        new_symbols=[cpp_sym],
        raw_files={"api.h": "class Widget { public: int Get() const; };"},
        languages={"c": 99},  # stale caller-provided data; should be ignored
    )

    assert updated is not None
    assert updated.languages == {"python": 1, "cpp": 1}


def test_get_ref_count_reads_stored_field(tmp_path):
    """get_ref_count returns the value stored in ref_count field."""
    store = IndexStore(base_path=str(tmp_path))
    refs = [
        {"callee": "foo", "ref_type": "call", "caller_file": "a.py",
         "caller_line": i, "caller_symbol_id": None, "is_test": False}
        for i in range(7)
    ]
    store.save_refs("owner", "repo", refs)
    assert store.get_ref_count("owner", "repo") == 7


def test_get_ref_count_fallback_for_old_format(tmp_path):
    """Falls back to len(refs) when ref_count field is absent (old format)."""
    store = IndexStore(base_path=str(tmp_path))
    refs_path = store._refs_path("owner", "repo")
    refs_path.write_text(
        json.dumps({"repo": "owner/repo", "refs": [{}, {}, {}]}),
        encoding="utf-8",
    )
    assert store.get_ref_count("owner", "repo") == 3


def test_get_ref_count_returns_zero_when_file_missing(tmp_path):
    """Returns 0 when no refs file exists yet."""
    store = IndexStore(base_path=str(tmp_path))
    assert store.get_ref_count("owner", "repo") == 0

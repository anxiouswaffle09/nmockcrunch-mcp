# Compact get_file_tree + Tool Description Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace get_file_tree's bloated JSON output with compact indented text, fix its token savings baseline, and correct inaccurate tool descriptions.

**Architecture:** Rewrite `_build_tree` and `_dict_to_list` into a single text renderer that produces indented lines with symbol counts and language. Remove the `include_summaries` feature. Change the savings baseline from sum-of-file-sizes to sum-of-path-lengths (Glob equivalent).

**Tech Stack:** Python, pytest

---

### Task 1: Rewrite get_file_tree tests for compact text format

**Files:**
- Modify: `tests/test_get_file_tree.py` (full rewrite)

**Step 1: Write the new tests**

Replace the entire file with tests that verify the compact text format:

```python
"""Tests for get_file_tree compact text output."""

from jcodemunch_mcp.parser import Symbol
from jcodemunch_mcp.storage import IndexStore
from jcodemunch_mcp.tools.get_file_tree import get_file_tree


def _make_index(tmp_path, owner, name, source_files, symbols, raw_files, languages):
    """Helper to create a test index."""
    store = IndexStore(base_path=str(tmp_path))
    store.save_index(
        owner=owner, name=name,
        source_files=source_files,
        symbols=symbols,
        raw_files=raw_files,
        languages=languages,
    )
    return store


def test_compact_output_contains_symbol_counts_and_language(tmp_path):
    """Compact text output should show symbol count and language per file."""
    sym = Symbol(
        id="src-main-py::main#function",
        file="src/main.py",
        name="main",
        qualified_name="main",
        kind="function",
        language="python",
        signature="def main()",
        byte_offset=0,
        byte_length=20,
    )
    _make_index(
        tmp_path, "tree", "demo",
        source_files=["src/main.py"],
        symbols=[sym],
        raw_files={"src/main.py": "def main():\n    pass\n"},
        languages={"python": 1},
    )

    result = get_file_tree("tree/demo", storage_path=str(tmp_path))
    assert "error" not in result
    tree_text = result["tree"]
    assert isinstance(tree_text, str)
    assert "main.py" in tree_text
    assert "[1 symbol]" in tree_text or "[1 symbols]" in tree_text
    assert "python" in tree_text


def test_compact_output_hides_empty_files_by_default(tmp_path):
    """Files with zero symbols should be hidden when show_empty=False (default)."""
    sym = Symbol(
        id="src-main-py::main#function",
        file="src/main.py",
        name="main",
        qualified_name="main",
        kind="function",
        language="python",
        signature="def main()",
        byte_offset=0,
        byte_length=20,
    )
    _make_index(
        tmp_path, "tree", "demo2",
        source_files=["src/main.py", "src/__init__.py"],
        symbols=[sym],
        raw_files={
            "src/main.py": "def main():\n    pass\n",
            "src/__init__.py": "",
        },
        languages={"python": 1},
    )

    result = get_file_tree("tree/demo2", storage_path=str(tmp_path))
    tree_text = result["tree"]
    assert "main.py" in tree_text
    assert "__init__.py" not in tree_text


def test_compact_output_shows_empty_files_when_requested(tmp_path):
    """Files with zero symbols should appear when show_empty=True."""
    sym = Symbol(
        id="src-main-py::main#function",
        file="src/main.py",
        name="main",
        qualified_name="main",
        kind="function",
        language="python",
        signature="def main()",
        byte_offset=0,
        byte_length=20,
    )
    _make_index(
        tmp_path, "tree", "demo3",
        source_files=["src/main.py", "src/__init__.py"],
        symbols=[sym],
        raw_files={
            "src/main.py": "def main():\n    pass\n",
            "src/__init__.py": "",
        },
        languages={"python": 1},
    )

    result = get_file_tree("tree/demo3", show_empty=True, storage_path=str(tmp_path))
    tree_text = result["tree"]
    assert "main.py" in tree_text
    assert "__init__.py" in tree_text


def test_compact_output_sorts_by_symbol_count_descending(tmp_path):
    """Within a directory, files should be sorted by symbol count descending."""
    sym_a = Symbol(
        id="src-a-py::a#function", file="src/a.py", name="a",
        qualified_name="a", kind="function", language="python",
        signature="def a()", byte_offset=0, byte_length=10,
    )
    syms_b = [
        Symbol(
            id=f"src-b-py::b{i}#function", file="src/b.py", name=f"b{i}",
            qualified_name=f"b{i}", kind="function", language="python",
            signature=f"def b{i}()", byte_offset=i * 10, byte_length=10,
        )
        for i in range(5)
    ]
    _make_index(
        tmp_path, "tree", "demo4",
        source_files=["src/a.py", "src/b.py"],
        symbols=[sym_a] + syms_b,
        raw_files={"src/a.py": "def a(): pass\n", "src/b.py": "# five funcs\n"},
        languages={"python": 2},
    )

    result = get_file_tree("tree/demo4", storage_path=str(tmp_path))
    tree_text = result["tree"]
    b_pos = tree_text.index("b.py")
    a_pos = tree_text.index("a.py")
    assert b_pos < a_pos, "b.py (5 symbols) should appear before a.py (1 symbol)"


def test_compact_output_prefers_symbol_language_over_extension(tmp_path):
    """A .h file with C symbols should show language='c'."""
    sym = Symbol(
        id="include-api-h::only_c#function",
        file="include/api.h",
        name="only_c",
        qualified_name="only_c",
        kind="function",
        language="c",
        signature="int only_c(void)",
        byte_offset=0,
        byte_length=20,
    )
    _make_index(
        tmp_path, "tree", "demo5",
        source_files=["include/api.h"],
        symbols=[sym],
        raw_files={"include/api.h": "int only_c(void) { return 0; }\n"},
        languages={"c": 1},
    )

    result = get_file_tree("tree/demo5", storage_path=str(tmp_path))
    tree_text = result["tree"]
    # The line for api.h should contain "c" as the language
    for line in tree_text.splitlines():
        if "api.h" in line:
            assert line.rstrip().endswith("c")
            break
    else:
        raise AssertionError("api.h not found in tree output")


def test_path_prefix_filters_output(tmp_path):
    """path_prefix should limit output to matching files."""
    sym1 = Symbol(
        id="src-main-py::main#function", file="src/main.py", name="main",
        qualified_name="main", kind="function", language="python",
        signature="def main()", byte_offset=0, byte_length=10,
    )
    sym2 = Symbol(
        id="tests-test-py::test#function", file="tests/test.py", name="test",
        qualified_name="test", kind="function", language="python",
        signature="def test()", byte_offset=0, byte_length=10,
    )
    _make_index(
        tmp_path, "tree", "demo6",
        source_files=["src/main.py", "tests/test.py"],
        symbols=[sym1, sym2],
        raw_files={"src/main.py": "def main(): pass\n", "tests/test.py": "def test(): pass\n"},
        languages={"python": 2},
    )

    result = get_file_tree("tree/demo6", path_prefix="src/", storage_path=str(tmp_path))
    tree_text = result["tree"]
    assert "main.py" in tree_text
    assert "test.py" not in tree_text
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/wendy/MCPs/jcodemunch-mcp && python -m pytest tests/test_get_file_tree.py -v`
Expected: FAIL — tests expect string `tree` field but current code returns list of dicts

**Step 3: Commit the failing tests**

```bash
git add tests/test_get_file_tree.py
git commit -m "test: rewrite get_file_tree tests for compact text format (red)"
```

---

### Task 2: Rewrite get_file_tree implementation

**Files:**
- Modify: `src/jcodemunch_mcp/tools/get_file_tree.py` (major rewrite)

**Step 1: Rewrite the implementation**

Replace the full file with:

```python
"""Get file tree for a repository."""

import os
import time
from typing import Optional

from ..storage import IndexStore, record_savings, estimate_savings, cost_avoided
from ._utils import resolve_repo


def get_file_tree(
    repo: str,
    path_prefix: str = "",
    show_empty: bool = False,
    storage_path: Optional[str] = None
) -> dict:
    """Get repository file tree as compact indented text.

    Args:
        repo: Repository identifier (owner/repo or just repo name)
        path_prefix: Optional path prefix to filter
        show_empty: Show files with zero symbols (default False)
        storage_path: Custom storage path

    Returns:
        Dict with tree (str), repo, path_prefix, and _meta envelope.
    """
    start = time.perf_counter()

    try:
        owner, name = resolve_repo(repo, storage_path)
    except ValueError as e:
        return {"error": str(e)}

    store = IndexStore(base_path=storage_path)
    index = store.load_index(owner, name)

    if not index:
        return {"error": f"Repository not indexed: {owner}/{name}"}

    # Filter files by prefix
    files = [f for f in index.source_files if f.startswith(path_prefix)]

    if not files:
        return {
            "repo": f"{owner}/{name}",
            "path_prefix": path_prefix,
            "tree": ""
        }

    # Precompute symbol counts and languages per file
    symbol_counts: dict[str, int] = {}
    file_languages: dict[str, str] = {}
    for sym in index.symbols:
        fp = sym.get("file", "")
        symbol_counts[fp] = symbol_counts.get(fp, 0) + 1
        if fp not in file_languages:
            lang = sym.get("language", "")
            if lang:
                file_languages[fp] = lang

    # Build tree text
    tree_text = _render_tree(files, path_prefix, symbol_counts, file_languages, show_empty)

    elapsed = (time.perf_counter() - start) * 1000

    # Token savings: path listing (Glob equivalent) vs compact tree response
    raw_bytes = sum(len(f.encode()) + 1 for f in files)  # +1 for newline per path
    response_bytes = len(tree_text.encode())
    tokens_saved = estimate_savings(raw_bytes, response_bytes)
    total_saved = record_savings(tokens_saved)

    file_count = len(files)
    if not show_empty:
        file_count = sum(1 for f in files if symbol_counts.get(f, 0) > 0)

    return {
        "repo": f"{owner}/{name}",
        "path_prefix": path_prefix,
        "tree": tree_text,
        "_meta": {
            "timing_ms": round(elapsed, 1),
            "file_count": file_count,
            "tokens_saved": tokens_saved,
            "total_tokens_saved": total_saved,
            **cost_avoided(tokens_saved, total_saved),
        },
    }


def _render_tree(
    files: list[str],
    path_prefix: str,
    symbol_counts: dict[str, int],
    file_languages: dict[str, str],
    show_empty: bool,
) -> str:
    """Render file list as compact indented text tree.

    Files are sorted by symbol count descending within each directory.
    Zero-symbol files are hidden unless show_empty is True.
    """
    # Build nested dict: dir -> list of (rel_path, full_path, count, lang)
    root: dict = {}

    for file_path in files:
        count = symbol_counts.get(file_path, 0)
        if not show_empty and count == 0:
            continue

        rel_path = file_path[len(path_prefix):].lstrip("/")
        parts = rel_path.split("/")

        # Get language
        lang = file_languages.get(file_path, "")
        if not lang:
            _, ext = os.path.splitext(file_path)
            from ..parser import LANGUAGE_EXTENSIONS
            lang = LANGUAGE_EXTENSIONS.get(ext, "")

        # Navigate to parent dir
        current = root
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Store file entry as tuple
        filename = parts[-1]
        current[filename] = (file_path, count, lang)

    lines: list[str] = []
    _render_node(root, lines, indent=0)
    return "\n".join(lines)


def _render_node(node: dict, lines: list[str], indent: int) -> None:
    """Recursively render tree nodes to text lines."""
    prefix = "  " * indent

    # Separate dirs and files
    dirs = []
    file_entries = []
    for key, value in node.items():
        if isinstance(value, tuple):
            file_entries.append((key, value))
        else:
            dirs.append((key, value))

    # Sort dirs alphabetically
    dirs.sort(key=lambda x: x[0])

    # Sort files by symbol count descending, then name ascending
    file_entries.sort(key=lambda x: (-x[1][1], x[0]))

    # Render dirs first, then files
    for dir_name, children in dirs:
        lines.append(f"{prefix}{dir_name}/")
        _render_node(children, lines, indent + 1)

    for filename, (full_path, count, lang) in file_entries:
        symbol_label = "symbol" if count == 1 else "symbols"
        lang_suffix = f"  {lang}" if lang else ""
        lines.append(f"{prefix}{filename:<30} [{count} {symbol_label}]{lang_suffix}")
```

**Step 2: Run tests to verify they pass**

Run: `cd /home/wendy/MCPs/jcodemunch-mcp && python -m pytest tests/test_get_file_tree.py -v`
Expected: All 6 tests PASS

**Step 3: Run full test suite to check for regressions**

Run: `cd /home/wendy/MCPs/jcodemunch-mcp && python -m pytest --tb=short -q`
Expected: All tests pass. The `test_fixes.py::TestNoDuplicateIndexStoreGetFileTree` test checks source for `store2` — unaffected by our changes.

**Step 4: Commit**

```bash
git add src/jcodemunch_mcp/tools/get_file_tree.py
git commit -m "feat: compact text output for get_file_tree, fix savings baseline"
```

---

### Task 3: Update server.py — remove include_summaries, fix descriptions

**Files:**
- Modify: `src/jcodemunch_mcp/server.py:296-319` (get_file_tree tool schema)
- Modify: `src/jcodemunch_mcp/server.py:320-337` (get_file_outline description)
- Modify: `src/jcodemunch_mcp/server.py:446-487` (search_text description)
- Modify: `src/jcodemunch_mcp/server.py:674-680` (get_file_tree call site)

**Step 1: Update the get_file_tree tool schema**

In server.py, replace the `get_file_tree` Tool definition (around line 296) with:

```python
Tool(
    name="get_file_tree",
    description="Get the file tree of an indexed repository as compact indented text. Each file shows its symbol count and language. Files sorted by symbol count (most important first). Zero-symbol files hidden by default.",
    inputSchema={
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository identifier (owner/repo or just repo name)"
            },
            "path_prefix": {
                "type": "string",
                "description": "Optional path prefix to filter (e.g., 'src/utils')",
                "default": ""
            },
            "show_empty": {
                "type": "boolean",
                "description": "Show files with zero symbols (default false)",
                "default": False
            }
        },
        "required": ["repo"]
    }
),
```

**Step 2: Update the get_file_tree call site**

In server.py, replace the `get_file_tree` call (around line 674) with:

```python
elif name == "get_file_tree":
    result = get_file_tree(
        repo=arguments["repo"],
        path_prefix=arguments.get("path_prefix", ""),
        show_empty=arguments.get("show_empty", False),
        storage_path=storage_path
    )
```

**Step 3: Update get_file_outline description**

Change the description string (around line 321) from:
```
"Get all symbols (functions, classes, methods) in a file with signatures and summaries."
```
to:
```
"Get all symbols (functions, classes, methods, constants) in a file. Returns hierarchical symbol tree with signatures, summaries, and line numbers."
```

**Step 4: Update search_text description**

Change the description string (around line 447) from:
```
"Full-text search across indexed file contents. Useful when symbol search misses (e.g., string literals, comments, config values). Use exact=true for punctuation-heavy queries like Foo::new(, enum variants, macro invocations. Check 'total_hits' — if it exceeds result_count, use offset/exhaustive to get more."
```
to:
```
"Full-text search across indexed file contents. Useful when symbol search misses (e.g., string literals, comments, config values). Use exact=true for punctuation-heavy queries like Foo::new(, enum variants, macro invocations. Check 'total_hits' — if it exceeds result_count, use offset/exhaustive to get more. Scans up to 50MB of content — use file_pattern to narrow scope on large repos."
```

**Step 5: Run full test suite**

Run: `cd /home/wendy/MCPs/jcodemunch-mcp && python -m pytest --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/jcodemunch_mcp/server.py
git commit -m "fix: update tool descriptions for accuracy, remove include_summaries from get_file_tree"
```

---

### Task 4: Manual smoke test with live jcodemunch

**Step 1: Run get_file_tree via jcodemunch on this repo**

Use the `mcp__jcodemunch__get_file_tree` tool on `jcodemunch-mcp` and verify:
- Output is compact indented text (not JSON)
- Symbol counts appear per file
- Language shown per file
- Zero-symbol files (like `__init__.py`) are hidden
- `_meta.tokens_saved` is realistic (not 111k)

**Step 2: Run get_file_tree with show_empty=True**

Verify `__init__.py` files now appear.

**Step 3: Run get_file_tree with path_prefix**

Use `path_prefix="src/jcodemunch_mcp/tools/"` and verify only tools files appear.

**Step 4: Compare token counts**

Count characters of the compact response and compare against a Glob of the same scope. The compact response should be comparable to or smaller than Glob output.

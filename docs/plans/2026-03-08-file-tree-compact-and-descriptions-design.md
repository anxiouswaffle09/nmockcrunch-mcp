# Design: Compact get_file_tree + Tool Description Fixes

**Date:** 2026-03-08
**Scope:** `get_file_tree` output format, token savings baseline, tool descriptions

---

## Problem

### get_file_tree output bloat

The current `get_file_tree` returns nested JSON with verbose per-file objects. On this repo (46 files), the response is 5,910 characters — 6x larger than a plain Glob file listing (1,033 chars). The server claims 111k tokens saved when the tool actually costs *more* tokens than the built-in alternative.

The root causes:
- Verbose JSON structure (~90 chars per file node)
- Token savings baseline compares against reading all file *contents*, not against a directory listing (the realistic alternative)

### Tool description inaccuracies

Several tool descriptions in `server.py` are incomplete or misleading:
- `get_file_outline` omits constants from symbol kinds and doesn't mention hierarchical output
- `search_text` doesn't mention its 50MB scan cap

---

## Changes

### A. get_file_tree compact output

**Remove:**
- `include_summaries` parameter (unused feature)
- `_flatten_tree_nodes` helper (only used for summary warning)
- Nested JSON tree format (replaced entirely)

**Add:**
- `show_empty` parameter (default `False`) — hides zero-symbol files like `__init__.py`
- Compact indented text renderer

**Modify:**
- `_build_tree` / `_dict_to_list` — replace with text rendering
- Token savings: `raw_bytes` changes from sum-of-file-sizes to sum-of-path-lengths (Glob baseline)
- `response_bytes` uses actual rendered text length

**Output format:**
```
src/jcodemunch_mcp/
  storage/
    index_store.py         [39 symbols]  python
    token_tracker.py       [12 symbols]  python
  tools/
    find_references.py     [10 symbols]  python
    index_repo.py          [10 symbols]  python
    ...
  server.py               [16 symbols]  python
tests/
  test_fixes.py           [84 symbols]  python
  test_hardening.py       [83 symbols]  python
  ...
```

Key behaviors:
- Files sorted by symbol count descending within each directory
- Zero-symbol files hidden by default (`show_empty=True` to include)
- Language shown per file
- Indentation encodes hierarchy (same info as nested JSON)

**Return value:** The function still returns a dict with `repo`, `path_prefix`, `tree` (now a string), and `_meta`. The `tree` field becomes the compact text string instead of a list of dicts.

### B. Tool description fixes in server.py

1. **`get_file_outline`**: Change `(functions, classes, methods)` to `(functions, classes, methods, constants)`. Add "Returns hierarchical symbol tree with line numbers."

2. **`get_file_tree`**: Rewrite description for compact text format. Mention symbol counts, language per file, `show_empty` parameter.

3. **`search_text`**: Add mention of 50MB scan cap so agents know to use `file_pattern` on large repos.

### C. Test updates

- `test_get_file_tree.py`: Rewrite both tests for compact text format. Verify symbol counts and language labels appear in output text. Add test for `show_empty=True/False`.
- `test_fixes.py`: `TestNoDuplicateIndexStoreGetFileTree` (source inspection) — unaffected.

---

## Files touched

| File | Change |
|---|---|
| `src/jcodemunch_mcp/tools/get_file_tree.py` | Rewrite output format, remove summaries, fix savings baseline |
| `src/jcodemunch_mcp/server.py` | Update tool descriptions (get_file_outline, get_file_tree, search_text), remove include_summaries param |
| `tests/test_get_file_tree.py` | Rewrite tests for compact format |

---

## What doesn't change

- `get_repo_outline` savings calculation (response_bytes=0 is defensible)
- `get_symbol`, `search_symbols`, `search_text`, `get_file_outline` savings calculations
- The `estimate_savings` / `record_savings` / `cost_avoided` functions in token_tracker.py
- The `_BYTES_PER_TOKEN = 4` constant
- The `include_summaries` feature on `index_folder` / `index_repo` (per-symbol summaries stay, only per-file tree summaries are removed)

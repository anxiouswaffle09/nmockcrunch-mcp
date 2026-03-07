# Revert Guide

## Path guard feature (2026-03-07)

**Commit:** `76e9fde` — feat: add path guard, watchlist management tools, and fix autorefresh bugs

**What it added:**
- Path guard on all `_REFRESH_TOOLS` — blocks queries when cwd is not in `~/.code-index/autorefresh.json`
- Path guard on `index_folder` — blocks indexing paths not under a watched directory
- Three management tools: `add_to_watchlist`, `remove_from_watchlist`, `list_watched_paths`
- `register_path` returns bool; `add_to_watchlist` correctly reports cap-hit failure
- 30 new tests in `tests/test_autorefresher.py`
- Tool count in `test_server.py` updated 16 → 19

**State before this feature:** `23d55ee`

### To revert the entire feature

```bash
git revert 76e9fde
```

This creates a new commit that undoes everything in `1c5b809`. The test file `tests/test_autorefresher.py` will be deleted, tool count will go back to 16, and the path guard + management tools will be removed from `server.py`.

### To revert only a specific file

```bash
git checkout 23d55ee -- src/jcodemunch_mcp/server.py tests/test_server.py
git rm tests/test_autorefresher.py
```

### Also revert autorefresh.json (nmockdrunk-mcp was removed in this session)

`nmockdrunk-mcp` was removed from `~/.code-index/autorefresh.json` manually (not via git). To restore it, add `/home/wendy/MCPs/nmockdrunk-mcp` back to the `paths` array in that file.

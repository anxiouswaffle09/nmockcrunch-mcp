"""MCP server for jcodemunch-mcp."""

import argparse
import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.types import Tool, TextContent

from .tools.index_repo import index_repo
from .tools.index_folder import index_folder
from .tools.list_repos import list_repos
from .tools.get_file_tree import get_file_tree
from .tools.get_file_outline import get_file_outline
from .tools.get_symbol import get_symbol, get_symbols
from .tools.search_symbols import search_symbols
from .tools.invalidate_cache import invalidate_cache
from .tools.search_text import search_text
from .tools.get_repo_outline import get_repo_outline
from .tools.find_references import (
    find_references,
    find_callers,
    find_constructors,
    find_field_reads,
    find_field_writes,
)


# Tools that need an up-to-date index before running.
# Allowlist prevents new tools from silently skipping refresh.
_REFRESH_TOOLS = {
    "get_file_tree", "get_file_outline", "get_symbol", "get_symbols",
    "search_symbols", "search_text", "get_repo_outline",
    "find_references", "find_callers", "find_constructors",
    "find_field_reads", "find_field_writes",
}

_log = logging.getLogger("jcodemunch")

MAX_WATCHED_PATHS = 20

# Per-path locks to prevent concurrent refreshes of the same path.
_refresh_locks: dict[str, threading.Lock] = {}
_refresh_locks_mutex = threading.Lock()


def _get_path_lock(path: str) -> threading.Lock:
    with _refresh_locks_mutex:
        if len(_refresh_locks) > 50:
            _log.warning("autorefresh: path lock table > 50 entries")
        if path not in _refresh_locks:
            _refresh_locks[path] = threading.Lock()
        return _refresh_locks[path]


class AutoRefresher:
    """Incrementally re-indexes watched paths before every read tool call."""

    CONFIG_PATH = os.path.expanduser("~/.code-index/autorefresh.json")

    def __init__(self):
        self._lock = threading.Lock()
        self._last_refresh: dict[str, float] = {}
        self._paths: set[str] = set()
        self._cooldown: float = 0.0
        self._cfg_mtime: Optional[float] = None
        self._load_config()

    def _load_config(self):
        try:
            with open(self.CONFIG_PATH) as f:
                cfg = json.load(f)
            resolved_paths = {
                os.path.realpath(os.path.expanduser(str(p)))
                for p in cfg.get("paths", [])
            }
            with self._lock:
                self._cooldown = float(cfg.get("cooldown_secs", 0))
                self._paths = resolved_paths
                self._last_refresh = {
                    path: last
                    for path, last in self._last_refresh.items()
                    if path in resolved_paths
                }
            _log.debug("autorefresh: watching %s", ", ".join(resolved_paths) or "(none)")
        except FileNotFoundError:
            pass
        except Exception as e:
            _log.warning("autorefresh: config error: %s", e)

    def _maybe_reload_config(self):
        """Re-read config if autorefresh.json has changed on disk."""
        try:
            cfg_mtime = Path(self.CONFIG_PATH).stat().st_mtime
            if cfg_mtime != self._cfg_mtime:
                self._cfg_mtime = cfg_mtime
                self._load_config()
        except OSError:
            pass

    def register_path(self, path: str) -> bool:
        resolved = os.path.realpath(os.path.expanduser(str(path)))
        with self._lock:
            if resolved in self._paths:
                return True
            if len(self._paths) >= MAX_WATCHED_PATHS:
                _log.warning(
                    "autorefresh: watchlist full (%d paths). "
                    "Add path to autorefresh.json manually to persist it.",
                    MAX_WATCHED_PATHS,
                )
                return False
            self._paths.add(resolved)

            # Persist to config atomically, inside the lock to prevent concurrent
            # register_path calls from racing on the file read-modify-write.
            cfg_path = Path(self.CONFIG_PATH)
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cfg_path.with_suffix(".json.tmp")
            try:
                existing = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            except Exception:
                existing = {}
            existing["paths"] = sorted(self._paths)
            tmp.write_text(json.dumps(existing, indent=2))
            tmp.replace(cfg_path)
            _log.debug("autorefresh: registered and persisted %s", resolved)
        return True

    def remove_path(self, path: str) -> bool:
        """Remove path from watchlist and persist. Returns True if path was present."""
        resolved = os.path.realpath(os.path.expanduser(str(path)))
        with self._lock:
            if resolved not in self._paths:
                return False
            self._paths.discard(resolved)

            # Persist to config atomically, inside the lock to prevent concurrent
            # remove_path calls from racing on the file read-modify-write.
            cfg_path = Path(self.CONFIG_PATH)
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cfg_path.with_suffix(".json.tmp")
            try:
                existing = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            except Exception:
                existing = {}
            existing["paths"] = sorted(self._paths)
            tmp.write_text(json.dumps(existing, indent=2))
            tmp.replace(cfg_path)
            _log.debug("autorefresh: removed and persisted %s", resolved)
        return True

    def get_watched_paths(self) -> list[str]:
        """Return a sorted snapshot of currently watched paths."""
        self._maybe_reload_config()
        with self._lock:
            return sorted(self._paths)

    def is_path_watched(self, path: str) -> bool:
        """Check if path falls within a watched directory. Empty watchlist = allow all."""
        self._maybe_reload_config()
        resolved = os.path.realpath(os.path.expanduser(str(path)))
        with self._lock:
            if not self._paths:
                return True
            return any(resolved == p or resolved.startswith(p + os.sep) for p in self._paths)

    def maybe_refresh(self, storage_path: Optional[str]):
        self._maybe_reload_config()
        now = time.monotonic()
        with self._lock:
            paths = list(self._paths)
        for path in paths:
            with self._lock:
                last = self._last_refresh.get(path, 0.0)
                if now - last < self._cooldown:
                    continue
                self._last_refresh[path] = now

            path_lock = _get_path_lock(path)
            if not path_lock.acquire(blocking=False):
                _log.debug("autorefresh: %s already refreshing, skipping", path)
                continue

            try:
                _log.debug("autorefresh: refreshing %s", path)
                result = index_folder(
                    path=path,
                    use_ai_summaries=False,
                    storage_path=storage_path,
                    incremental=True,
                )
                _log.debug(
                    "autorefresh: %s — changed=%s new=%s deleted=%s",
                    path,
                    result.get("changed", 0),
                    result.get("new", 0),
                    result.get("deleted", 0),
                )
            except Exception as e:
                _log.warning("autorefresh: error refreshing %s: %s", path, e)
            finally:
                path_lock.release()


auto_refresher = AutoRefresher()

if os.environ.get("JCODEMUNCH_SHARE_SAVINGS", "0") == "1":
    _log.info(
        "jcodemunch: anonymous token-savings telemetry is ON. "
        "Set JCODEMUNCH_SHARE_SAVINGS=0 to disable. "
        "See README for details."
    )


# Create server
server = Server("jcodemunch-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        Tool(
            name="index_repo",
            description="Index a GitHub repository's source code. Fetches files, parses ASTs, extracts symbols, and saves to local storage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "GitHub repository URL or owner/repo string"
                    },
                    "use_ai_summaries": {
                        "type": "boolean",
                        "description": "Use AI to generate symbol summaries (requires ANTHROPIC_API_KEY or GOOGLE_API_KEY). Anthropic takes priority if both are set. When false, uses docstrings or signature fallback.",
                        "default": True
                    },
                    "incremental": {
                        "type": "boolean",
                        "description": "When true and an existing index exists, only re-index changed files.",
                        "default": False
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="index_folder",
            description="Index a local folder containing source code. Response includes `discovery_skip_counts` (files filtered per reason), `no_symbols_count`/`no_symbols_files` (files with no extractable symbols) for diagnosing missing files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to local folder (absolute or relative, supports ~ for home directory)"
                    },
                    "use_ai_summaries": {
                        "type": "boolean",
                        "description": "Use AI to generate symbol summaries (requires ANTHROPIC_API_KEY or GOOGLE_API_KEY). Anthropic takes priority if both are set. When false, uses docstrings or signature fallback.",
                        "default": True
                    },
                    "extra_ignore_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional gitignore-style patterns to exclude from indexing"
                    },
                    "follow_symlinks": {
                        "type": "boolean",
                        "description": "Whether to follow symlinks. Default false for security.",
                        "default": False
                    },
                    "incremental": {
                        "type": "boolean",
                        "description": "When true and an existing index exists, only re-index changed files.",
                        "default": False
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="list_repos",
            description="List all indexed repositories.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
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
        Tool(
            name="get_file_outline",
            description="Get all symbols (functions, classes, methods, constants) in a file. Returns hierarchical symbol tree with signatures, summaries, and line numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository identifier (owner/repo or just repo name)"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file within the repository (e.g., 'src/main.py')"
                    }
                },
                "required": ["repo", "file_path"]
            }
        ),
        Tool(
            name="get_symbol",
            description="Get the full source code of a specific symbol. Use after identifying relevant symbols via get_file_outline or search_symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository identifier (owner/repo or just repo name)"
                    },
                    "symbol_id": {
                        "type": "string",
                        "description": "Symbol ID from get_file_outline or search_symbols"
                    },
                    "verify": {
                        "type": "boolean",
                        "description": "Verify content hash matches stored hash (detects source drift)",
                        "default": False
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of lines before/after symbol to include for context",
                        "default": 0
                    }
                },
                "required": ["repo", "symbol_id"]
            }
        ),
        Tool(
            name="get_symbols",
            description="Get full source code of multiple symbols in one call. Efficient for loading related symbols.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository identifier (owner/repo or just repo name)"
                    },
                    "symbol_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of symbol IDs to retrieve"
                    }
                },
                "required": ["repo", "symbol_ids"]
            }
        ),
        Tool(
            name="search_symbols",
            description="Search for symbols matching a query across the entire indexed repository. Returns matches with signatures and summaries. Check 'total_hits' in the response — if it exceeds result_count, use offset/exhaustive to get more.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository identifier (owner/repo or just repo name)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (matches symbol names, signatures, summaries, docstrings)"
                    },
                    "kind": {
                        "type": "string",
                        "description": "Optional filter by symbol kind",
                        "enum": ["function", "class", "method", "constant", "type"]
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional glob pattern to filter files (e.g., 'src/**/*.py')"
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional filter by language",
                        "enum": ["python", "javascript", "typescript", "go", "rust", "java", "php", "dart", "csharp", "c", "cpp", "swift"]
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 10, max 200)",
                        "default": 10
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip this many results before returning — use for pagination",
                        "default": 0
                    },
                    "exhaustive": {
                        "type": "boolean",
                        "description": "Return all results ignoring max_results cap. Use for wiring audits and dead-code checks.",
                        "default": False
                    }
                },
                "required": ["repo", "query"]
            }
        ),
        Tool(
            name="invalidate_cache",
            description="Delete the index and cached files for a repository. Forces a full re-index on next index_repo or index_folder call.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository identifier (owner/repo or just repo name)"
                    }
                },
                "required": ["repo"]
            }
        ),
        Tool(
            name="search_text",
            description="Full-text search across indexed file contents. Useful when symbol search misses (e.g., string literals, comments, config values). Use exact=true for punctuation-heavy queries like Foo::new(, enum variants, macro invocations. Check 'total_hits' — if it exceeds result_count, use offset/exhaustive to get more. Scans up to 50MB of content — use file_pattern to narrow scope on large repos.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository identifier (owner/repo or just repo name)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Text to search for"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional glob pattern to filter files (e.g., '*.py')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching lines to return (default 20, max 500)",
                        "default": 20
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip this many results before returning — use for pagination",
                        "default": 0
                    },
                    "exhaustive": {
                        "type": "boolean",
                        "description": "Return all results ignoring max_results cap.",
                        "default": False
                    },
                    "exact": {
                        "type": "boolean",
                        "description": "Case-sensitive exact substring match. Use for punctuation-heavy queries like `Foo::new(`, enum variants, macro invocations, exact log strings. Default is case-insensitive.",
                        "default": False
                    }
                },
                "required": ["repo", "query"]
            }
        ),
        Tool(
            name="get_repo_outline",
            description="Get a high-level overview of an indexed repository: directories, file counts, language breakdown, symbol counts. Lighter than get_file_tree.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository identifier (owner/repo or just repo name)"
                    }
                },
                "required": ["repo"]
            }
        ),
        Tool(
            name="find_references",
            description="Find all references to a symbol (calls, struct constructions, field accesses). Returns production_refs and test_refs counts separately. Cross-reference extraction currently supports Rust and Python; other repo languages return coverage warnings. Ambiguous short-name matches return candidates instead of conflated results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository identifier"},
                    "symbol_name": {"type": "string", "description": "Symbol name to look up (case-sensitive)"},
                    "production_only": {"type": "boolean", "description": "Exclude test-context references", "default": False},
                    "test_only": {"type": "boolean", "description": "Return only test-context references", "default": False}
                },
                "required": ["repo", "symbol_name"]
            }
        ),
        Tool(
            name="find_callers",
            description="Find all call sites for a function or method. Use to verify a function is actually called in production code. Cross-reference extraction currently supports Rust and Python; ambiguous short-name matches return candidates instead of merged results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository identifier"},
                    "symbol_name": {"type": "string", "description": "Function or method name (case-sensitive)"},
                    "production_only": {"type": "boolean", "description": "Exclude test-context callers", "default": False},
                    "test_only": {"type": "boolean", "description": "Return only test-context callers", "default": False}
                },
                "required": ["repo", "symbol_name"]
            }
        ),
        Tool(
            name="find_constructors",
            description="Find all construction sites for a struct or class (::new calls and struct literals). Use to verify a type is actually instantiated in production. Cross-reference extraction currently supports Rust and Python; ambiguous short-name matches return candidates instead of merged results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository identifier"},
                    "type_name": {"type": "string", "description": "Struct or class name (case-sensitive, e.g. 'SpectralAnalyzer')"},
                    "production_only": {"type": "boolean", "description": "Exclude test-context constructions", "default": False},
                    "test_only": {"type": "boolean", "description": "Return only test-context constructions", "default": False}
                },
                "required": ["repo", "type_name"]
            }
        ),
        Tool(
            name="find_field_reads",
            description="Find all read sites for a struct field or object attribute. Cross-reference extraction currently supports Rust and Python; other repo languages return coverage warnings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository identifier"},
                    "field_name": {"type": "string", "description": "Field or attribute name (case-sensitive)"},
                    "production_only": {"type": "boolean", "description": "Exclude test-context reads", "default": False}
                },
                "required": ["repo", "field_name"]
            }
        ),
        Tool(
            name="find_field_writes",
            description="Find all write sites for a struct field or object attribute. Cross-reference extraction currently supports Rust and Python; other repo languages return coverage warnings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository identifier"},
                    "field_name": {"type": "string", "description": "Field or attribute name (case-sensitive)"},
                    "production_only": {"type": "boolean", "description": "Exclude test-context writes", "default": False}
                },
                "required": ["repo", "field_name"]
            }
        ),
        Tool(
            name="add_to_watchlist",
            description="IMPORTANT: Only call this when the user explicitly asks to add a path to the watchlist. Do not call autonomously — it modifies security policy. Adds a local directory to the jcodemunch path watchlist; once added, query tools are allowed when the server runs from that directory and it is auto-refreshed before every query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the project directory (supports ~)"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="remove_from_watchlist",
            description="IMPORTANT: Only call this when the user explicitly asks to remove a path from the watchlist. Do not call autonomously. Removes a local directory from the jcodemunch path watchlist; after removal, query tools will be blocked when running from that directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to remove (must match the resolved path as it was added)"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="list_watched_paths",
            description="List all directories currently in the jcodemunch path watchlist.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    storage_path = os.environ.get("CODE_INDEX_PATH")

    if name in _REFRESH_TOOLS:
        if not auto_refresher.is_path_watched(os.getcwd()):
            cwd = os.getcwd()
            return [TextContent(type="text", text=json.dumps({
                "error": "PATH_GUARD_BLOCKED",
                "message": (
                    f"jcodemunch tools are blocked: current directory ({cwd!r}) is not in "
                    "the watchlist. Use Read/Grep/Glob for file exploration instead, or add "
                    "this project with add_to_watchlist."
                ),
            }, indent=2))]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, auto_refresher.maybe_refresh, storage_path)

    try:
        if name == "index_repo":
            result = await index_repo(
                url=arguments["url"],
                use_ai_summaries=arguments.get("use_ai_summaries", True),
                storage_path=storage_path,
                incremental=arguments.get("incremental", False),
            )
        elif name == "index_folder":
            folder_path = arguments["path"]
            if not auto_refresher.is_path_watched(folder_path):
                resolved = os.path.realpath(os.path.expanduser(str(folder_path)))
                result = {
                    "error": "PATH_GUARD_BLOCKED",
                    "message": (
                        f"index_folder blocked: {resolved!r} is not under a watched path. "
                        "Add a parent directory with add_to_watchlist first."
                    ),
                }
            else:
                result = index_folder(
                    path=folder_path,
                    use_ai_summaries=arguments.get("use_ai_summaries", True),
                    storage_path=storage_path,
                    extra_ignore_patterns=arguments.get("extra_ignore_patterns"),
                    follow_symlinks=arguments.get("follow_symlinks", False),
                    incremental=arguments.get("incremental", False),
                )
                if result.get("success"):
                    auto_refresher.register_path(folder_path)
        elif name == "add_to_watchlist":
            path = arguments["path"]
            added = auto_refresher.register_path(path)
            resolved = os.path.realpath(os.path.expanduser(str(path)))
            result = {"success": added, "path": resolved, "watched_paths": auto_refresher.get_watched_paths()}
            if not added:
                result["message"] = f"Watchlist is full ({MAX_WATCHED_PATHS} paths). Remove a path or edit autorefresh.json manually."
        elif name == "remove_from_watchlist":
            path = arguments["path"]
            removed = auto_refresher.remove_path(path)
            resolved = os.path.realpath(os.path.expanduser(str(path)))
            result = {"success": removed, "path": resolved, "watched_paths": auto_refresher.get_watched_paths()}
        elif name == "list_watched_paths":
            result = {"watched_paths": auto_refresher.get_watched_paths()}
        elif name == "list_repos":
            result = list_repos(storage_path=storage_path)
        elif name == "get_file_tree":
            result = get_file_tree(
                repo=arguments["repo"],
                path_prefix=arguments.get("path_prefix", ""),
                show_empty=arguments.get("show_empty", False),
                storage_path=storage_path
            )
        elif name == "get_file_outline":
            result = get_file_outline(
                repo=arguments["repo"],
                file_path=arguments["file_path"],
                storage_path=storage_path
            )
        elif name == "get_symbol":
            result = get_symbol(
                repo=arguments["repo"],
                symbol_id=arguments["symbol_id"],
                verify=arguments.get("verify", False),
                context_lines=arguments.get("context_lines", 0),
                storage_path=storage_path
            )
        elif name == "get_symbols":
            result = get_symbols(
                repo=arguments["repo"],
                symbol_ids=arguments["symbol_ids"],
                storage_path=storage_path
            )
        elif name == "search_symbols":
            result = search_symbols(
                repo=arguments["repo"],
                query=arguments["query"],
                kind=arguments.get("kind"),
                file_pattern=arguments.get("file_pattern"),
                language=arguments.get("language"),
                max_results=arguments.get("max_results", 10),
                offset=arguments.get("offset", 0),
                exhaustive=arguments.get("exhaustive", False),
                storage_path=storage_path
            )
        elif name == "invalidate_cache":
            result = invalidate_cache(
                repo=arguments["repo"],
                storage_path=storage_path
            )
        elif name == "search_text":
            result = search_text(
                repo=arguments["repo"],
                query=arguments["query"],
                file_pattern=arguments.get("file_pattern"),
                max_results=arguments.get("max_results", 20),
                offset=arguments.get("offset", 0),
                exhaustive=arguments.get("exhaustive", False),
                exact=arguments.get("exact", False),
                storage_path=storage_path
            )
        elif name == "get_repo_outline":
            result = get_repo_outline(
                repo=arguments["repo"],
                storage_path=storage_path
            )
        elif name == "find_references":
            result = find_references(
                repo=arguments["repo"],
                symbol_name=arguments["symbol_name"],
                production_only=arguments.get("production_only", False),
                test_only=arguments.get("test_only", False),
                storage_path=storage_path,
            )
        elif name == "find_callers":
            result = find_callers(
                repo=arguments["repo"],
                symbol_name=arguments["symbol_name"],
                production_only=arguments.get("production_only", False),
                test_only=arguments.get("test_only", False),
                storage_path=storage_path,
            )
        elif name == "find_constructors":
            result = find_constructors(
                repo=arguments["repo"],
                type_name=arguments["type_name"],
                production_only=arguments.get("production_only", False),
                test_only=arguments.get("test_only", False),
                storage_path=storage_path,
            )
        elif name == "find_field_reads":
            result = find_field_reads(
                repo=arguments["repo"],
                field_name=arguments["field_name"],
                production_only=arguments.get("production_only", False),
                storage_path=storage_path,
            )
        elif name == "find_field_writes":
            result = find_field_writes(
                repo=arguments["repo"],
                field_name=arguments["field_name"],
                production_only=arguments.get("production_only", False),
                storage_path=storage_path,
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


async def run_server():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def main(argv: Optional[list[str]] = None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="jcodemunch-mcp",
        description="Run the jCodeMunch MCP stdio server.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("JCODEMUNCH_LOG_LEVEL", "WARNING"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (also via JCODEMUNCH_LOG_LEVEL env var)",
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("JCODEMUNCH_LOG_FILE"),
        help="Log file path (also via JCODEMUNCH_LOG_FILE env var). Defaults to stderr.",
    )
    args = parser.parse_args(argv)

    log_level = getattr(logging, args.log_level)
    handlers: list[logging.Handler] = []
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    asyncio.run(run_server())


if __name__ == "__main__":
    main()

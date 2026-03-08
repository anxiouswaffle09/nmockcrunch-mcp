"""Search symbols across repository."""
import json

import os
import time
from typing import Optional

from ..storage import IndexStore, record_savings, estimate_savings, cost_avoided
from ._utils import resolve_repo


def search_symbols(
    repo: str,
    query: str,
    kind: Optional[str] = None,
    file_pattern: Optional[str] = None,
    language: Optional[str] = None,
    max_results: int = 10,
    offset: int = 0,
    exhaustive: bool = False,
    storage_path: Optional[str] = None
) -> dict:
    """Search for symbols matching a query.

    Args:
        repo: Repository identifier (owner/repo or just repo name).
        query: Search query.
        kind: Optional filter by symbol kind.
        file_pattern: Optional glob pattern to filter files.
        language: Optional filter by language (e.g., "python", "javascript").
        max_results: Maximum results to return (ignored when exhaustive=True).
        offset: Skip this many results before returning (for pagination).
        exhaustive: Return all results regardless of max_results cap.
        storage_path: Custom storage path.

    Returns:
        Dict with search results and _meta envelope.
    """
    start = time.perf_counter()
    max_results = max(1, min(max_results, 200))
    offset = max(0, offset)

    try:
        owner, name = resolve_repo(repo, storage_path)
    except ValueError as e:
        return {"error": str(e)}

    # Load index
    store = IndexStore(base_path=storage_path)
    index = store.load_index(owner, name)

    if not index:
        return {"error": f"Repository not indexed: {owner}/{name}"}

    # Search — returns ALL matching symbols, sorted by score, with language filter applied
    results = index.search(query, kind=kind, file_pattern=file_pattern, language=language)

    total_hits = len(results)

    # Paginate / cap
    if exhaustive:
        page = results[offset:]
    else:
        page = results[offset:offset + max_results]

    truncated = (not exhaustive) and (offset + len(page)) < total_hits

    # Format results — score is already embedded by CodeIndex.search
    scored_results = []
    for sym in page:
        scored_results.append({
            "id": sym["id"],
            "kind": sym["kind"],
            "name": sym["name"],
            "file": sym["file"],
            "line": sym["line"],
            "signature": sym["signature"],
            "summary": sym.get("summary", ""),
            "score": sym.get("score", 0),
        })

    # Token savings: files containing matches vs actual response size
    raw_bytes = 0
    seen_files: set = set()
    content_dir = store._content_dir(owner, name)
    for sym in page:
        f = sym["file"]
        if f not in seen_files:
            seen_files.add(f)
            try:
                raw_bytes += os.path.getsize(content_dir / f)
            except OSError:
                pass
    response_bytes = len(json.dumps(scored_results).encode())
    tokens_saved = estimate_savings(raw_bytes, response_bytes)
    total_saved = record_savings(tokens_saved)

    elapsed = (time.perf_counter() - start) * 1000

    response = {
        "repo": f"{owner}/{name}",
        "query": query,
        "result_count": len(scored_results),
        "total_hits": total_hits,
        "offset": offset,
        "results": scored_results,
        "_meta": {
            "timing_ms": round(elapsed, 1),
            "total_symbols": len(index.symbols),
            "truncated": truncated,
            "exhaustive": exhaustive,
            "tokens_saved": tokens_saved,
            "total_tokens_saved": total_saved,
            **cost_avoided(tokens_saved, total_saved),
        },
    }

    if truncated:
        response["warning"] = (
            f"WARNING: results truncated — showing {offset}–{offset + len(scored_results)} "
            f"of {total_hits} total matches. "
            f"Rerun with higher max_results, offset={offset + max_results} to page, "
            f"or exhaustive=true to get all."
        )

    return response

"""Get file outline - symbols in a specific file."""
import json

import os
import time
from typing import Optional

from ..storage import IndexStore, record_savings, estimate_savings, cost_avoided
from ..parser import build_symbol_tree_from_dicts
from ._utils import resolve_repo


def get_file_outline(
    repo: str,
    file_path: str,
    storage_path: Optional[str] = None
) -> dict:
    """Get symbols in a file with hierarchical structure.

    Args:
        repo: Repository identifier (owner/repo or just repo name)
        file_path: Path to file within repository
        storage_path: Custom storage path

    Returns:
        Dict with symbols outline
    """
    start = time.perf_counter()

    try:
        owner, name = resolve_repo(repo, storage_path)
    except ValueError as e:
        return {"error": str(e)}
    
    # Load index
    store = IndexStore(base_path=storage_path)
    index = store.load_index(owner, name)
    
    if not index:
        return {"error": f"Repository not indexed: {owner}/{name}"}
    
    # Filter symbols to this file
    file_symbols = [s for s in index.symbols if s.get("file") == file_path]
    
    if not file_symbols:
        return {
            "repo": f"{owner}/{name}",
            "file": file_path,
            "language": "",
            "symbols": []
        }
    
    # Build symbol tree directly from dicts (no roundtrip through Symbol dataclass)
    tree_nodes = build_symbol_tree_from_dicts(file_symbols)
    symbols_output = [_dict_node_to_output(n) for n in tree_nodes]
    
    # Get language
    language = file_symbols[0].get("language", "")
    
    elapsed = (time.perf_counter() - start) * 1000

    # Token savings: raw file size vs outline response size
    raw_bytes = 0
    try:
        raw_file = store._content_dir(owner, name) / file_path
        raw_bytes = os.path.getsize(raw_file)
    except OSError:
        pass
    response_bytes = len(json.dumps(symbols_output).encode())
    tokens_saved = estimate_savings(raw_bytes, response_bytes)
    total_saved = record_savings(tokens_saved)

    return {
        "repo": f"{owner}/{name}",
        "file": file_path,
        "language": language,
        "symbols": symbols_output,
        "_meta": {
            "timing_ms": round(elapsed, 1),
            "symbol_count": len(symbols_output),
            "tokens_saved": tokens_saved,
            "total_tokens_saved": total_saved,
            **cost_avoided(tokens_saved, total_saved),
        },
    }


def _dict_node_to_output(node: dict) -> dict:
    """Convert a build_symbol_tree_from_dicts node to output format."""
    result = {
        "id": node["id"],
        "kind": node["kind"],
        "name": node["name"],
        "signature": node["signature"],
        "summary": node.get("summary", ""),
        "line": node["line"],
    }
    children = node.get("_children", [])
    if children:
        result["children"] = [_dict_node_to_output(c) for c in children]
    return result

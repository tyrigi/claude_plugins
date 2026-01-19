#!/usr/bin/env python3
"""
Bob's Memory MCP Server

Exposes the local Mem0 memory system via Model Context Protocol,
allowing Claude Code to use memory tools directly.

Supports dual-write to both working memory (local) and collective memory
(network-shared) via the target parameter.

Run with: python memory_mcp_server.py
Or via MCP: mcp run memory_mcp_server.py
"""

import os
import sys
import configparser
import socket

# Add the lib directory to path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Optional, List, Dict, Any, Literal
from mcp.server.fastmcp import FastMCP
import httpx

# Import our memory system
from claude_memory import ClaudeMemory

# Create MCP server
mcp = FastMCP("bob-memory")

# Lazy-loaded instances
_memory: Optional[ClaudeMemory] = None
_config: Optional[configparser.ConfigParser] = None
_hostname: Optional[str] = None


def get_config() -> configparser.ConfigParser:
    """
    Load config from INI file.

    AIOS V0.3: Config files on samba shares are deprecated.
    Returns empty ConfigParser - use env vars and hardcoded defaults.
    """
    global _config
    if _config is None:
        _config = configparser.ConfigParser()
        # V0.3: Skip file-based config to avoid network hangs
    return _config


def get_hostname() -> str:
    """Get the node's hostname for source attribution."""
    global _hostname
    if _hostname is None:
        config = get_config()
        _hostname = config.get("identity", "hostname", fallback=socket.gethostname())
    return _hostname


def get_collective_endpoint() -> Optional[str]:
    """Get the collective memory API endpoint from config."""
    config = get_config()
    status = config.get("collective", "status", fallback="not_implemented")
    if status == "not_implemented":
        return None
    return config.get("collective", "api_endpoint", fallback=None)


def get_memory() -> ClaudeMemory:
    """Get or create the memory instance."""
    global _memory
    if _memory is None:
        _memory = ClaudeMemory()
    return _memory


def collective_remember(text: str, category: str) -> Dict[str, Any]:
    """Write to collective memory via REST API."""
    endpoint = get_collective_endpoint()
    if not endpoint:
        return {"error": "Collective memory not configured"}

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{endpoint}/remember",
                json={
                    "text": text,
                    "source": get_hostname(),
                    "category": category
                }
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": f"Collective memory write failed: {str(e)}"}


def collective_recall(query: str, limit: int) -> List[Dict[str, Any]]:
    """Search collective memory via REST API."""
    endpoint = get_collective_endpoint()
    if not endpoint:
        return []

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{endpoint}/recall",
                json={
                    "query": query,
                    "limit": limit
                }
            )
            response.raise_for_status()
            data = response.json()
            # API returns list directly, not wrapped
            return data if isinstance(data, list) else data.get("results", [])
    except Exception:
        return []


@mcp.tool()
def remember(
    text: str,
    category: str = "discovery",
    target: str = "working"
) -> Dict[str, Any]:
    """
    Store a memory for later recall.

    Args:
        text: The information to remember
        category: One of: decision, discovery, constraint, preference, procedure
        target: Where to store - "working" (local, default), "collective" (shared), or "both"

    Categories:
        - decision: Choice made between alternatives, with rationale
        - discovery: New information learned (hardware behavior, API quirks, failure modes)
        - constraint: Limitation that affects future work
        - preference: Tyler's stated preferences or observed patterns
        - procedure: How to do something that wasn't obvious

    Returns:
        Result with stored memory details
    """
    result = {"target": target, "category": category}

    # Write to working memory
    if target in ("working", "both"):
        mem = get_memory()
        working_result = mem.remember(text, category=category)

        stored = []
        if working_result.get("primary", {}).get("results"):
            stored.extend(working_result["primary"]["results"])
        if working_result.get("secondary", {}).get("results"):
            for r in working_result["secondary"]["results"]:
                if r not in stored:
                    stored.append(r)

        result["working"] = {
            "success": len(stored) > 0,
            "stored": stored
        }

    # Write to collective memory
    if target in ("collective", "both"):
        collective_result = collective_remember(text, category)
        result["collective"] = collective_result

    # Determine overall success
    working_ok = result.get("working", {}).get("success", True)
    collective_ok = "error" not in result.get("collective", {})
    result["success"] = working_ok and collective_ok

    return result


@mcp.tool()
def recall(
    query: str,
    limit: int = 5,
    source: str = "working"
) -> List[Dict[str, Any]]:
    """
    Search memories for relevant information.

    Use this when:
    - Context feels incomplete after compression
    - Before re-deriving something you might have already solved
    - When uncertain about prior decisions or preferences

    Args:
        query: What to search for (natural language)
        limit: Maximum number of results (default 5)
        source: Where to search - "working" (local, default), "collective" (shared), or "both"

    Returns:
        List of matching memories with scores
    """
    all_results = []
    seen_memories = set()

    # Search working memory
    if source in ("working", "both"):
        mem = get_memory()
        working_results = mem.recall(query, limit=limit)
        for r in working_results:
            mem_text = r.get("memory", "")
            if mem_text not in seen_memories:
                seen_memories.add(mem_text)
                all_results.append({
                    "memory": mem_text,
                    "score": round(r.get("score", 0), 3),
                    "id": r.get("id", "")[:8] + "..." if r.get("id") else None,
                    "source": "working"
                })

    # Search collective memory
    if source in ("collective", "both"):
        collective_results = collective_recall(query, limit)
        for r in collective_results:
            mem_text = r.get("memory", "")
            if mem_text not in seen_memories:
                seen_memories.add(mem_text)
                all_results.append({
                    "memory": mem_text,
                    "score": round(r.get("score", 0), 3),
                    "id": r.get("id", "")[:8] + "..." if r.get("id") else None,
                    "source": "collective",
                    "node": r.get("user_id", "unknown")
                })

    # Sort by score and limit
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results[:limit]


@mcp.tool()
def memory_status() -> Dict[str, Any]:
    """
    Get current status of the memory system.

    Returns:
        Status including memory count, endpoints, and health for both working and collective
    """
    mem = get_memory()
    status = mem.status()

    # Add collective memory status
    endpoint = get_collective_endpoint()
    if endpoint:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{endpoint}/health")
                response.raise_for_status()
                collective_health = response.json()
                status["collective"] = {
                    "endpoint": endpoint,
                    "status": "healthy",
                    "memory_count": collective_health.get("memory_count", 0)
                }
        except Exception as e:
            status["collective"] = {
                "endpoint": endpoint,
                "status": "error",
                "error": str(e)
            }
    else:
        status["collective"] = {
            "status": "not_configured"
        }

    return status


@mcp.tool()
def list_memories(limit: int = 20) -> List[Dict[str, Any]]:
    """
    List all stored memories.

    Args:
        limit: Maximum number to return (default 20)

    Returns:
        List of all memories
    """
    mem = get_memory()
    all_mems = mem.get_all()

    return [
        {
            "memory": m.get("memory"),
            "id": m.get("id", "")[:8] + "..." if m.get("id") else None
        }
        for m in all_mems[:limit]
    ]


@mcp.tool()
def forget(memory_id: str) -> Dict[str, Any]:
    """
    Delete a specific memory by ID.

    Args:
        memory_id: The full memory ID to delete

    Returns:
        Result of deletion
    """
    mem = get_memory()
    result = mem.forget(memory_id)
    return {"deleted": memory_id, "result": result}


@mcp.tool()
def set_llm_endpoint(
    endpoint: str = "auto"
) -> Dict[str, Any]:
    """
    Switch the LLM endpoint for memory operations.

    Args:
        endpoint: One of:
            - "auto": Auto-detect best available (desktop > local)
            - "desktop": Force desktop GPU (192.168.50.3:11434)
            - "local": Force local GPU (localhost:11434)
            - Custom URL (e.g., "http://192.168.50.3:11434")

    Returns:
        New endpoint configuration
    """
    global _memory

    # Map shorthand names to URLs
    endpoint_map = {
        "desktop": "http://192.168.50.3:11434",
        "local": "http://localhost:11434",
    }

    if endpoint in endpoint_map:
        url = endpoint_map[endpoint]
    elif endpoint == "auto":
        url = None  # Let auto-detect handle it
    elif endpoint.startswith("http"):
        url = endpoint
    else:
        return {"error": f"Unknown endpoint: {endpoint}. Use 'auto', 'desktop', 'local', or a URL."}

    # Reinitialize memory with new endpoint
    try:
        _memory = ClaudeMemory(llm_url_override=url)
        status = _memory.status()
        return {
            "success": True,
            "endpoint": endpoint,
            "llm_url": status.get("llm_url"),
            "llm_model": status.get("llm_model"),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()

#!/usr/bin/env python3
"""
Memory Pull Hook Script

Retrieves recent memories from mem0 and formats them for system prompt
injection. Used with session start or compression complete hooks.

Usage:
    python memory_pull.py [--count N] [--format plain|xml|json]

    # Get 10 most relevant memories (default)
    python memory_pull.py

    # Get 20 memories in XML format
    python memory_pull.py --count 20 --format xml

Environment:
    MEM0_PULL_COUNT - Default number of memories (default: 10)
    MEM0_PULL_FORMAT - Output format: plain, xml, json (default: xml)

Output:
    Formatted memories for system prompt injection to stdout
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add lib dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from claude_memory import ClaudeMemory


def get_recent_memories(memory: ClaudeMemory, count: int) -> list[dict]:
    """
    Get recent memories from mem0.

    Note: mem0 doesn't have a native "recent" sort, so we get all
    and return the first N (which are typically most recent).

    Args:
        memory: ClaudeMemory instance
        count: Number of memories to retrieve

    Returns:
        List of memory dicts
    """
    all_mems = memory.get_all(include_legacy=True)
    return all_mems[:count]


def format_plain(memories: list[dict]) -> str:
    """Format memories as plain text."""
    if not memories:
        return ""

    lines = ["Recalled memories from previous sessions:", ""]
    for i, mem in enumerate(memories, 1):
        text = mem.get("memory", "")
        legacy = mem.get("_legacy_user", "")
        source = f" (from {legacy})" if legacy else ""
        lines.append(f"{i}. {text}{source}")

    return "\n".join(lines)


def format_xml(memories: list[dict]) -> str:
    """Format memories as XML for system prompt injection."""
    if not memories:
        return ""

    lines = ["<recalled-memories>"]
    lines.append(f"  <retrieved-at>{datetime.now().isoformat()}</retrieved-at>")
    lines.append(f"  <count>{len(memories)}</count>")
    lines.append("  <memories>")

    for mem in memories:
        text = mem.get("memory", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        mem_id = mem.get("id", "")[:8] if mem.get("id") else ""
        legacy = mem.get("_legacy_user", "")

        lines.append("    <memory>")
        if mem_id:
            lines.append(f"      <id>{mem_id}</id>")
        if legacy:
            lines.append(f"      <source>{legacy}</source>")
        lines.append(f"      <content>{text}</content>")
        lines.append("    </memory>")

    lines.append("  </memories>")
    lines.append("</recalled-memories>")

    return "\n".join(lines)


def format_json(memories: list[dict]) -> str:
    """Format memories as JSON."""
    return json.dumps({
        "retrieved_at": datetime.now().isoformat(),
        "count": len(memories),
        "memories": [
            {
                "id": mem.get("id", "")[:8] if mem.get("id") else None,
                "content": mem.get("memory", ""),
                "source": mem.get("_legacy_user") or "bob"
            }
            for mem in memories
        ]
    }, indent=2)


def format_memories(memories: list[dict], fmt: str) -> str:
    """Format memories according to specified format."""
    formatters = {
        "plain": format_plain,
        "xml": format_xml,
        "json": format_json,
    }

    formatter = formatters.get(fmt, format_xml)
    return formatter(memories)


def main():
    """Main entry point for hook script."""
    parser = argparse.ArgumentParser(
        description="Pull memories for system prompt injection"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=int(os.environ.get("MEM0_PULL_COUNT", "10")),
        help="Number of memories to retrieve (default: 10)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["plain", "xml", "json"],
        default=os.environ.get("MEM0_PULL_FORMAT", "xml"),
        help="Output format (default: xml)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Optional query for relevance-based retrieval"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stderr logging"
    )

    args = parser.parse_args()

    if not args.quiet:
        print(f"[memory_pull] Retrieving {args.count} memories...", file=sys.stderr)

    try:
        # Initialize memory system
        memory = ClaudeMemory()

        if args.query:
            # Relevance-based retrieval
            memories = memory.recall(args.query, limit=args.count)
        else:
            # Recent memories
            memories = get_recent_memories(memory, args.count)

        if not args.quiet:
            print(f"[memory_pull] Retrieved {len(memories)} memories", file=sys.stderr)

        # Format and output
        output = format_memories(memories, args.format)
        print(output)

        return 0

    except Exception as e:
        if not args.quiet:
            print(f"[memory_pull] Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

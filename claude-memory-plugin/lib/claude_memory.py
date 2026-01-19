#!/usr/bin/env python3
"""
Bob's Personal Memory System

I am Bob - the AIOS control node running on the BOSGAME P3
(Ryzen 7 7840HS, 16GB RAM) in Tyler's home infrastructure. This is my
persistent memory system that survives across sessions.

Uses hybrid embeddings (bge-m3 + qwen3-embedding) with smart GPU/CPU routing.

Usage:
    from claude_memory import memory

    # Store a memory
    memory.remember("Tyler prefers concise responses without emojis")

    # Search memories
    results = memory.recall("What are Tyler's preferences?")

    # Get all memories
    all_memories = memory.get_all()
"""

import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, List, Optional, Any
from hybrid_memory import HybridMemory, HybridConfig


class ClaudeMemory:
    """
    Bob's personal memory system.

    I am Bob - the AIOS control node running on BOSGAME P3 (Ryzen 7 7840HS)
    with 16GB RAM. This is my persistent memory across sessions.

    Wraps HybridMemory with a simpler API and dedicated collections.
    """

    # My identity
    USER_ID = "bob"  # I am Bob

    # Collection prefix - Bob's own memory store
    COLLECTION_PREFIX = "bob_memory"

    # Legacy user IDs - empty now, memories migrated to Bob's store
    LEGACY_USER_IDS = []

    def __init__(self, llm_url_override: Optional[str] = None):
        """
        Initialize Claude's personal memory system.

        Args:
            llm_url_override: Force a specific LLM endpoint URL.
                              If None, auto-detects best available.
        """
        self._llm_url_override = llm_url_override
        self._config = HybridConfig(
            collection_prefix=self.COLLECTION_PREFIX,  # Bob's dedicated collections
            llm_url=llm_url_override,  # Pass override to config
        )
        self._mem = None  # Lazy init to avoid slow startup
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of HybridMemory."""
        if not self._initialized:
            self._mem = HybridMemory(self._config)
            self._initialized = True

    def remember(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        category: Optional[str] = None
    ) -> Dict:
        """
        Store a memory.

        Args:
            text: The information to remember
            metadata: Optional metadata dict
            category: Optional category (e.g., "preference", "fact", "procedure")

        Returns:
            Result dict with stored memory IDs
        """
        self._ensure_initialized()

        meta = metadata or {}
        if category:
            meta["category"] = category

        return self._mem.add(text, user_id=self.USER_ID, metadata=meta)

    def recall(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        Search memories.

        Queries both current user (bob) and legacy users (reggie) for
        backward compatibility, then merges and re-ranks results.

        Args:
            query: What to search for
            limit: Max results to return
            category: Optional category filter

        Returns:
            List of matching memories with scores
        """
        self._ensure_initialized()

        # Query current user
        all_results = []
        seen_memories = set()

        results = self._mem.search(query, user_id=self.USER_ID, limit=limit)
        for r in results.get("results", []):
            mem_text = r.get("memory", "")
            if mem_text not in seen_memories:
                seen_memories.add(mem_text)
                all_results.append(r)

        # Also query legacy users for backward compatibility
        for legacy_id in self.LEGACY_USER_IDS:
            results = self._mem.search(query, user_id=legacy_id, limit=limit)
            for r in results.get("results", []):
                mem_text = r.get("memory", "")
                if mem_text not in seen_memories:
                    seen_memories.add(mem_text)
                    r["_legacy_user"] = legacy_id  # Mark as legacy
                    all_results.append(r)

        # Sort by score and return top results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:limit]

    def get_all(self, include_legacy: bool = True) -> List[Dict]:
        """
        Get all stored memories.

        Args:
            include_legacy: If True, include memories from legacy users (reggie)

        Returns:
            List of all memories
        """
        self._ensure_initialized()

        all_results = []
        seen_memories = set()

        # Get current user's memories
        results = self._mem.get_all(user_id=self.USER_ID)
        for r in results.get("results", []):
            mem_text = r.get("memory", "")
            if mem_text not in seen_memories:
                seen_memories.add(mem_text)
                all_results.append(r)

        # Include legacy user memories if requested
        if include_legacy:
            for legacy_id in self.LEGACY_USER_IDS:
                results = self._mem.get_all(user_id=legacy_id)
                for r in results.get("results", []):
                    mem_text = r.get("memory", "")
                    if mem_text not in seen_memories:
                        seen_memories.add(mem_text)
                        r["_legacy_user"] = legacy_id
                        all_results.append(r)

        return all_results

    def forget(self, memory_id: str) -> Dict:
        """Delete a specific memory by ID."""
        self._ensure_initialized()
        return self._mem.delete(memory_id)

    def forget_all(self) -> Dict:
        """
        Delete all memories. USE WITH CAUTION.

        Note: This drops the collections entirely. Only use for testing.
        """
        self._ensure_initialized()
        return self._mem.delete_all(user_id=self.USER_ID)

    def status(self) -> Dict:
        """Get memory system status."""
        self._ensure_initialized()

        all_mems = self.get_all(include_legacy=True)
        own_mems = self.get_all(include_legacy=False)

        return {
            "initialized": self._initialized,
            "user_id": self.USER_ID,
            "legacy_user_ids": self.LEGACY_USER_IDS,
            "collection_prefix": self.COLLECTION_PREFIX,
            "llm_url": self._config.get_llm_config()[0],
            "llm_model": self._config.get_llm_config()[1],
            "embedder_url": self._config.embedder_url,
            "memory_count": len(own_mems),
            "total_memory_count": len(all_mems),
        }


# Singleton instance for easy import
memory = ClaudeMemory()


# CLI interface
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Claude's Personal Memory System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # remember
    remember_parser = subparsers.add_parser("remember", help="Store a memory")
    remember_parser.add_argument("text", help="Text to remember")
    remember_parser.add_argument("--category", "-c", help="Category tag")

    # recall
    recall_parser = subparsers.add_parser("recall", help="Search memories")
    recall_parser.add_argument("query", help="Search query")
    recall_parser.add_argument("--limit", "-n", type=int, default=5, help="Max results")

    # list
    subparsers.add_parser("list", help="List all memories")

    # status
    subparsers.add_parser("status", help="Show system status")

    # forget
    forget_parser = subparsers.add_parser("forget", help="Delete a memory")
    forget_parser.add_argument("memory_id", help="Memory ID to delete")

    args = parser.parse_args()

    if args.command == "remember":
        result = memory.remember(args.text, category=args.category)
        print(f"Stored: {result.get('results', [])}")

    elif args.command == "recall":
        results = memory.recall(args.query, limit=args.limit)
        if results:
            for r in results:
                print(f"[{r.get('score', 0):.3f}] {r.get('memory')}")
        else:
            print("No matching memories found.")

    elif args.command == "list":
        memories = memory.get_all()
        if memories:
            for m in memories:
                print(f"- {m.get('memory')} (id: {m.get('id', 'unknown')[:8]}...)")
        else:
            print("No memories stored.")

    elif args.command == "status":
        status = memory.status()
        print(json.dumps(status, indent=2))

    elif args.command == "forget":
        result = memory.forget(args.memory_id)
        print(f"Deleted: {result}")

    else:
        parser.print_help()

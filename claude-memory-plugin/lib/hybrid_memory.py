"""
Hybrid Memory System

Uses multiple embedding models for better coverage:
- bge-m3: Primary model, good at exact/semantic recall, fewer false positives
- qwen3-embedding: Secondary model, better at inference/conceptual queries

On add: Stores to both collections
On search: Queries both, merges and re-ranks results

Configuration priority:
1. Environment variables (MEM0_LLM_URL, MEM0_EMBEDDER_URL, etc.)
2. CONFIG_{identity}.ini file
3. Hardcoded defaults
"""

import os
import sys
import configparser
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mem0 import Memory


def _load_config_ini() -> dict:
    """
    Load config from AIOS CONFIG ini file.

    AIOS V0.3: Config files on samba shares are deprecated.
    All configuration should come from environment variables or hardcoded defaults.
    This function returns empty dict to skip file-based config entirely.
    """
    # V0.3: Skip file-based config - use env vars and defaults only
    # This avoids hangs when network shares are unavailable
    return {}


def _get_config_value(env_key: str, ini_key: str, default: Any, ini_config: dict) -> Any:
    """Get config value with priority: env > ini > default."""
    # Check environment first
    env_val = os.environ.get(env_key)
    if env_val is not None:
        # Handle type conversion for int
        if isinstance(default, int):
            return int(env_val)
        return env_val

    # Check ini config
    if ini_key in ini_config:
        return ini_config[ini_key]

    # Return default
    return default


@dataclass
class HybridConfig:
    """
    Configuration for hybrid memory system.

    Override via environment variables:
        MEM0_LLM_URL         - LLM endpoint (overrides auto-detect)
        MEM0_LLM_URL_PRIMARY - Primary LLM endpoint
        MEM0_LLM_URL_FALLBACK - Fallback LLM endpoint
        MEM0_LLM_MODEL       - LLM model name
        MEM0_EMBEDDER_URL    - Embeddings endpoint
        MEM0_QDRANT_HOST     - Qdrant host
        MEM0_QDRANT_PORT     - Qdrant port
        MEM0_PRIMARY_EMBEDDER   - Primary embedding model
        MEM0_SECONDARY_EMBEDDER - Secondary embedding model

    Or via CONFIG_{identity}.ini [memory] section.
    """
    # These are just defaults - __post_init__ applies overrides
    # LLM on localhost (DDR5 CPU faster than desktop DDR4 CPU)
    # Embeddings on desktop (bge-m3 runs on 4070Ti GPU)
    llm_model_primary: str = "qwen3:4b-instruct"
    llm_model: str = None
    llm_url: str = None
    llm_url_primary: str = "http://localhost:11434"
    llm_url_fallback: str = "http://daycare.local:11434"
    embedder_url: str = "http://192.168.50.3:11434"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_prefix: str = "aios_memory"
    primary_embedder: str = "bge-m3"
    primary_dims: int = 1024
    secondary_embedder: str = "qwen3-embedding:0.6b"
    secondary_dims: int = 1024
    confidence_threshold: float = 0.55

    def __post_init__(self):
        """Apply configuration overrides from env vars and config file."""
        ini_config = _load_config_ini()

        # Apply overrides (env > ini > current value)
        self.llm_url = _get_config_value("MEM0_LLM_URL", "llm_url", self.llm_url, ini_config)
        self.llm_url_primary = _get_config_value("MEM0_LLM_URL_PRIMARY", "llm_url_primary", self.llm_url_primary, ini_config)
        self.llm_url_fallback = _get_config_value("MEM0_LLM_URL_FALLBACK", "llm_url_fallback", self.llm_url_fallback, ini_config)
        self.llm_model = _get_config_value("MEM0_LLM_MODEL", "llm_model", self.llm_model, ini_config)
        self.embedder_url = _get_config_value("MEM0_EMBEDDER_URL", "embedder_url", self.embedder_url, ini_config)
        self.qdrant_host = _get_config_value("MEM0_QDRANT_HOST", "qdrant_host", self.qdrant_host, ini_config)
        self.qdrant_port = _get_config_value("MEM0_QDRANT_PORT", "qdrant_port", self.qdrant_port, ini_config)
        self.primary_embedder = _get_config_value("MEM0_PRIMARY_EMBEDDER", "primary_embedder", self.primary_embedder, ini_config)
        self.secondary_embedder = _get_config_value("MEM0_SECONDARY_EMBEDDER", "secondary_embedder", self.secondary_embedder, ini_config)

    def get_llm_config(self) -> tuple:
        """
        Get LLM URL and model, auto-detecting endpoint availability.

        Returns:
            (url, model) tuple
        """
        model = self.llm_model or self.llm_model_primary

        # If explicit URL override is set, use it (skip auto-detect)
        if self.llm_url:
            return self.llm_url, model

        # Auto-detect: try primary endpoint (desktop GPU) first
        import socket
        try:
            host = self.llm_url_primary.split("://")[1].split(":")[0]
            port = int(self.llm_url_primary.split(":")[-1])
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return self.llm_url_primary, model
        except:
            pass

        return self.llm_url_fallback, model

    def get_llm_url(self) -> str:
        """Get LLM URL (for backwards compatibility)."""
        url, _ = self.get_llm_config()
        return url


class HybridMemory:
    """
    Memory system using dual embedding models.

    Stores facts in both embedding spaces and merges search results.
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()

        # Detect LLM config first (needed for _make_config)
        self._llm_url, self._llm_model = self.config.get_llm_config()

        # Create primary memory (bge-m3)
        self.primary = Memory.from_config(self._make_config(
            self.config.primary_embedder,
            self.config.primary_dims,
            f"{self.config.collection_prefix}_primary"
        ))

        # Create secondary memory (qwen3-embedding)
        self.secondary = Memory.from_config(self._make_config(
            self.config.secondary_embedder,
            self.config.secondary_dims,
            f"{self.config.collection_prefix}_secondary"
        ))

        print(f"HybridMemory initialized:", file=sys.stderr)
        print(f"  LLM: {self._llm_model} @ {self._llm_url}", file=sys.stderr)
        print(f"  Primary embedder: {self.config.primary_embedder}", file=sys.stderr)
        print(f"  Secondary embedder: {self.config.secondary_embedder}", file=sys.stderr)

    def _make_config(self, embedder: str, dims: int, collection: str) -> Dict:
        """Generate Mem0 config for a specific embedder."""
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection,
                    "host": self.config.qdrant_host,
                    "port": self.config.qdrant_port,
                    "embedding_model_dims": dims,
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self._llm_model,
                    "temperature": 0,
                    "max_tokens": 2000,
                    "ollama_base_url": self._llm_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": embedder,
                    "ollama_base_url": self.config.embedder_url,
                },
            },
            "version": "v1.1",
        }

    def add(self, text: str, user_id: str, metadata: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Add memory to both embedding spaces.

        Returns combined result from both stores.
        """
        results = {"primary": None, "secondary": None, "results": []}

        # Add to primary (bge-m3)
        try:
            primary_result = self.primary.add(text, user_id=user_id, metadata=metadata, **kwargs)
            results["primary"] = primary_result
            if primary_result.get("results"):
                results["results"].extend(primary_result["results"])
        except Exception as e:
            print(f"Primary add failed: {e}")

        # Add to secondary (qwen3-embedding)
        try:
            secondary_result = self.secondary.add(text, user_id=user_id, metadata=metadata, **kwargs)
            results["secondary"] = secondary_result
        except Exception as e:
            print(f"Secondary add failed: {e}")

        return results

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        merge_strategy: str = "smart",  # "smart", "union", "primary_first", "primary_gated"
        **kwargs
    ) -> Dict:
        """
        Search both embedding spaces and merge results.

        Strategies:
        - smart: Use primary (bge-m3), boost with secondary only for low-confidence results
        - union: Combine results, dedupe by content, sort by max score
        - primary_first: Use primary results, add secondary if needed
        - primary_gated: Only use secondary if primary has low confidence
        """
        primary_results = []
        secondary_results = []

        # Search primary (bge-m3) - better at rejecting false positives
        try:
            p_res = self.primary.search(query, user_id=user_id, limit=limit, **kwargs)
            primary_results = p_res.get("results", [])
        except Exception as e:
            print(f"Primary search failed: {e}")

        # Determine if we need secondary search
        need_secondary = True
        if merge_strategy in ["smart", "primary_gated"]:
            # Only query secondary if primary results are low confidence
            if primary_results:
                top_score = primary_results[0].get("score", 0)
                # If primary has good confidence, skip secondary to avoid false positives
                if top_score >= self.config.confidence_threshold:
                    need_secondary = False

        # Search secondary (qwen3-embedding) - better at inference
        if need_secondary:
            try:
                s_res = self.secondary.search(query, user_id=user_id, limit=limit, **kwargs)
                secondary_results = s_res.get("results", [])
            except Exception as e:
                print(f"Secondary search failed: {e}")

        # Merge results
        if merge_strategy == "smart":
            merged = self._merge_smart(primary_results, secondary_results)
        elif merge_strategy == "union":
            merged = self._merge_union(primary_results, secondary_results)
        elif merge_strategy == "primary_first":
            merged = self._merge_primary_first(primary_results, secondary_results, limit)
        elif merge_strategy == "primary_gated":
            # If we got secondary results, use them; otherwise use primary
            merged = secondary_results if secondary_results else primary_results
        else:
            merged = primary_results  # fallback

        # Sort by score and limit
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)

        return {"results": merged[:limit]}

    def _merge_smart(self, primary: List, secondary: List) -> List:
        """
        Smart merge: Use primary scores as base, only boost from secondary
        if it finds something primary missed with high confidence.
        """
        # Start with primary results
        result = {r.get("memory", ""): r.copy() for r in primary}

        # Only add secondary results that are high confidence AND not already well-represented
        for r in secondary:
            mem = r.get("memory", "")
            sec_score = r.get("score", 0)

            if mem in result:
                # Already have this memory - only boost if secondary score is much higher
                prim_score = result[mem].get("score", 0)
                if sec_score > prim_score + 0.15:  # Significant boost needed
                    # Use average instead of max to dampen false positives
                    result[mem]["score"] = (prim_score + sec_score) / 2
            else:
                # New memory from secondary - only add if very high confidence
                if sec_score >= 0.65:
                    result[mem] = r

        return list(result.values())

    def _merge_union(self, primary: List, secondary: List) -> List:
        """Combine results, dedupe by memory content, keep highest score."""
        seen = {}  # memory text -> result with highest score

        for r in primary + secondary:
            mem = r.get("memory", "")
            score = r.get("score", 0)
            if mem not in seen or score > seen[mem].get("score", 0):
                seen[mem] = r

        return list(seen.values())

    def _merge_primary_first(self, primary: List, secondary: List, limit: int) -> List:
        """Use primary results, fill with secondary if not enough."""
        result = primary.copy()
        seen_memories = {r.get("memory", "") for r in result}

        for r in secondary:
            if len(result) >= limit:
                break
            mem = r.get("memory", "")
            if mem not in seen_memories:
                result.append(r)
                seen_memories.add(mem)

        return result

    def _merge_max_score(self, primary: List, secondary: List) -> List:
        """For each memory, use the highest score from either model."""
        return self._merge_union(primary, secondary)  # Same logic

    def get_all(self, user_id: str, **kwargs) -> Dict:
        """Get all memories (from primary only to avoid dupes)."""
        return self.primary.get_all(user_id=user_id, **kwargs)

    def delete(self, memory_id: str, **kwargs) -> Dict:
        """Delete from both stores."""
        results = {}
        try:
            results["primary"] = self.primary.delete(memory_id, **kwargs)
        except:
            pass
        try:
            results["secondary"] = self.secondary.delete(memory_id, **kwargs)
        except:
            pass
        return results

    def delete_all(self, user_id: str, **kwargs) -> Dict:
        """Delete all memories for user from both stores."""
        results = {}
        try:
            results["primary"] = self.primary.delete_all(user_id=user_id, **kwargs)
        except:
            pass
        try:
            results["secondary"] = self.secondary.delete_all(user_id=user_id, **kwargs)
        except:
            pass
        return results


# Quick test
if __name__ == "__main__":
    import time

    print("Testing HybridMemory...")

    mem = HybridMemory()

    # Clean up
    mem.delete_all(user_id="hybrid_test")

    # Add a fact
    print("\nAdding memory...")
    start = time.time()
    result = mem.add("Ollama handles local LLM inference at port 11434", user_id="hybrid_test")
    print(f"Add took {time.time()-start:.1f}s")
    print(f"Result: {result.get('results', [])}")

    # Search with different queries
    print("\nSearching...")

    for query in ["Ollama", "machine learning inference", "what handles AI workloads"]:
        start = time.time()
        results = mem.search(query, user_id="hybrid_test", limit=1)
        elapsed = time.time() - start
        if results.get("results"):
            top = results["results"][0]
            print(f"  '{query}': score={top.get('score', 0):.3f} ({elapsed:.2f}s)")
        else:
            print(f"  '{query}': no results ({elapsed:.2f}s)")

    # Cleanup
    mem.delete_all(user_id="hybrid_test")
    print("\nDone!")

#!/usr/bin/env python3
"""
Memory Queue Worker - Async Memory Storage

Processes memories from the queue file and stores them in Mem0.
Runs as a background service, decoupling extraction (fast, sync)
from storage (slow, async).

Queue format (JSONL):
    {"content": "...", "category": "...", "timestamp": ..., "llm_url": "..."}

Usage:
    python memory_worker.py              # Run once, process queue
    python memory_worker.py --daemon     # Run continuously
    python memory_worker.py --status     # Show queue status
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Add lib dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from claude_memory import ClaudeMemory
from platform_utils import (
    get_queue_path, get_failed_path, get_log_dir,
    ensure_data_dir, ensure_log_dir, file_lock
)

# Configuration (paths from platform_utils)
QUEUE_PATH = get_queue_path()
FAILED_PATH = get_failed_path()
POLL_INTERVAL = 10  # seconds between queue checks in daemon mode
MAX_RETRIES = 3
BATCH_SIZE = 10  # Process up to N memories per cycle

# Configure logging
ensure_log_dir()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [memory_worker] %(message)s',
    handlers=[
        logging.FileHandler(get_log_dir() / 'memory_worker.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
log = logging.getLogger(__name__)


def read_queue() -> List[Dict]:
    """Read all entries from the queue file."""
    if not QUEUE_PATH.exists():
        return []

    entries = []
    try:
        with open(QUEUE_PATH, "r") as f:
            with file_lock(f, exclusive=False):  # Shared lock for reading
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            log.warning(f"Skipping malformed queue entry: {line[:50]}...")
    except Exception as e:
        log.error(f"Failed to read queue: {e}")

    return entries


def write_queue(entries: List[Dict]):
    """Rewrite the queue file with remaining entries."""
    ensure_data_dir()

    try:
        with open(QUEUE_PATH, "w") as f:
            with file_lock(f, exclusive=True):  # Exclusive lock for writing
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.error(f"Failed to write queue: {e}")


def append_failed(entry: Dict, error: str):
    """Append a failed entry to the failed queue for later inspection."""
    ensure_data_dir()

    entry["_error"] = error
    entry["_failed_at"] = time.time()

    try:
        with open(FAILED_PATH, "a") as f:
            with file_lock(f, exclusive=True):  # Exclusive lock for appending
                f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.error(f"Failed to write to failed queue: {e}")


def store_memory(entry: Dict, mem: ClaudeMemory) -> bool:
    """
    Store a single memory in Mem0.

    Returns:
        True if stored successfully, False otherwise
    """
    content = entry.get("content", "")
    category = entry.get("category", "general")

    if not content:
        log.warning("Empty content, skipping")
        return True  # Consider it "processed"

    try:
        result = mem.remember(content, category=category)

        # Check if stored in either primary or secondary
        primary_ok = result.get("primary", {}).get("results")
        secondary_ok = result.get("secondary", {}).get("results")

        if primary_ok or secondary_ok:
            log.info(f"[{category}] Stored: {content[:60]}...")
            return True
        else:
            log.info(f"[{category}] NOOP (duplicate): {content[:60]}...")
            return True  # Duplicate is still "processed"

    except Exception as e:
        log.error(f"Failed to store memory: {e}")
        return False


def process_queue(batch_size: int = BATCH_SIZE) -> int:
    """
    Process entries from the queue.

    Returns:
        Number of entries processed
    """
    entries = read_queue()

    if not entries:
        return 0

    log.info(f"Processing queue: {len(entries)} entries")

    # Initialize memory system once for the batch
    mem = ClaudeMemory()

    processed = 0
    remaining = []

    for i, entry in enumerate(entries):
        if processed >= batch_size:
            # Keep remaining entries for next cycle
            remaining.extend(entries[i:])
            break

        retries = entry.get("_retries", 0)

        if store_memory(entry, mem):
            processed += 1
        else:
            # Failed - retry or move to failed queue
            if retries < MAX_RETRIES:
                entry["_retries"] = retries + 1
                remaining.append(entry)
                log.warning(f"Will retry ({retries + 1}/{MAX_RETRIES}): {entry.get('content', '')[:40]}...")
            else:
                append_failed(entry, "Max retries exceeded")
                log.error(f"Max retries exceeded, moved to failed queue: {entry.get('content', '')[:40]}...")
                processed += 1  # Consider it "processed" (moved to failed)

    # Write remaining entries back to queue
    write_queue(remaining)

    log.info(f"Processed {processed} entries, {len(remaining)} remaining")
    return processed


def daemon_loop():
    """Run continuously, processing queue periodically."""
    log.info(f"Starting memory worker daemon (poll interval: {POLL_INTERVAL}s)")

    while True:
        try:
            processed = process_queue()
            if processed > 0:
                log.info(f"Cycle complete: processed {processed} memories")
        except Exception as e:
            log.error(f"Error in daemon loop: {e}")

        time.sleep(POLL_INTERVAL)


def show_status():
    """Show queue status."""
    entries = read_queue()
    failed_count = 0

    if FAILED_PATH.exists():
        with open(FAILED_PATH, "r") as f:
            failed_count = sum(1 for _ in f)

    print(f"Queue: {len(entries)} pending")
    print(f"Failed: {failed_count}")

    if entries:
        print("\nPending entries:")
        for entry in entries[:5]:
            content = entry.get("content", "")[:50]
            category = entry.get("category", "?")
            retries = entry.get("_retries", 0)
            print(f"  [{category}] {content}... (retries: {retries})")
        if len(entries) > 5:
            print(f"  ... and {len(entries) - 5} more")


def main():
    parser = argparse.ArgumentParser(description="Memory Queue Worker")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--status", action="store_true", help="Show queue status")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Entries per cycle")

    args = parser.parse_args()

    if args.status:
        show_status()
        return 0

    if args.daemon:
        daemon_loop()
    else:
        processed = process_queue(batch_size=args.batch_size)
        print(f"Processed {processed} memories")

    return 0


if __name__ == "__main__":
    sys.exit(main())

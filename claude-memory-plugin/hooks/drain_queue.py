#!/usr/bin/env python3
"""
Drain Queue Script - Waits for memory queue to empty before returning.

Used as ExecStop for memory-worker.service to prevent shutdown
while memories are still being processed.

Usage:
    python drain_queue.py [--timeout SECONDS]
"""

import sys
import time
import argparse
from pathlib import Path

# Add lib dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from platform_utils import get_queue_path

QUEUE_PATH = get_queue_path()
POLL_INTERVAL = 5  # seconds between checks


def get_queue_count() -> int:
    """Count entries in the queue file."""
    if not QUEUE_PATH.exists():
        return 0
    try:
        with open(QUEUE_PATH, 'r') as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Wait for memory queue to drain")
    parser.add_argument("--timeout", type=int, default=600, help="Max wait time in seconds")
    args = parser.parse_args()

    start = time.time()
    deadline = start + args.timeout

    count = get_queue_count()
    if count == 0:
        print("Queue already empty")
        return 0

    print(f"Waiting for queue to drain ({count} pending)...")

    while time.time() < deadline:
        count = get_queue_count()
        if count == 0:
            elapsed = time.time() - start
            print(f"Queue drained in {elapsed:.1f}s")
            return 0

        remaining = int(deadline - time.time())
        print(f"Queue: {count} pending, {remaining}s until timeout")
        time.sleep(POLL_INTERVAL)

    # Timeout reached
    count = get_queue_count()
    print(f"Timeout reached with {count} entries still pending")
    return 1


if __name__ == "__main__":
    sys.exit(main())

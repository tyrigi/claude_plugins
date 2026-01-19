"""
Platform utilities for cross-platform compatibility.

Handles:
- File locking (fcntl on Unix, msvcrt on Windows)
- Platform-appropriate paths for queue, logs, etc.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

# Platform detection
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')


def get_data_dir() -> Path:
    """Get platform-appropriate data directory for the memory system."""
    env_override = os.environ.get("MEM0_DATA_DIR")
    if env_override:
        return Path(env_override)

    if IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "claude-memory"
    elif IS_MACOS:
        return Path.home() / "Library" / "Application Support" / "claude-memory"
    else:  # Linux and others
        # Prefer /var/lib if we have write access, otherwise use user dir
        system_dir = Path("/var/lib/aios")
        if system_dir.exists() and os.access(system_dir, os.W_OK):
            return system_dir
        # Fall back to XDG state dir or home
        xdg_state = os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
        return Path(xdg_state) / "claude-memory"


def get_log_dir() -> Path:
    """Get platform-appropriate log directory."""
    env_override = os.environ.get("MEM0_LOG_DIR")
    if env_override:
        return Path(env_override)

    if IS_WINDOWS:
        return get_data_dir() / "logs"
    elif IS_MACOS:
        return Path.home() / "Library" / "Logs" / "claude-memory"
    else:  # Linux
        return Path("/tmp")  # Or could use /var/log with proper permissions


def get_queue_path() -> Path:
    """Get path to the memory queue file."""
    return get_data_dir() / "memory_queue.jsonl"


def get_failed_path() -> Path:
    """Get path to the failed memories file."""
    return get_data_dir() / "memory_failed.jsonl"


def ensure_data_dir():
    """Ensure the data directory exists."""
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def ensure_log_dir():
    """Ensure the log directory exists."""
    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# Cross-platform file locking
if IS_WINDOWS:
    import msvcrt

    def _lock_file(f, exclusive: bool = True):
        """Lock a file on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK, 1)

    def _unlock_file(f):
        """Unlock a file on Windows."""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except:
            pass  # May already be unlocked
else:
    import fcntl

    def _lock_file(f, exclusive: bool = True):
        """Lock a file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def _unlock_file(f):
        """Unlock a file on Unix."""
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextmanager
def file_lock(f, exclusive: bool = True):
    """
    Context manager for cross-platform file locking.

    Usage:
        with open(path, 'r') as f:
            with file_lock(f, exclusive=False):  # Shared lock for reading
                data = f.read()

        with open(path, 'w') as f:
            with file_lock(f, exclusive=True):  # Exclusive lock for writing
                f.write(data)
    """
    try:
        _lock_file(f, exclusive)
        yield
    finally:
        _unlock_file(f)

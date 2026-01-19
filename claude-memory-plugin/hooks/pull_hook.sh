#!/bin/bash
# Memory Pull Hook - SessionStart handler
# Recalls relevant memories at session start

# UPDATE: Set these paths for your installation
PLUGIN_PATH="${PLUGIN_PATH:-$HOME/claude-memory-plugin}"
VENV_PATH="${VENV_PATH:-$HOME/mem0_env}"

# Activate venv and run recall
source "$VENV_PATH/bin/activate"
cd "$PLUGIN_PATH"
exec python hooks/memory_pull.py --count 10 --format xml

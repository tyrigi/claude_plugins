#!/bin/bash
# Memory Push Hook - Stop/PreCompact handler
# Extracts memories from transcript and queues for async storage

# UPDATE: Set these paths for your installation
PLUGIN_PATH="${PLUGIN_PATH:-$HOME/claude-memory-plugin}"
VENV_PATH="${VENV_PATH:-$HOME/mem0_env}"

# Activate venv and run extractor
source "$VENV_PATH/bin/activate"
cd "$PLUGIN_PATH"
exec python hooks/memory_push.py

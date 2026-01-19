# Claude Memory Plugin

Automatic, background-processed persistent memory for Claude Code using Mem0.

## Features

- **Automatic Memory Extraction** - Stop hook extracts preferences, decisions, and insights from each conversation turn
- **Async Processing** - Queue-based architecture returns immediately while memories are stored in background
- **Hybrid Embeddings** - Dual-model embedding (bge-m3 + qwen3-embedding) for robust semantic search
- **GPU Accelerated** - Supports NVIDIA (CUDA) and AMD (ROCm) for fast inference
- **Graceful Shutdown** - Systemd service drains queue before shutdown (10 min timeout)
- **MCP Integration** - Full MCP server for manual memory operations

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Claude Code    │────▶│  Stop Hook   │────▶│  Memory Queue   │
│  (conversation) │     │  (extract)   │     │  (JSONL file)   │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  SessionStart   │◀────│   Mem0 DB    │◀────│  Worker Daemon  │
│  Hook (recall)  │     │   (Qdrant)   │     │  (async store)  │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## Quick Start

See [skills/setup.md](skills/setup.md) for detailed deployment instructions.

### Prerequisites
- Linux with systemd
- Python 3.10+
- Qdrant (vector database)
- Ollama (local LLM)

### Install

```bash
# Clone plugin
git clone <repo> ~/claude-memory-plugin
cd ~/claude-memory-plugin

# Create Python environment
python3 -m venv ~/mem0_env
source ~/mem0_env/bin/activate
pip install -r requirements.txt

# Run setup skill
# Follow steps in skills/setup.md
```

## Directory Structure

```
claude-memory-plugin/
├── README.md
├── requirements.txt
├── lib/                    # Core library
│   ├── claude_memory.py    # High-level memory interface
│   └── hybrid_memory.py    # Dual-embedding backend
├── mcp/                    # MCP server
│   └── memory_mcp_server.py
├── hooks/                  # Claude Code hooks
│   ├── memory_push.py      # Stop hook - extraction
│   ├── memory_pull.py      # SessionStart hook - recall
│   ├── memory_worker.py    # Background queue processor
│   ├── drain_queue.py      # Shutdown drain script
│   ├── push_hook.sh        # Shell wrapper
│   └── pull_hook.sh        # Shell wrapper
├── skills/                 # Skills
│   └── setup.md            # Deployment guide
├── systemd/                # Service files
│   └── memory-worker.service
└── config/                 # Configuration templates
    ├── hooks.json          # Claude Code hooks config
    └── mcp-server.json     # MCP server config
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_LLM_URL` | auto-detect | Ollama endpoint |
| `MEM0_EMBEDDER_URL` | localhost:11434 | Embeddings endpoint |
| `MEM0_QDRANT_URL` | localhost:6333 | Qdrant endpoint |
| `MEM0_EXTRACT_MODEL` | qwen3:4b-instruct | Extraction model |

### Memory Categories

Memories are automatically categorized:
- `preference` - User preferences and requirements
- `decision` - Architecture choices with reasoning
- `insight` - Debugging discoveries and corrections
- `state` - Project state (disabled by default)

## MCP Tools

When the MCP server is configured, these tools are available:

| Tool | Description |
|------|-------------|
| `recall(query, limit)` | Search memories by semantic similarity |
| `remember(content, category)` | Manually store a memory |
| `memory_status()` | Check system health |
| `list_memories(limit)` | List recent memories |
| `forget(memory_id)` | Delete a specific memory |

## Monitoring

```bash
# Queue status
python hooks/memory_worker.py --status

# Worker logs
journalctl -u memory-worker -f

# Push hook logs
tail -f /tmp/memory_push.log

# Worker processing logs
tail -f /tmp/memory_worker.log
```

## License

MIT

# Memory System Setup Skill

Deploy the Claude Memory Plugin with automatic adaptation to the target system.

## Philosophy

This skill guides deployment but does NOT prescribe exact commands. The deploying Claude instance should:
1. **Detect** the target environment (OS, GPU, existing services)
2. **Adapt** installation methods and paths to the platform
3. **Verify** each component before proceeding
4. **Document** the specific configuration chosen

## Component Overview

| Component | Purpose | Alternatives |
|-----------|---------|--------------|
| Vector DB | Memory storage | Qdrant (recommended), Chroma, Milvus |
| LLM Runtime | Extraction & dedup | Ollama (recommended), llama.cpp, vLLM |
| Embeddings | Semantic search | bge-m3, qwen3-embedding, OpenAI API |
| Queue | Async processing | File-based (included), Redis, SQLite |
| Background Worker | Process queue | systemd, launchd, Windows Service, supervisor |

## Phase 1: Environment Detection

Before installing anything, detect and document:

### Operating System
```
- Linux (which distro? has systemd?)
- macOS (which version? Apple Silicon or Intel?)
- Windows (WSL available? native?)
```

### GPU Availability
```
- NVIDIA GPU → CUDA (check nvidia-smi)
- AMD GPU → ROCm on Linux, Metal on macOS (check rocminfo or system_profiler)
- Intel GPU → oneAPI on Linux, Metal on macOS
- CPU only → Still works, just slower
```

### Existing Services
```
- Qdrant already running? (check port 6333)
- Ollama already installed? (check ollama --version)
- Docker available? (often easiest for Qdrant)
```

### Python Environment
```
- Python version (3.10+ required)
- Existing venv to use or create new?
- pip or conda?
```

## Phase 2: Vector Database (Qdrant)

### Decision Points
- **Docker available?** → Easiest deployment, handles persistence
- **Native install preferred?** → Download binary for platform
- **Remote Qdrant?** → Just need the URL, skip local install

### Platform Considerations

| Platform | Recommended Method | Data Path |
|----------|-------------------|-----------|
| Linux | Docker or native binary | `/var/lib/qdrant` or `~/.local/share/qdrant` |
| macOS | Docker or Homebrew | `~/Library/Application Support/qdrant` |
| Windows | Docker Desktop or native | `%LOCALAPPDATA%\qdrant` |

### Storage Location Considerations

**The vector database can grow significantly.** Before deploying, evaluate:

1. **Available drives and their purposes**
   - OS drive (often SSD, limited space)
   - Data drives (HDD or large SSD, bulk storage)
   - NAS or network storage (slower but larger)

2. **Storage requirements**
   - Each memory: ~4KB (embeddings + metadata)
   - 10,000 memories ≈ 40MB
   - 100,000 memories ≈ 400MB
   - Collections also have index overhead

3. **Performance vs capacity tradeoffs**
   - SSD: Fast queries, limited space
   - HDD: Slower but adequate for most uses, more space
   - Network: High latency, use only if local storage unavailable

4. **Questions to ask the user**
   - "Where should the memory database be stored?"
   - "Is there a dedicated data drive separate from the OS drive?"
   - "Should this survive OS reinstalls?" (→ put on separate partition/drive)

5. **Docker volume mapping**
   ```bash
   # Map to specific location instead of default
   docker run -v /path/to/bulk/storage/qdrant:/qdrant/storage qdrant/qdrant
   ```

6. **Environment variable override**
   ```bash
   # For native Qdrant
   export QDRANT__STORAGE__STORAGE_PATH=/path/to/bulk/storage/qdrant
   ```

### Verification
- HTTP endpoint responds: `GET http://localhost:6333/collections`
- Can create/delete test collection

## Phase 3: LLM Runtime (Ollama)

### Decision Points
- **GPU available?** → Ensure Ollama detects it
- **Multiple GPUs?** → May need CUDA_VISIBLE_DEVICES or similar
- **Remote Ollama?** → Just need URL, skip local install

### Required Models
```
qwen3:4b-instruct    # Memory extraction (small, fast)
bge-m3               # Primary embeddings
qwen3-embedding:0.6b # Secondary embeddings (optional, improves recall)
```

### GPU Acceleration Notes

| GPU Type | Framework | Setup Complexity |
|----------|-----------|------------------|
| NVIDIA | CUDA | Usually automatic with Ollama |
| AMD (discrete) | ROCm | Linux only, may need HSA_OVERRIDE_GFX_VERSION |
| AMD (iGPU) | ROCm | Needs HSA_OVERRIDE_GFX_VERSION for RDNA3 |
| Apple Silicon | Metal | Automatic with Ollama |
| Intel | oneAPI/SYCL | Limited support, check Ollama docs |

### Verification
- Model loads: `ollama run qwen3:4b-instruct "test"`
- GPU detected (if applicable): check `size_vram` in verbose output

## Phase 4: Python Environment

### Decision Points
- **Isolated venv?** → Recommended to avoid conflicts
- **System Python?** → Works but may conflict with other packages
- **Conda?** → Fine, adjust activation commands

### Required Packages
```
mem0ai        # Core memory library
httpx         # HTTP client for Ollama API
mcp           # MCP server framework
qdrant-client # Vector DB client (usually pulled by mem0ai)
```

### Platform Path Conventions

| Platform | Venv Location | Plugin Location |
|----------|---------------|-----------------|
| Linux | `~/mem0_env` or `~/.venv/mem0` | `~/.local/share/claude-memory` or custom |
| macOS | `~/mem0_env` or `~/.venv/mem0` | `~/Library/Application Support/claude-memory` |
| Windows | `%USERPROFILE%\mem0_env` | `%LOCALAPPDATA%\claude-memory` |

## Phase 5: Queue System

### Decision Points
- **File-based queue** → Included, works everywhere, uses fcntl/portalocker
- **Redis** → Better for high volume, requires Redis server
- **SQLite** → Good middle ground, but needs code changes

### File Locking Portability

| Platform | Method |
|----------|--------|
| Linux/macOS | `fcntl.flock()` (POSIX) - works as-is |
| Windows | Replace with `portalocker` library or `msvcrt.locking()` |

### Queue Path Conventions

| Platform | Queue Directory |
|----------|-----------------|
| Linux | `/var/lib/aios` (needs sudo) or `~/.local/state/claude-memory` |
| macOS | `~/Library/Application Support/claude-memory/queue` |
| Windows | `%LOCALAPPDATA%\claude-memory\queue` |

**Override via environment:**
```bash
export MEM0_DATA_DIR=/custom/path/to/queue
```

Note: Queue files are small and transient (cleared after processing), so they don't need bulk storage. Fast local storage (SSD) is ideal for queue performance.

## Phase 6: Background Worker

The worker processes the queue asynchronously. It MUST:
- Start on boot
- Restart on failure
- Drain queue before shutdown (prevent data loss)

### Platform-Specific Service Management

| Platform | Service Manager | Config Location |
|----------|-----------------|-----------------|
| Linux (systemd) | systemctl | `/etc/systemd/system/memory-worker.service` |
| Linux (no systemd) | supervisor, runit, or cron | varies |
| macOS | launchd | `~/Library/LaunchAgents/com.claude.memory-worker.plist` |
| Windows | Task Scheduler or NSSM | Registry or XML |

### Service Requirements
1. **ExecStart** - Run `memory_worker.py --daemon`
2. **ExecStop** - Run `drain_queue.py` (wait for queue to empty)
3. **Restart** - On failure, with backoff
4. **StopTimeout** - At least 600 seconds (10 min) for queue drain

### Example: systemd (Linux)
```ini
[Service]
ExecStart=/path/to/venv/bin/python /path/to/hooks/memory_worker.py --daemon
ExecStop=/path/to/venv/bin/python /path/to/hooks/drain_queue.py
TimeoutStopSec=600
Restart=on-failure
```

### Example: launchd (macOS)
```xml
<key>ProgramArguments</key>
<array>
  <string>/path/to/venv/bin/python</string>
  <string>/path/to/hooks/memory_worker.py</string>
  <string>--daemon</string>
</array>
<key>KeepAlive</key>
<true/>
```

### Example: Windows Task Scheduler
- Trigger: At startup
- Action: Start program `pythonw.exe` with args `memory_worker.py --daemon`
- Settings: Restart on failure

## Phase 7: Claude Code Integration

### Hook Configuration (`~/.claude/settings.json`)

Hooks must be configured with correct paths for the platform:

```json
{
  "hooks": {
    "SessionStart": [{"hooks": [{"type": "command", "command": "PULL_HOOK_PATH"}]}],
    "Stop": [{"hooks": [{"type": "command", "command": "PUSH_HOOK_PATH", "timeout": 120}]}],
    "PreCompact": [
      {"matcher": "auto", "hooks": [{"type": "command", "command": "PUSH_HOOK_PATH", "timeout": 120}]},
      {"matcher": "manual", "hooks": [{"type": "command", "command": "PUSH_HOOK_PATH", "timeout": 120}]}
    ]
  }
}
```

### Hook Invocation

| Platform | Method |
|----------|--------|
| Linux/macOS | Shell script wrapper or direct Python |
| Windows | Batch file, PowerShell, or direct Python with `pythonw.exe` |

### MCP Server Configuration (`~/.claude.json`)

```json
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/mcp/memory_mcp_server.py"]
    }
  }
}
```

## Phase 8: Verification Checklist

Before considering deployment complete:

- [ ] Qdrant responds to health check
- [ ] Ollama runs extraction model with GPU (if available)
- [ ] Python environment has all dependencies
- [ ] Queue directory exists with correct permissions
- [ ] Worker service starts and stays running
- [ ] Manual `remember()` via MCP works
- [ ] Manual `recall()` via MCP works
- [ ] Stop hook extracts and queues memories
- [ ] Worker processes queued memories
- [ ] SessionStart hook recalls memories
- [ ] Service survives reboot
- [ ] Shutdown waits for queue drain

## Adaptation Notes for Deploying Claude

When deploying this system:

1. **Don't assume paths** - Detect home directory, check for existing installations
2. **Don't assume package managers** - Check what's available (apt, brew, choco, etc.)
3. **Don't assume GPU** - Detect and adapt, CPU-only is valid
4. **Don't assume Docker** - It's convenient but not universal
5. **Test each phase** - Verify before moving to next component
6. **Document choices** - Record what was installed where for future reference
7. **Handle errors gracefully** - If GPU setup fails, fall back to CPU
8. **Respect existing config** - Merge with existing settings.json, don't overwrite

## Troubleshooting Decision Tree

```
Memory not being stored?
├── Check queue: memory_worker.py --status
│   ├── Queue empty? → Hook not triggering, check settings.json
│   └── Queue has items? → Worker not processing
│       ├── Worker running? → Check service status
│       └── Worker stuck? → Check /tmp/memory_worker.log
│           ├── Qdrant error? → Check Qdrant health
│           └── Ollama error? → Check Ollama health

Memory not being recalled?
├── MCP server running? → Check Claude Code MCP status
├── Memories exist? → list_memories via MCP
└── Query returning empty? → Check embedding model loaded
```

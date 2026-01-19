# Installation Guide

## Part 1: Install Claude Code

### Linux / macOS
```bash
# Install via npm (recommended)
npm install -g @anthropic-ai/claude-code

# Or via curl
curl -fsSL https://claude.ai/install-cli | sh
```

### Windows
```powershell
# Install via npm
npm install -g @anthropic-ai/claude-code

# Or download installer from claude.ai/download
```

### First Run
```bash
# Launch Claude Code
claude

# You'll be prompted to authenticate via browser
# This links to your Anthropic account
```

### Verify Installation
```bash
claude --version
claude /help
```

---

## Part 2: Install Memory Plugin

### Prerequisites
Before installing the memory plugin, you need:
- **Qdrant** - Vector database (Docker or native)
- **Ollama** - Local LLM runtime
- **Python 3.10+** - For the plugin scripts

### Quick Install (Let Claude Do It)

Start Claude Code and say:
```
Read the setup skill at github.com/tyrigi/claude_plugins/blob/main/claude-memory-plugin/skills/setup.md
and deploy the memory plugin on this system.
```

Claude will:
1. Detect your OS, GPU, and existing services
2. Install missing dependencies
3. Configure hooks and MCP server
4. Set up background worker service
5. Verify the installation

### Manual Install

#### 1. Clone the plugin
```bash
git clone https://github.com/tyrigi/claude_plugins.git
cd claude_plugins/claude-memory-plugin
```

#### 2. Install dependencies

**Qdrant** (pick one):
```bash
# Docker
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

# Or native - see https://qdrant.tech/documentation/quick-start/
```

**Ollama**:
```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull qwen3:4b-instruct
ollama pull bge-m3
```

**Python environment**:
```bash
python3 -m venv ~/mem0_env
source ~/mem0_env/bin/activate  # or `mem0_env\Scripts\activate` on Windows
pip install -r requirements.txt
```

#### 3. Configure Claude Code hooks

Edit `~/.claude/settings.json`:
```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/claude-memory-plugin/hooks/pull_hook.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/claude-memory-plugin/hooks/push_hook.sh",
            "timeout": 120
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "matcher": "auto",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/claude-memory-plugin/hooks/push_hook.sh",
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```

#### 4. Configure MCP server

Edit `~/.claude.json`, add to `mcpServers`:
```json
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "/path/to/mem0_env/bin/python",
      "args": ["/path/to/claude-memory-plugin/mcp/memory_mcp_server.py"]
    }
  }
}
```

#### 5. Set up background worker

**Linux (systemd)**:
```bash
# Edit paths in systemd/memory-worker.service, then:
sudo cp systemd/memory-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now memory-worker
```

**macOS (launchd)**:
```bash
# Create ~/Library/LaunchAgents/com.claude.memory-worker.plist
# See skills/setup.md for template
launchctl load ~/Library/LaunchAgents/com.claude.memory-worker.plist
```

**Windows**:
```powershell
# Use Task Scheduler to run memory_worker.py --daemon at startup
# See skills/setup.md for details
```

#### 6. Create queue directory
```bash
# Linux
sudo mkdir -p /var/lib/aios && sudo chown $USER:$USER /var/lib/aios

# macOS
mkdir -p ~/Library/Application\ Support/claude-memory/queue

# Windows
mkdir %LOCALAPPDATA%\claude-memory\queue
```

#### 7. Verify installation
```bash
# Check Qdrant
curl http://localhost:6333/collections

# Check Ollama
ollama run qwen3:4b-instruct "test"

# Check worker
python hooks/memory_worker.py --status

# Test MCP (in Claude Code)
# Use /mcp to see if "memory" server is connected
```

---

## Troubleshooting

**Hooks not triggering**: Check `~/.claude/settings.json` syntax and paths

**MCP server not connecting**: Check `~/.claude.json` syntax and Python path

**Worker not processing**: Check `systemctl status memory-worker` or logs at `/tmp/memory_worker.log`

**GPU not detected by Ollama**: See GPU acceleration section in `skills/setup.md`

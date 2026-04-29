# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniClaw is a multi-channel AI assistant framework, heavily forked toward Feishu (Lark) integration. It connects messaging platforms (Feishu, Telegram, Discord, WhatsApp) to LLMs via LiteLLM, with a persistent memory system, session compression, cron scheduling, and subagent support.

The repo has two runtimes:
- **`miniclaw/`** — Python assistant framework (channels, agent loop, tools, memory, session)
- **`bridge/`** — Node.js/TypeScript WhatsApp bridge (communicates with Python over WebSocket)

## Commands

### Python

```bash
# Install (editable)
pip install -e .

# Run the gateway (all channels + agent loop)
miniclaw gateway

# Run in CLI mode (single interactive session)
miniclaw agent

# Lint
ruff check .

# Tests
pytest

# Run a single test file
pytest tests/test_foo.py
```

### WhatsApp Bridge (Node)

```bash
cd bridge
npm install
npm run build   # tsc
npm start       # node dist/index.js
npm run dev     # build + start
```

### Docker

```bash
docker build -t miniclaw .
# Default entrypoint: miniclaw; default command: status
```

## Architecture

### Message Flow

```
Platform → Channel → InboundMessage → MessageBus → AgentLoop
                                                        ↓
                                               ContextBuilder
                                                  (bootstrap files + skills + memory + history)
                                                        ↓
                                               LiteLLMProvider
                                                        ↓
                                               ToolRegistry → tool execution
                                                        ↓
                                               OutboundMessage → ChannelManager → Platform
```

### Key Layers

| Layer | Location | Role |
|---|---|---|
| Channels | `miniclaw/channels/` | Platform adapters (Feishu, Telegram, Discord, WhatsApp) |
| Message Bus | `miniclaw/bus/` | Decouples channels from the agent |
| Agent Loop | `miniclaw/agent/loop.py` | Orchestrates context, LLM calls, tools, streaming |
| Context Builder | `miniclaw/agent/context.py` | Assembles system prompt from workspace files + skills + memory |
| Tools | `miniclaw/agent/tools/` | Registered tool implementations |
| Session | `miniclaw/session/` | JSONL-based conversation persistence + rolling compression |
| Memory | `miniclaw/agent/memory*.py` | Long-term personal memory extraction and retrieval |
| Cron | `miniclaw/cron/service.py` | Scheduled jobs that call `agent.process_direct()` |
| Heartbeat | `miniclaw/heartbeat/service.py` | 30-minute check of `HEARTBEAT.md` |
| Config | `miniclaw/config/` | Pydantic Settings schema; env prefix `miniclaw_`, delimiter `__` |
| CLI | `miniclaw/cli/commands.py` | Typer app; all `miniclaw` subcommands |

### Configuration

- Config file: `$miniclaw_HOME/config.json` (defaults to `~/.miniclaw/config.json`)
- Schema: `miniclaw/config/schema.py` — all defaults live here
- Default model: `anthropic/claude-opus-4-5`; default gateway port: `18790`
- WhatsApp bridge URL: `ws://localhost:3001` (bridge listens on port 3001 by default)
- Env vars override config using `miniclaw_` prefix and `__` nesting, e.g. `miniclaw_agents__defaults__model`

### Context Assembly (`miniclaw/agent/context.py`)

The system prompt is built each turn by loading workspace bootstrap files in order: `AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md`, `IDENTITY.md`. Skills are loaded from `miniclaw/skills/` (builtins) and `workspace/skills/` (overrides). Always-on skills are injected in full; others appear as summaries until the model reads their `SKILL.md`.

### Session Storage (`miniclaw/session/manager.py`)

Sessions are JSONL files. Line 0 is metadata; subsequent lines are messages. Active sessions are indexed in `_active.json`. The compressor creates a `.summary.json` per session with a rolling LLM-generated summary when compression is enabled.

### Feishu Streaming

`AgentLoop` emits `feishu_stream` metadata during generation. `FeishuChannel` interprets this to drive CardKit streaming updates (init / append / tool_update / finalize). Local markdown images are uploaded and replaced with Feishu image keys before sending.

### Subagents

`SpawnTool` delegates to `SubagentManager`, which runs a child agent with its own tool registry. Results return as `system` messages on the main bus and are summarized by the parent agent.

### Procedural Memory

After each task completes, `AgentLoop` fires a background `asyncio.create_task` that asks the LLM whether the execution process is worth saving as a reusable procedure (only non-trivial, repeatable workflows qualify). If yes, the procedure is stored in a SQLite database with FTS5 full-text search. On the next task, the agent searches the DB for relevant past procedures and injects them into the system prompt under `# Procedural Memory`.

When a similar procedure already exists, a second LLM call merges old and new into the best combined version. Usage counts are incremented each time a procedure is retrieved.

Key files:
- `miniclaw/procedural_memory/store.py` — SQLite + FTS5 CRUD
- `miniclaw/procedural_memory/manager.py` — evaluate / retrieve / merge logic
- Config: `tools.procedural_memory` (`enabled`, `db_path`, `llm_model`, `top_k`, `min_similarity_score`)

## Project Layout Highlights

```
miniclaw/
  cli/commands.py       # All CLI subcommands (gateway, agent, memory-*, cron-*, channels-*)
  agent/loop.py         # Core agent loop
  agent/context.py      # System prompt assembly
  config/schema.py      # All config defaults and structure
  config/loader.py      # Config file load/save with key migration
  session/manager.py    # JSONL session persistence
  session/compressor.py # Rolling session compression
  providers/litellm_provider.py  # LLM calls via LiteLLM
bridge/
  src/index.ts          # Bridge entry; port 3001
  src/server.ts         # WebSocket server forwarding Python ↔ WhatsApp
  src/whatsapp.ts       # Baileys wrapper
workspace/
  AGENTS.md             # Agent behavior instructions (loaded into every context)
  TOOLS.md              # Tool reference for the agent
```

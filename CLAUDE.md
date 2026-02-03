# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Build
cargo build              # Debug build
cargo build --release    # Release build (~7MB binary)

# Run
cargo run -- <subcommand>   # Run with arguments
cargo run -- chat           # Interactive chat
cargo run -- ask "question" # Single question
cargo run -- daemon start   # Start daemon with HTTP server

# Test
cargo test                  # Run all tests
cargo test <test_name>      # Run specific test
cargo test -- --nocapture   # Show test output

# Lint
cargo clippy
cargo fmt --check
```

## Architecture

LocalGPT is a local-only AI assistant with persistent markdown-based memory and optional autonomous operation via heartbeat.

### Core Modules (`src/`)

- **agent/** - LLM interaction layer
  - `providers.rs` - Trait `LLMProvider` with implementations for OpenAI, Anthropic, Ollama, and Claude CLI. Model prefix determines provider (`claude-cli/*` → Claude CLI, `gpt-*` → OpenAI, `claude-*` → Anthropic API, else Ollama)
  - `session.rs` - Conversation state with automatic compaction when approaching context window limits
  - `session_store.rs` - Session metadata store (`sessions.json`) with CLI session ID persistence
  - `system_prompt.rs` - Builds system prompt with identity, safety, workspace info, tools, skills, and special tokens
  - `skills.rs` - Loads SKILL.md files from workspace/skills/ for specialized task handling
  - `tools.rs` - Agent tools: `bash`, `read_file`, `write_file`, `edit_file`, `memory_search`, `memory_append`, `web_fetch`

- **memory/** - Markdown-based knowledge store
  - `index.rs` - SQLite FTS5 index for fast search. Chunks files (~400 tokens with 80 token overlap)
  - `watcher.rs` - File system watcher for automatic reindexing
  - Files: `MEMORY.md` (curated knowledge), `HEARTBEAT.md` (pending tasks), `memory/YYYY-MM-DD.md` (daily logs)

- **heartbeat/** - Autonomous task runner
  - `runner.rs` - Runs on configurable interval within active hours. Reads `HEARTBEAT.md` and executes pending tasks

- **server/** - HTTP/WebSocket API
  - `http.rs` - Axum-based REST API. Note: creates new Agent per request (no session persistence via HTTP)
  - Endpoints: `/health`, `/api/status`, `/api/chat`, `/api/memory/search`, `/api/memory/stats`

- **config/** - TOML configuration at `~/.localgpt/config.toml`
  - Supports `${ENV_VAR}` expansion in API keys
  - `workspace_path()` returns expanded memory workspace path
  - `migrate.rs` - Auto-migrates from OpenClaw's `~/.openclaw/config.json5` if LocalGPT config doesn't exist

- **cli/** - Clap-based subcommands: `chat`, `ask`, `daemon`, `memory`, `config`

### Key Patterns

- Agent is not `Send+Sync` due to SQLite connections - HTTP handler uses `spawn_blocking`
- Session compaction triggers memory flush (prompts LLM to save important context before truncating)
- Memory context automatically loaded into new sessions: `MEMORY.md`, recent daily logs, `HEARTBEAT.md`
- Tools use `shellexpand::tilde()` for path expansion

## Configuration

Default config path: `~/.localgpt/config.toml` (see `config.example.toml`)

Key settings:
- `agent.default_model` - Model name (determines provider). Default: `claude-cli/opus`
- `agent.context_window` / `reserve_tokens` - Context management
- `heartbeat.interval` - Duration string (e.g., "30m", "1h")
- `heartbeat.active_hours` - Optional `{start, end}` in "HH:MM" format
- `server.port` - HTTP server port (default: 18790)

## Skills System

Skills are SKILL.md files in `workspace/skills/<skill-name>/SKILL.md` that provide specialized instructions.

```
~/.localgpt/workspace/
└── skills/
    ├── code-review/
    │   └── SKILL.md
    └── commit-message/
        └── SKILL.md
```

When a skill applies, the agent reads the SKILL.md and follows its instructions. The system prompt includes an `<available_skills>` list with skill names, descriptions, and paths.

## CLI Commands (Interactive Chat)

In the `chat` command, these slash commands are available:

- `/help` - Show available commands
- `/quit`, `/exit`, `/q` - Exit chat
- `/new` - Start fresh session (reloads system prompt and memory context)
- `/compact` - Compact session history (summarize and truncate)
- `/clear` - Clear session history (keeps current context)
- `/memory <query>` - Search memory files
- `/save` - Save current session to disk
- `/status` - Show session info (ID, messages, tokens, compactions)

## OpenClaw Compatibility

LocalGPT uses a file structure compatible with OpenClaw for easy migration.

### Directory Structure (matches OpenClaw)

```
~/.localgpt/                          # State directory (OpenClaw: ~/.openclaw/)
├── config.toml                       # Config (OpenClaw uses JSON5)
├── agents/
│   └── main/                         # Default agent ID
│       └── sessions/
│           ├── sessions.json         # Session metadata + CLI session IDs
│           └── <sessionId>.jsonl     # Session transcripts
├── workspace/                        # Memory workspace
│   ├── MEMORY.md                     # Long-term memory
│   ├── HEARTBEAT.md                  # Pending tasks
│   ├── SOUL.md                       # Persona/tone (optional)
│   └── memory/
│       └── YYYY-MM-DD.md             # Daily logs
└── logs/
```

### Migrating from OpenClaw

Best-effort migration from OpenClaw:

```bash
# Copy OpenClaw data to LocalGPT
cp -r ~/.openclaw/agents ~/.localgpt/agents
cp -r ~/.openclaw/workspace ~/.localgpt/workspace

# sessions.json format is compatible
# CLI session IDs (cliSessionIds, claudeCliSessionId) are preserved
```

**What works:**
- `sessions.json` session store (same format)
- CLI session ID persistence (`cliSessionIds` map)
- Workspace files: `MEMORY.md`, `HEARTBEAT.md`, `SOUL.md`, `memory/*.md`
- Session transcripts (`<sessionId>.jsonl`)

**What differs:**
- Config format: LocalGPT uses TOML, OpenClaw uses JSON5
- No multi-channel routing (LocalGPT is local-only)
- No bindings/agents.list (LocalGPT uses single "main" agent)
- No subagent spawning (yet)

**Auto-config migration:**
If `~/.localgpt/config.toml` doesn't exist but `~/.openclaw/config.json5` does, LocalGPT will auto-migrate:
- `agents.defaults.workspace` → `memory.workspace`
- `agents.defaults.model` → `agent.default_model`
- `models.openai.apiKey` → `providers.openai.api_key`
- `models.anthropic.apiKey` → `providers.anthropic.api_key`

### sessions.json Format

```json
{
  "main": {
    "sessionId": "uuid-here",
    "updatedAt": 1234567890,
    "cliSessionIds": {
      "claude-cli": "cli-session-uuid"
    },
    "claudeCliSessionId": "cli-session-uuid",
    "compactionCount": 0
  }
}
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Build
cargo build                           # Debug build
cargo build --release                 # Release build (~27MB binary)
cargo build --no-default-features     # Headless (no desktop GUI)

# Run
cargo run -- chat                     # Interactive chat
cargo run -- ask "question"           # Single question
cargo run -- daemon start             # Start daemon with HTTP server

# Test
cargo test                            # Run all tests
cargo test <test_name>                # Run specific test
cargo test -- --nocapture             # Show test output

# Lint
cargo clippy
cargo fmt --check
```

## Architecture

LocalGPT is a local-only AI assistant with persistent markdown-based memory and optional autonomous operation via heartbeat. Single binary, no runtime dependencies.

### Module Overview (`src/`)

- **agent/** - LLM interaction layer (providers, session management, tools, skills, system prompt)
- **memory/** - Markdown-based knowledge store with SQLite FTS5 index and local embeddings
- **heartbeat/** - Autonomous task runner (reads `HEARTBEAT.md`, runs on configurable interval within active hours)
- **server/** - Axum-based HTTP/WebSocket API with embedded web UI (`ui/` directory via `rust-embed`)
- **desktop/** - Optional eframe/egui native GUI (behind `desktop` feature flag)
- **config/** - TOML config at `~/.localgpt/config.toml` with `${ENV_VAR}` expansion and OpenClaw auto-migration
- **cli/** - Clap-based subcommands: `chat`, `ask`, `daemon`, `memory`, `config`, `desktop`
- **concurrency/** - `TurnGate` (serializes agent turns) and `WorkspaceLock` (file-based workspace locking)

### Key Architectural Patterns

**Provider routing by model prefix** (`agent/providers.rs`): The model name string determines which LLM provider is used: `claude-cli/*` → Claude CLI subprocess, `gpt-*`/`o1-*`/`o3-*` → OpenAI API, `claude-*` → Anthropic API, anything else → Ollama.

**Agent is not `Send+Sync`**: Due to SQLite connections (rusqlite). The HTTP handler uses `spawn_blocking` and wraps Agent in `tokio::sync::Mutex`. The desktop GUI runs the agent in a background thread communicating via channels.

**Session compaction flow**: When approaching context window limits, a two-stage process runs: (1) memory flush — prompts the LLM to save important context to workspace markdown files, (2) session compaction — summarizes and truncates conversation history. The soft threshold (`MEMORY_FLUSH_SOFT_THRESHOLD = 4000` tokens) triggers flush before the hard compaction limit.

**Tool call loop**: `handle_response` is recursive — if the LLM returns tool calls, they're executed and the results fed back, repeating until the LLM returns text. `stream_with_tool_loop` caps at 10 iterations.

**Memory context loading order** on new session (`build_memory_context`): IDENTITY.md → USER.md → SOUL.md → AGENTS.md → TOOLS.md → MEMORY.md → recent daily logs (2 days) → HEARTBEAT.md.

**Content sanitization** (`agent/sanitize.rs`): Tool outputs and memory content can be wrapped in special delimiter tokens to prevent prompt injection. Controlled by `tools.use_content_delimiters` config.

### Feature Flags

- `desktop` (default) — enables eframe/egui desktop GUI. Disable with `--no-default-features` for headless/server/Docker builds.
- `gguf` — GGUF embedding model support via llama.cpp (requires C++ compiler). Used in `memory/embeddings.rs`.

### Configuration

Default config: `~/.localgpt/config.toml` (see `config.example.toml`). Auto-created on first run. Auto-migrates from OpenClaw's `~/.openclaw/config.json5` if present.

Workspace path resolution: `LOCALGPT_WORKSPACE` env → `LOCALGPT_PROFILE` env (`~/.localgpt/workspace-{profile}`) → `memory.workspace` config → `~/.localgpt/workspace`.

Key settings: `agent.default_model` (default: `claude-cli/opus`), `agent.context_window`/`reserve_tokens`, `memory.workspace`, `heartbeat.interval`/`active_hours`, `server.port` (default: 31327).

### OpenClaw Compatibility

The workspace file format (MEMORY.md, SOUL.md, HEARTBEAT.md, etc.), session JSONL format (Pi-compatible), session metadata store (`sessions.json`), skills system (SKILL.md with YAML frontmatter), and memory search parameters (400-token chunks, 80-token overlap, 0.7/0.3 vector/BM25 hybrid weighting) are all compatible with OpenClaw. See the workspace directory structure in README.md.

## Commands

### Task Tracking
- `bd` is deprecated. ALWAYS use `br` for task tracking.

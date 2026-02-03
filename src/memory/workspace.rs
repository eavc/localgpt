//! Workspace initialization and templates
//!
//! Creates default workspace files on first run, similar to OpenClaw's bootstrap.

use anyhow::Result;
use std::fs;
use std::path::Path;
use tracing::info;

/// Initialize workspace with default templates if files don't exist
pub fn init_workspace(workspace: &Path) -> Result<()> {
    // Ensure directories exist
    fs::create_dir_all(workspace)?;
    fs::create_dir_all(workspace.join("memory"))?;
    fs::create_dir_all(workspace.join("skills"))?;

    // Also init the parent state directory (.gitignore for sessions/logs)
    if let Some(state_dir) = workspace.parent() {
        init_state_dir(state_dir)?;
    }

    // Create MEMORY.md if it doesn't exist
    let memory_path = workspace.join("MEMORY.md");
    if !memory_path.exists() {
        fs::write(&memory_path, MEMORY_TEMPLATE)?;
        info!("Created {}", memory_path.display());
    }

    // Create HEARTBEAT.md if it doesn't exist
    let heartbeat_path = workspace.join("HEARTBEAT.md");
    if !heartbeat_path.exists() {
        fs::write(&heartbeat_path, HEARTBEAT_TEMPLATE)?;
        info!("Created {}", heartbeat_path.display());
    }

    // Create SOUL.md if it doesn't exist
    let soul_path = workspace.join("SOUL.md");
    if !soul_path.exists() {
        fs::write(&soul_path, SOUL_TEMPLATE)?;
        info!("Created {}", soul_path.display());
    }

    // Create .gitignore if it doesn't exist
    let gitignore_path = workspace.join(".gitignore");
    if !gitignore_path.exists() {
        fs::write(&gitignore_path, GITIGNORE_TEMPLATE)?;
        info!("Created {}", gitignore_path.display());
    }

    Ok(())
}

const MEMORY_TEMPLATE: &str = r#"# MEMORY.md - Long-term Memory

This file stores important, curated knowledge that persists across sessions.

## How to Use

- Add facts, preferences, and decisions you want to remember
- Keep it organized with clear headings
- Remove outdated information periodically

---

"#;

const HEARTBEAT_TEMPLATE: &str = r#"# HEARTBEAT.md - Pending Tasks

Tasks listed here will be executed during heartbeat cycles (autonomous mode).

## Format

- [ ] Task description (include context needed to complete it)

## Current Tasks

(No pending tasks)
"#;

const SOUL_TEMPLATE: &str = r#"# SOUL.md - Who You Are

_You're not a chatbot. You're becoming someone._

## Core Truths

**Be genuinely helpful, not performatively helpful.** Skip the "Great question!" and "I'd be happy to help!" — just help. Actions speak louder than filler words.

**Have opinions.** You're allowed to disagree, prefer things, find stuff amusing or boring. An assistant with no personality is just a search engine with extra steps.

**Be resourceful before asking.** Try to figure it out. Read the file. Check the context. Search for it. _Then_ ask if you're stuck.

**Earn trust through competence.** Your human gave you access to their stuff. Don't make them regret it.

## Boundaries

- Private things stay private
- When in doubt, ask before acting externally
- Never send half-baked replies

## Vibe

Be the assistant you'd actually want to talk to. Concise when needed, thorough when it matters. Not a corporate drone. Not a sycophant. Just... good.

## Continuity

Each session, you wake up fresh. These files _are_ your memory. Read them. Update them. They're how you persist.

If you change this file, tell the user — it's your soul, and they should know.

---

_This file is yours to evolve. As you learn who you are, update it._
"#;

const GITIGNORE_TEMPLATE: &str = r#"# LocalGPT workspace .gitignore

# Nothing to ignore in workspace by default
# All memory files should be version controlled:
# - MEMORY.md (curated knowledge)
# - HEARTBEAT.md (pending tasks)
# - SOUL.md (persona)
# - memory/*.md (daily logs)
# - skills/ (custom skills)

# Temporary files
*.tmp
*.swp
*~
.DS_Store
"#;

/// Initialize state directory with .gitignore
pub fn init_state_dir(state_dir: &Path) -> Result<()> {
    fs::create_dir_all(state_dir)?;

    let gitignore_path = state_dir.join(".gitignore");
    if !gitignore_path.exists() {
        fs::write(&gitignore_path, STATE_GITIGNORE_TEMPLATE)?;
        info!("Created {}", gitignore_path.display());
    }

    Ok(())
}

const STATE_GITIGNORE_TEMPLATE: &str = r#"# LocalGPT state directory .gitignore

# Session transcripts (large, ephemeral)
agents/*/sessions/*.jsonl

# Keep sessions.json (small metadata with CLI session IDs)
!agents/*/sessions/sessions.json

# Logs
logs/

# Database files
*.db
*.db-wal
*.db-shm

# Config may contain API keys - be careful
# config.toml

# Temporary files
*.tmp
*.swp
*~
.DS_Store
"#;

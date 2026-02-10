use anyhow::Result;
use clap::{Args, Subcommand};

use localgpt::config::Config;

#[derive(Args)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub command: ConfigCommands,
}

#[derive(Subcommand)]
pub enum ConfigCommands {
    /// Show current configuration
    Show {
        /// Output format: toml (default) or json
        #[arg(short, long, default_value = "toml")]
        format: String,
    },

    /// Get a configuration value
    Get {
        /// Config key (e.g., agent.default_model)
        key: String,
    },

    /// Set a configuration value
    Set {
        /// Config key (e.g., agent.default_model)
        key: String,

        /// Value to set
        value: String,
    },

    /// Show config file path
    Path,

    /// Initialize default config file
    Init {
        /// Overwrite existing config
        #[arg(short, long)]
        force: bool,
    },
}

pub async fn run(args: ConfigArgs) -> Result<()> {
    match args.command {
        ConfigCommands::Show { format } => show_config(&format),
        ConfigCommands::Get { key } => get_config(&key),
        ConfigCommands::Set { key, value } => set_config(&key, &value),
        ConfigCommands::Path => show_path(),
        ConfigCommands::Init { force } => init_config(force),
    }
}

fn show_config(format: &str) -> Result<()> {
    let config = Config::load()?;

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&config)?;
            println!("{}", json);
        }
        _ => {
            let toml = toml::to_string_pretty(&config)?;
            println!("{}", toml);
        }
    }

    Ok(())
}

fn get_config(key: &str) -> Result<()> {
    let config = Config::load()?;
    let value = config.get_value(key)?;
    println!("{}", value);
    Ok(())
}

fn set_config(key: &str, value: &str) -> Result<()> {
    let mut config = Config::load()?;
    config.set_value(key, value)?;
    config.save()?;
    println!("Set {} = {}", key, value);
    Ok(())
}

fn show_path() -> Result<()> {
    let path = Config::config_path()?;
    println!("{}", path.display());
    Ok(())
}

fn init_config(force: bool) -> Result<()> {
    let path = Config::config_path()?;

    if path.exists() && !force {
        anyhow::bail!(
            "Config file already exists at {}. Use --force to overwrite.",
            path.display()
        );
    }

    // Create parent directories
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Write template config with comments
    std::fs::write(&path, DEFAULT_CONFIG_TEMPLATE)?;

    println!("Created config file at {}", path.display());
    Ok(())
}

const DEFAULT_CONFIG_TEMPLATE: &str = r#"# LocalGPT Configuration

[agent]
# Default model: claude-cli/opus, anthropic/claude-sonnet-4-5, openai/gpt-4o, etc.
default_model = "claude-cli/opus"
context_window = 128000
reserve_tokens = 8000

# Anthropic API (for anthropic/* models)
# [providers.anthropic]
# api_key = "${ANTHROPIC_API_KEY}"

# OpenAI API (for openai/* models)
# [providers.openai]
# api_key = "${OPENAI_API_KEY}"

# Claude CLI (for claude-cli/* models, requires claude CLI installed)
[providers.claude_cli]
command = "claude"

[heartbeat]
enabled = true
interval = "30m"

# Only run during these hours (optional)
# [heartbeat.active_hours]
# start = "09:00"
# end = "22:00"

[memory]
workspace = "~/.localgpt/workspace"

[server]
enabled = true
port = 31327
bind = "127.0.0.1"
# rate_limit_per_minute = 120
# api_key = ""

[session]
# Days to keep session transcripts (0 = keep forever, no auto-deletion).
# Sessions older than this are automatically purged on startup.
# retention_days = 0
#
# SECURITY NOTE: Session transcripts are stored as plaintext JSONL files
# at ~/.localgpt/agents/<id>/sessions/. They contain full conversation
# history including user prompts, assistant responses, and tool call
# arguments. Sensitive information discussed in conversations persists
# on disk until sessions are deleted or expire. Consider setting a
# retention period if your conversations may contain credentials,
# personal data, or other sensitive material.

[logging]
level = "info"
"#;

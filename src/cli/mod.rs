pub mod ask;
pub mod chat;
pub mod config;
pub mod daemon;
#[cfg(feature = "desktop")]
pub mod desktop;
pub mod memory;
pub mod sandbox;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "localgpt")]
#[command(author, version, about = "A lightweight, local-only AI assistant")]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Path to config file
    #[arg(short, long, global = true, env = "LOCALGPT_CONFIG")]
    pub config: Option<String>,

    /// Agent ID to use (default: "main", OpenClaw-compatible)
    #[arg(
        short,
        long,
        global = true,
        default_value = "main",
        env = "LOCALGPT_AGENT"
    )]
    pub agent: String,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start an interactive chat session
    Chat(chat::ChatArgs),

    /// Ask a single question
    Ask(ask::AskArgs),

    /// Launch the desktop GUI
    #[cfg(feature = "desktop")]
    Desktop(desktop::DesktopArgs),

    /// Manage the daemon
    Daemon(daemon::DaemonArgs),

    /// Memory operations
    Memory(memory::MemoryArgs),

    /// Configuration management
    Config(config::ConfigArgs),

    /// Shell sandbox management
    Sandbox(sandbox::SandboxArgs),
}

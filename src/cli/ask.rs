use anyhow::Result;
use clap::Args;

use localgpt::agent::{purge_expired_sessions, Agent, AgentConfig};
use localgpt::concurrency::WorkspaceLock;
use localgpt::config::Config;
use localgpt::memory::MemoryManager;

#[derive(Args)]
pub struct AskArgs {
    /// The question or task to perform
    pub question: String,

    /// Model to use (overrides config)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Output format: text (default) or json
    #[arg(short, long, default_value = "text")]
    pub format: String,
}

pub async fn run(args: AskArgs, agent_id: &str) -> Result<()> {
    let config = Config::load()?;

    // Purge expired sessions on startup (no-op if retention_days is 0)
    if let Err(e) = purge_expired_sessions(agent_id, config.session.retention_days) {
        eprintln!("Warning: could not purge expired sessions: {}", e);
    }

    let memory = MemoryManager::new_with_full_config(&config.memory, Some(&config), agent_id)?;

    let agent_config = AgentConfig {
        model: args.model.unwrap_or(config.agent.default_model.clone()),
        context_window: config.agent.context_window,
        reserve_tokens: config.agent.reserve_tokens,
    };

    let mut agent = Agent::new(agent_config, &config, memory).await?;
    agent.new_session().await?;

    let workspace_lock = WorkspaceLock::new()?;
    let _lock_guard = workspace_lock.acquire()?;
    let response = agent.chat(&args.question).await?;

    match args.format.as_str() {
        "json" => {
            let output = serde_json::json!({
                "question": args.question,
                "response": response,
                "model": agent.model(),
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        _ => {
            println!("{}", response);
        }
    }

    Ok(())
}

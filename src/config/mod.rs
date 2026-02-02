mod schema;

pub use schema::*;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub agent: AgentConfig,

    #[serde(default)]
    pub providers: ProvidersConfig,

    #[serde(default)]
    pub heartbeat: HeartbeatConfig,

    #[serde(default)]
    pub memory: MemoryConfig,

    #[serde(default)]
    pub server: ServerConfig,

    #[serde(default)]
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "default_model")]
    pub default_model: String,

    #[serde(default = "default_context_window")]
    pub context_window: usize,

    #[serde(default = "default_reserve_tokens")]
    pub reserve_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvidersConfig {
    #[serde(default)]
    pub openai: Option<OpenAIConfig>,

    #[serde(default)]
    pub anthropic: Option<AnthropicConfig>,

    #[serde(default)]
    pub ollama: Option<OllamaConfig>,

    #[serde(default)]
    pub claude_cli: Option<ClaudeCliConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    pub api_key: String,

    #[serde(default = "default_openai_base_url")]
    pub base_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    pub api_key: String,

    #[serde(default = "default_anthropic_base_url")]
    pub base_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_endpoint")]
    pub endpoint: String,

    #[serde(default = "default_ollama_model")]
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCliConfig {
    #[serde(default = "default_claude_cli_command")]
    pub command: String,

    #[serde(default = "default_claude_cli_model")]
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_interval")]
    pub interval: String,

    #[serde(default)]
    pub active_hours: Option<ActiveHours>,

    #[serde(default)]
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveHours {
    pub start: String,
    pub end: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    #[serde(default = "default_workspace")]
    pub workspace: String,

    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,

    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_bind")]
    pub bind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,

    #[serde(default = "default_log_file")]
    pub file: String,
}

// Default value functions
fn default_model() -> String {
    "claude-cli/opus".to_string()
}
fn default_context_window() -> usize {
    128000
}
fn default_reserve_tokens() -> usize {
    8000
}
fn default_openai_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}
fn default_anthropic_base_url() -> String {
    "https://api.anthropic.com".to_string()
}
fn default_ollama_endpoint() -> String {
    "http://localhost:11434".to_string()
}
fn default_ollama_model() -> String {
    "llama3".to_string()
}
fn default_claude_cli_command() -> String {
    "claude".to_string()
}
fn default_claude_cli_model() -> String {
    "opus".to_string()
}
fn default_true() -> bool {
    true
}
fn default_interval() -> String {
    "30m".to_string()
}
fn default_workspace() -> String {
    "~/.localgpt/workspace".to_string()
}
fn default_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}
fn default_chunk_size() -> usize {
    400
}
fn default_chunk_overlap() -> usize {
    80
}
fn default_port() -> u16 {
    18790
}
fn default_bind() -> String {
    "127.0.0.1".to_string()
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_log_file() -> String {
    "~/.localgpt/logs/agent.log".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            agent: AgentConfig::default(),
            providers: ProvidersConfig::default(),
            heartbeat: HeartbeatConfig::default(),
            memory: MemoryConfig::default(),
            server: ServerConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            default_model: default_model(),
            context_window: default_context_window(),
            reserve_tokens: default_reserve_tokens(),
        }
    }
}

impl Default for ProvidersConfig {
    fn default() -> Self {
        Self {
            openai: None,
            anthropic: None,
            ollama: None,
            claude_cli: None,
        }
    }
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            interval: default_interval(),
            active_hours: None,
            timezone: None,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            workspace: default_workspace(),
            embedding_model: default_embedding_model(),
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            port: default_port(),
            bind: default_bind(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: default_log_file(),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = Self::config_path()?;

        if !path.exists() {
            // Return default config if file doesn't exist
            return Ok(Config::default());
        }

        let content = fs::read_to_string(&path)?;
        let mut config: Config = toml::from_str(&content)?;

        // Expand environment variables in API keys
        config.expand_env_vars();

        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::config_path()?;

        // Create parent directories
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        fs::write(&path, content)?;

        Ok(())
    }

    pub fn config_path() -> Result<PathBuf> {
        let base = directories::BaseDirs::new()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;

        Ok(base.home_dir().join(".localgpt").join("config.toml"))
    }

    fn expand_env_vars(&mut self) {
        if let Some(ref mut openai) = self.providers.openai {
            openai.api_key = expand_env(&openai.api_key);
        }
        if let Some(ref mut anthropic) = self.providers.anthropic {
            anthropic.api_key = expand_env(&anthropic.api_key);
        }
    }

    pub fn get_value(&self, key: &str) -> Result<String> {
        let parts: Vec<&str> = key.split('.').collect();

        match parts.as_slice() {
            ["agent", "default_model"] => Ok(self.agent.default_model.clone()),
            ["agent", "context_window"] => Ok(self.agent.context_window.to_string()),
            ["agent", "reserve_tokens"] => Ok(self.agent.reserve_tokens.to_string()),
            ["heartbeat", "enabled"] => Ok(self.heartbeat.enabled.to_string()),
            ["heartbeat", "interval"] => Ok(self.heartbeat.interval.clone()),
            ["server", "enabled"] => Ok(self.server.enabled.to_string()),
            ["server", "port"] => Ok(self.server.port.to_string()),
            ["server", "bind"] => Ok(self.server.bind.clone()),
            ["memory", "workspace"] => Ok(self.memory.workspace.clone()),
            ["logging", "level"] => Ok(self.logging.level.clone()),
            _ => anyhow::bail!("Unknown config key: {}", key),
        }
    }

    pub fn set_value(&mut self, key: &str, value: &str) -> Result<()> {
        let parts: Vec<&str> = key.split('.').collect();

        match parts.as_slice() {
            ["agent", "default_model"] => self.agent.default_model = value.to_string(),
            ["agent", "context_window"] => self.agent.context_window = value.parse()?,
            ["agent", "reserve_tokens"] => self.agent.reserve_tokens = value.parse()?,
            ["heartbeat", "enabled"] => self.heartbeat.enabled = value.parse()?,
            ["heartbeat", "interval"] => self.heartbeat.interval = value.to_string(),
            ["server", "enabled"] => self.server.enabled = value.parse()?,
            ["server", "port"] => self.server.port = value.parse()?,
            ["server", "bind"] => self.server.bind = value.to_string(),
            ["memory", "workspace"] => self.memory.workspace = value.to_string(),
            ["logging", "level"] => self.logging.level = value.to_string(),
            _ => anyhow::bail!("Unknown config key: {}", key),
        }

        Ok(())
    }

    /// Get workspace path, expanded
    pub fn workspace_path(&self) -> PathBuf {
        let expanded = shellexpand::tilde(&self.memory.workspace);
        PathBuf::from(expanded.to_string())
    }
}

fn expand_env(s: &str) -> String {
    if s.starts_with("${") && s.ends_with('}') {
        let var_name = &s[2..s.len() - 1];
        std::env::var(var_name).unwrap_or_else(|_| s.to_string())
    } else if s.starts_with('$') {
        let var_name = &s[1..];
        std::env::var(var_name).unwrap_or_else(|_| s.to_string())
    } else {
        s.to_string()
    }
}

//! OpenClaw config migration support
//!
//! Provides best-effort migration from OpenClaw's JSON5 config format to LocalGPT's TOML format.

use anyhow::Result;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info, warn};

use super::{AnthropicConfig, ClaudeCliConfig, Config, OpenAIConfig};

/// OpenClaw config structure (partial - only fields we can migrate)
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenClawConfig {
    #[serde(default)]
    agents: Option<OpenClawAgentsConfig>,

    #[serde(default)]
    models: Option<OpenClawModelsConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenClawAgentsConfig {
    #[serde(default)]
    defaults: Option<OpenClawAgentDefaults>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenClawAgentDefaults {
    #[serde(default)]
    workspace: Option<String>,

    #[serde(default)]
    model: Option<String>,

    #[serde(default)]
    context_window: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenClawModelsConfig {
    #[serde(default)]
    openai: Option<OpenClawOpenAIConfig>,

    #[serde(default)]
    anthropic: Option<OpenClawAnthropicConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenClawOpenAIConfig {
    #[serde(default)]
    api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenClawAnthropicConfig {
    #[serde(default)]
    api_key: Option<String>,
}

/// Path to OpenClaw's config file
pub fn openclaw_config_path() -> Result<PathBuf> {
    let base = directories::BaseDirs::new()
        .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;

    Ok(base.home_dir().join(".openclaw").join("config.json5"))
}

/// Try to load and migrate OpenClaw config
pub fn try_migrate_openclaw_config() -> Option<Config> {
    let path = match openclaw_config_path() {
        Ok(p) => p,
        Err(_) => return None,
    };

    if !path.exists() {
        debug!("No OpenClaw config found at {:?}", path);
        return None;
    }

    info!("Found OpenClaw config at {:?}, attempting migration", path);

    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            warn!("Failed to read OpenClaw config: {}", e);
            return None;
        }
    };

    let openclaw_config: OpenClawConfig = match json5::from_str(&content) {
        Ok(c) => c,
        Err(e) => {
            warn!("Failed to parse OpenClaw config: {}", e);
            return None;
        }
    };

    Some(convert_openclaw_config(openclaw_config))
}

/// Convert OpenClaw config to LocalGPT config
fn convert_openclaw_config(oc: OpenClawConfig) -> Config {
    let mut config = Config::default();

    // Migrate agent defaults
    if let Some(agents) = oc.agents {
        if let Some(defaults) = agents.defaults {
            if let Some(workspace) = defaults.workspace {
                // Convert OpenClaw workspace path to LocalGPT format
                let workspace = workspace.replace("~/.openclaw/", "~/.localgpt/");
                config.memory.workspace = workspace;
            }

            if let Some(model) = defaults.model {
                config.agent.default_model = model;
            }

            if let Some(context_window) = defaults.context_window {
                config.agent.context_window = context_window;
            }
        }
    }

    // Migrate model configs (API keys)
    if let Some(models) = oc.models {
        if let Some(openai) = models.openai {
            if let Some(api_key) = openai.api_key {
                config.providers.openai = Some(OpenAIConfig {
                    api_key,
                    base_url: "https://api.openai.com/v1".to_string(),
                });
            }
        }

        if let Some(anthropic) = models.anthropic {
            if let Some(api_key) = anthropic.api_key {
                config.providers.anthropic = Some(AnthropicConfig {
                    api_key,
                    base_url: "https://api.anthropic.com".to_string(),
                });
            }
        }
    }

    // Always enable Claude CLI as default (OpenClaw uses it too)
    if config.providers.claude_cli.is_none() {
        config.providers.claude_cli = Some(ClaudeCliConfig {
            command: "claude".to_string(),
            model: "opus".to_string(),
            skip_permissions: false,
        });
    }

    info!("Migrated OpenClaw config successfully");
    config
}

/// Check if OpenClaw workspace exists and can be reused
pub fn has_openclaw_workspace() -> bool {
    let base = match directories::BaseDirs::new() {
        Some(b) => b,
        None => return false,
    };

    let workspace = base.home_dir().join(".openclaw").join("workspace");
    workspace.exists()
}

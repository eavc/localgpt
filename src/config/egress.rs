//! Centralized data-egress classification for UI and server consumption.
//!
//! Determines whether the current configuration routes data to external
//! (non-local) providers for LLM inference or embedding generation.

use serde::Serialize;

use super::{Config, MemoryConfig};

/// Compact egress status for UI consumption.
#[derive(Debug, Clone, Serialize)]
pub struct EgressStatus {
    /// True when the configured LLM provider sends data externally.
    pub external_llm: bool,
    /// True when the configured embedding provider sends data externally.
    pub external_embeddings: bool,
    /// One-line human-readable summary.
    pub summary: String,
    /// Detail lines explaining each external provider in use.
    pub details: Vec<String>,
}

/// Returns true if the resolved model string routes to an external LLM provider.
///
/// External providers: `anthropic/*`, `openai/*`, and bare `claude-*`/`gpt-*`/`o1-*`
/// model names (which resolve to anthropic/openai respectively).
///
/// Local providers: `claude-cli/*`, `ollama/*`, and anything else that falls
/// through to local routing.
pub fn is_external_llm_model(model: &str, config: &Config) -> bool {
    // Resolve aliases to provider/model format (same logic as create_provider)
    let resolved = resolve_alias(model);

    if let Some(pos) = resolved.find('/') {
        let provider = resolved[..pos].to_lowercase();
        return matches!(provider.as_str(), "anthropic" | "openai");
    }

    // Bare model names without a provider prefix
    let lower = resolved.to_lowercase();
    if lower.starts_with("gpt-") || lower.starts_with("o1") {
        return true;
    }
    if lower.starts_with("claude-") {
        return true;
    }

    // Fallback routing: mirrors create_provider logic
    if config.providers.ollama.is_some() {
        return false; // routes to local ollama
    }
    if config.providers.anthropic.is_some() {
        return true; // routes to anthropic
    }

    false
}

/// Returns true if the configured embedding provider sends data externally.
pub fn is_external_embedding_provider(memory_cfg: &MemoryConfig) -> bool {
    memory_cfg.embedding_provider.eq_ignore_ascii_case("openai")
}

/// Build an [`EgressStatus`] from the current configuration.
pub fn compute_egress_status(config: &Config) -> EgressStatus {
    let external_llm = is_external_llm_model(&config.agent.default_model, config);
    let external_embeddings = is_external_embedding_provider(&config.memory);

    let mut details = Vec::new();

    if external_llm {
        details.push(format!(
            "LLM model '{}' routes to an external API. \
             Use claude-cli/* or ollama/* for local-only inference.",
            config.agent.default_model,
        ));
    }

    if external_embeddings {
        details.push(format!(
            "Embedding provider '{}' sends indexed memory to an external API. \
             Set embedding_provider = \"local\" for local-only embeddings.",
            config.memory.embedding_provider,
        ));
    }

    let summary = match (external_llm, external_embeddings) {
        (true, true) => {
            "External LLM and embedding providers are active — data leaves this device.".to_string()
        }
        (true, false) => "External LLM provider is active — prompts leave this device.".to_string(),
        (false, true) => {
            "External embedding provider is active — memory content leaves this device.".to_string()
        }
        (false, false) => "All providers are local — no data leaves this device.".to_string(),
    };

    EgressStatus {
        external_llm,
        external_embeddings,
        summary,
        details,
    }
}

/// Resolve model aliases (mirrors the logic in `providers::resolve_model_alias`).
fn resolve_alias(model: &str) -> String {
    match model.to_lowercase().as_str() {
        "opus" => "anthropic/claude-opus-4-5".to_string(),
        "sonnet" => "anthropic/claude-sonnet-4-5".to_string(),
        "gpt" => "openai/gpt-4o".to_string(),
        "gpt-mini" => "openai/gpt-4o-mini".to_string(),
        _ => model.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AnthropicConfig, OllamaConfig, OpenAIConfig};

    fn default_config() -> Config {
        Config::default()
    }

    // ── is_external_llm_model ───────────────────────────────────────

    #[test]
    fn test_external_llm_anthropic_prefix() {
        let cfg = default_config();
        assert!(is_external_llm_model("anthropic/claude-opus-4-5", &cfg));
        assert!(is_external_llm_model("anthropic/claude-sonnet-4-5", &cfg));
    }

    #[test]
    fn test_external_llm_openai_prefix() {
        let cfg = default_config();
        assert!(is_external_llm_model("openai/gpt-4o", &cfg));
        assert!(is_external_llm_model("openai/gpt-4o-mini", &cfg));
    }

    #[test]
    fn test_external_llm_mixed_case_anthropic() {
        let cfg = default_config();
        assert!(is_external_llm_model("Anthropic/claude-sonnet-4-5", &cfg));
        assert!(is_external_llm_model("ANTHROPIC/claude-opus-4-5", &cfg));
    }

    #[test]
    fn test_external_llm_mixed_case_openai() {
        let cfg = default_config();
        assert!(is_external_llm_model("OpenAI/gpt-4o", &cfg));
        assert!(is_external_llm_model("OPENAI/gpt-4o-mini", &cfg));
    }

    #[test]
    fn test_local_mixed_case_claude_cli() {
        let cfg = default_config();
        assert!(!is_external_llm_model("Claude-CLI/opus", &cfg));
        assert!(!is_external_llm_model("CLAUDE-CLI/sonnet", &cfg));
    }

    #[test]
    fn test_external_llm_bare_claude_model() {
        let cfg = default_config();
        assert!(is_external_llm_model("claude-sonnet-4-5", &cfg));
    }

    #[test]
    fn test_external_llm_bare_gpt_model() {
        let cfg = default_config();
        assert!(is_external_llm_model("gpt-4o", &cfg));
    }

    #[test]
    fn test_external_llm_bare_o1_model() {
        let cfg = default_config();
        assert!(is_external_llm_model("o1-mini", &cfg));
    }

    #[test]
    fn test_external_llm_alias_opus() {
        let cfg = default_config();
        assert!(is_external_llm_model("opus", &cfg));
    }

    #[test]
    fn test_external_llm_alias_gpt() {
        let cfg = default_config();
        assert!(is_external_llm_model("gpt", &cfg));
    }

    #[test]
    fn test_local_claude_cli() {
        let cfg = default_config();
        assert!(!is_external_llm_model("claude-cli/opus", &cfg));
        assert!(!is_external_llm_model("claude-cli/sonnet", &cfg));
    }

    #[test]
    fn test_local_ollama() {
        let cfg = default_config();
        assert!(!is_external_llm_model("ollama/llama3", &cfg));
        assert!(!is_external_llm_model("ollama/mistral", &cfg));
    }

    #[test]
    fn test_fallback_routes_to_ollama_when_configured() {
        let mut cfg = default_config();
        cfg.providers.ollama = Some(OllamaConfig {
            endpoint: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
        });
        // Unknown bare model → ollama (local)
        assert!(!is_external_llm_model("some-local-model", &cfg));
    }

    #[test]
    fn test_fallback_routes_to_anthropic_when_configured() {
        let mut cfg = default_config();
        cfg.providers.anthropic = Some(AnthropicConfig {
            api_key: "test".to_string(),
            base_url: "https://api.anthropic.com".to_string(),
        });
        // Unknown bare model → anthropic (external)
        assert!(is_external_llm_model("some-unknown-model", &cfg));
    }

    #[test]
    fn test_fallback_no_providers_is_local() {
        let mut cfg = default_config();
        cfg.providers.anthropic = None;
        cfg.providers.openai = None;
        cfg.providers.ollama = None;
        assert!(!is_external_llm_model("some-unknown-model", &cfg));
    }

    // ── is_external_embedding_provider ──────────────────────────────

    #[test]
    fn test_external_embedding_openai() {
        let mut mem = MemoryConfig::default();
        mem.embedding_provider = "openai".to_string();
        assert!(is_external_embedding_provider(&mem));
    }

    #[test]
    fn test_external_embedding_openai_case_insensitive() {
        let mut mem = MemoryConfig::default();
        mem.embedding_provider = "OpenAI".to_string();
        assert!(is_external_embedding_provider(&mem));
    }

    #[test]
    fn test_local_embedding_default() {
        let mem = MemoryConfig::default();
        assert!(!is_external_embedding_provider(&mem));
    }

    #[test]
    fn test_local_embedding_none() {
        let mut mem = MemoryConfig::default();
        mem.embedding_provider = "none".to_string();
        assert!(!is_external_embedding_provider(&mem));
    }

    // ── compute_egress_status ───────────────────────────────────────

    #[test]
    fn test_status_all_local() {
        let cfg = default_config(); // claude-cli/opus + local embeddings
        let status = compute_egress_status(&cfg);
        assert!(!status.external_llm);
        assert!(!status.external_embeddings);
        assert!(status.details.is_empty());
        assert!(status.summary.contains("local"));
    }

    #[test]
    fn test_status_external_llm_only() {
        let mut cfg = default_config();
        cfg.agent.default_model = "anthropic/claude-sonnet-4-5".to_string();
        let status = compute_egress_status(&cfg);
        assert!(status.external_llm);
        assert!(!status.external_embeddings);
        assert_eq!(status.details.len(), 1);
        assert!(status.summary.contains("External LLM"));
    }

    #[test]
    fn test_status_external_embeddings_only() {
        let mut cfg = default_config();
        cfg.memory.embedding_provider = "openai".to_string();
        let status = compute_egress_status(&cfg);
        assert!(!status.external_llm);
        assert!(status.external_embeddings);
        assert_eq!(status.details.len(), 1);
        assert!(status.summary.contains("embedding"));
    }

    #[test]
    fn test_status_both_external() {
        let mut cfg = default_config();
        cfg.agent.default_model = "openai/gpt-4o".to_string();
        cfg.memory.embedding_provider = "openai".to_string();
        cfg.providers.openai = Some(OpenAIConfig {
            api_key: "test".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
        });
        let status = compute_egress_status(&cfg);
        assert!(status.external_llm);
        assert!(status.external_embeddings);
        assert_eq!(status.details.len(), 2);
        assert!(status.summary.contains("LLM and embedding"));
    }
}

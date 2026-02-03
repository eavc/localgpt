use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::pin::Pin;
use std::sync::Mutex as StdMutex;
use tracing::debug;

use crate::config::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub enum LLMResponse {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub delta: String,
    pub done: bool,
}

/// Events emitted during streaming with tools
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Text content chunk
    Content(String),
    /// Tool call started
    ToolCallStart { name: String, id: String },
    /// Tool call completed
    ToolCallEnd { name: String, id: String, output: String },
    /// Stream completed
    Done,
}

pub type StreamResult = Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>;

#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn chat(&self, messages: &[Message], tools: Option<&[ToolSchema]>)
        -> Result<LLMResponse>;

    async fn summarize(&self, text: &str) -> Result<String>;

    /// Stream chat response (default: falls back to non-streaming)
    async fn chat_stream(
        &self,
        messages: &[Message],
        _tools: Option<&[ToolSchema]>,
    ) -> Result<StreamResult> {
        // Default implementation: single chunk with full response
        let resp = self.chat(messages, None).await?;
        let text = match resp {
            LLMResponse::Text(t) => t,
            LLMResponse::ToolCalls(_) => {
                return Err(anyhow::anyhow!("Tool calls not supported in streaming"))
            }
        };
        Ok(Box::pin(futures::stream::once(async move {
            Ok(StreamChunk {
                delta: text,
                done: true,
            })
        })))
    }
}

pub fn create_provider(model: &str, config: &Config) -> Result<Box<dyn LLMProvider>> {
    let workspace = config.workspace_path();

    // Claude CLI: prefix "claude-cli/"
    if model.starts_with("claude-cli/") {
        let model_name = model.strip_prefix("claude-cli/").unwrap_or("opus");
        let cli_config = config.providers.claude_cli.as_ref();
        let command = cli_config.map(|c| c.command.as_str()).unwrap_or("claude");
        return Ok(Box::new(ClaudeCliProvider::new(
            command, model_name, workspace,
        )?));
    }

    // Anthropic shorthand: "anthropic/opus", "anthropic/sonnet", "anthropic/haiku"
    if model.starts_with("anthropic/") {
        let model_name = model.strip_prefix("anthropic/").unwrap_or("sonnet");
        let full_model = normalize_anthropic_model(model_name);
        let anthropic_config = config
            .providers
            .anthropic
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "Anthropic provider not configured. Set ANTHROPIC_API_KEY or add [providers.anthropic] to config.toml"
            ))?;

        return Ok(Box::new(AnthropicProvider::new(
            &anthropic_config.api_key,
            &anthropic_config.base_url,
            &full_model,
        )?));
    }

    // Determine provider from model name
    if model.starts_with("gpt-") || model.starts_with("o1") {
        let openai_config = config
            .providers
            .openai
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "OpenAI provider not configured. Set OPENAI_API_KEY or add [providers.openai] to config.toml"
            ))?;

        Ok(Box::new(OpenAIProvider::new(
            &openai_config.api_key,
            &openai_config.base_url,
            model,
        )?))
    } else if model.starts_with("claude-") {
        // Full Anthropic model name (e.g., "claude-sonnet-4-5-20250514")
        let anthropic_config = config
            .providers
            .anthropic
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "Anthropic provider not configured. Set ANTHROPIC_API_KEY or add [providers.anthropic] to config.toml"
            ))?;

        Ok(Box::new(AnthropicProvider::new(
            &anthropic_config.api_key,
            &anthropic_config.base_url,
            model,
        )?))
    } else if let Some(ollama_config) = &config.providers.ollama {
        Ok(Box::new(OllamaProvider::new(
            &ollama_config.endpoint,
            model,
        )?))
    } else if let Some(cli_config) = &config.providers.claude_cli {
        // Final fallback: try Claude CLI if configured
        Ok(Box::new(ClaudeCliProvider::new(
            &cli_config.command,
            &cli_config.model,
            workspace,
        )?))
    } else {
        anyhow::bail!(
            "Unknown model '{}'. Use one of:\n  \
            - anthropic/opus, anthropic/sonnet, anthropic/haiku\n  \
            - claude-sonnet-4-5-20250514, claude-3-opus-20240229\n  \
            - claude-cli/opus, claude-cli/sonnet\n  \
            - gpt-4o, gpt-4o-mini\n  \
            - Or configure [providers.ollama] for local models",
            model
        )
    }
}

/// Map shorthand Anthropic model names to full model IDs
fn normalize_anthropic_model(shorthand: &str) -> String {
    match shorthand.to_lowercase().as_str() {
        "opus" | "opus-4" => "claude-sonnet-4-5-20250514".to_string(), // Note: Opus 4 not released yet, use Sonnet
        "sonnet" | "sonnet-4" => "claude-sonnet-4-5-20250514".to_string(),
        "haiku" | "haiku-3.5" => "claude-3-5-haiku-20241022".to_string(),
        "sonnet-3.5" => "claude-3-5-sonnet-20241022".to_string(),
        "opus-3" => "claude-3-opus-20240229".to_string(),
        other => other.to_string(), // Pass through full model names
    }
}

// OpenAI Provider
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAIProvider {
    pub fn new(api_key: &str, base_url: &str, model: &str) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
            model: model.to_string(),
        })
    }

    fn format_tools(&self, tools: &[ToolSchema]) -> Vec<Value> {
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                })
            })
            .collect()
    }

    fn format_messages(&self, messages: &[Message]) -> Vec<Value> {
        messages
            .iter()
            .map(|m| {
                let mut msg = json!({
                    "role": match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                    },
                    "content": m.content
                });

                if let Some(ref tool_calls) = m.tool_calls {
                    msg["tool_calls"] = json!(tool_calls
                        .iter()
                        .map(|tc| {
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments
                                }
                            })
                        })
                        .collect::<Vec<_>>());
                }

                if let Some(ref tool_call_id) = m.tool_call_id {
                    msg["tool_call_id"] = json!(tool_call_id);
                }

                msg
            })
            .collect()
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
    ) -> Result<LLMResponse> {
        let mut body = json!({
            "model": self.model,
            "messages": self.format_messages(messages)
        });

        if let Some(tools) = tools {
            if !tools.is_empty() {
                body["tools"] = json!(self.format_tools(tools));
            }
        }

        debug!("OpenAI request: {}", serde_json::to_string_pretty(&body)?);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let response_body: Value = response.json().await?;
        debug!(
            "OpenAI response: {}",
            serde_json::to_string_pretty(&response_body)?
        );

        // Check for errors
        if let Some(error) = response_body.get("error") {
            anyhow::bail!("OpenAI API error: {}", error);
        }

        let choice = response_body["choices"]
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;

        let message = &choice["message"];

        // Check for tool calls
        if let Some(tool_calls) = message.get("tool_calls") {
            if let Some(calls) = tool_calls.as_array() {
                let parsed_calls: Vec<ToolCall> = calls
                    .iter()
                    .map(|tc| ToolCall {
                        id: tc["id"].as_str().unwrap_or("").to_string(),
                        name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                        arguments: tc["function"]["arguments"]
                            .as_str()
                            .unwrap_or("{}")
                            .to_string(),
                    })
                    .collect();

                if !parsed_calls.is_empty() {
                    return Ok(LLMResponse::ToolCalls(parsed_calls));
                }
            }
        }

        let content = message["content"].as_str().unwrap_or("").to_string();

        Ok(LLMResponse::Text(content))
    }

    async fn summarize(&self, text: &str) -> Result<String> {
        let messages = vec![Message {
            role: Role::User,
            content: format!(
                "Summarize the following conversation concisely, preserving key information and context:\n\n{}",
                text
            ),
            tool_calls: None,
            tool_call_id: None,
        }];

        match self.chat(&messages, None).await? {
            LLMResponse::Text(summary) => Ok(summary),
            _ => anyhow::bail!("Unexpected response type"),
        }
    }
}

// Anthropic Provider
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: &str, base_url: &str, model: &str) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
            model: model.to_string(),
        })
    }

    fn format_tools(&self, tools: &[ToolSchema]) -> Vec<Value> {
        tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters
                })
            })
            .collect()
    }

    fn format_messages(&self, messages: &[Message]) -> (Option<String>, Vec<Value>) {
        let mut system_prompt = None;
        let mut formatted = Vec::new();

        for m in messages {
            match m.role {
                Role::System => {
                    system_prompt = Some(m.content.clone());
                }
                Role::User => {
                    formatted.push(json!({
                        "role": "user",
                        "content": m.content
                    }));
                }
                Role::Assistant => {
                    if let Some(ref tool_calls) = m.tool_calls {
                        let tool_use: Vec<Value> = tool_calls.iter().map(|tc| {
                            json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": serde_json::from_str::<Value>(&tc.arguments).unwrap_or(json!({}))
                            })
                        }).collect();
                        formatted.push(json!({
                            "role": "assistant",
                            "content": tool_use
                        }));
                    } else {
                        formatted.push(json!({
                            "role": "assistant",
                            "content": m.content
                        }));
                    }
                }
                Role::Tool => {
                    if let Some(ref tool_call_id) = m.tool_call_id {
                        formatted.push(json!({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": m.content
                            }]
                        }));
                    }
                }
            }
        }

        (system_prompt, formatted)
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
    ) -> Result<LLMResponse> {
        let (system_prompt, formatted_messages) = self.format_messages(messages);

        let mut body = json!({
            "model": self.model,
            "max_tokens": 4096,
            "messages": formatted_messages
        });

        if let Some(system) = system_prompt {
            body["system"] = json!(system);
        }

        if let Some(tools) = tools {
            if !tools.is_empty() {
                body["tools"] = json!(self.format_tools(tools));
            }
        }

        debug!(
            "Anthropic request: {}",
            serde_json::to_string_pretty(&body)?
        );

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let response_body: Value = response.json().await?;
        debug!(
            "Anthropic response: {}",
            serde_json::to_string_pretty(&response_body)?
        );

        // Check for errors
        if let Some(error) = response_body.get("error") {
            anyhow::bail!("Anthropic API error: {}", error);
        }

        let content = response_body["content"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("No content in response"))?;

        // Check for tool use
        let tool_calls: Vec<ToolCall> = content
            .iter()
            .filter(|c| c["type"] == "tool_use")
            .map(|c| ToolCall {
                id: c["id"].as_str().unwrap_or("").to_string(),
                name: c["name"].as_str().unwrap_or("").to_string(),
                arguments: serde_json::to_string(&c["input"]).unwrap_or("{}".to_string()),
            })
            .collect();

        if !tool_calls.is_empty() {
            return Ok(LLMResponse::ToolCalls(tool_calls));
        }

        // Get text content
        let text = content
            .iter()
            .filter(|c| c["type"] == "text")
            .map(|c| c["text"].as_str().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("");

        Ok(LLMResponse::Text(text))
    }

    async fn summarize(&self, text: &str) -> Result<String> {
        let messages = vec![Message {
            role: Role::User,
            content: format!(
                "Summarize the following conversation concisely, preserving key information and context:\n\n{}",
                text
            ),
            tool_calls: None,
            tool_call_id: None,
        }];

        match self.chat(&messages, None).await? {
            LLMResponse::Text(summary) => Ok(summary),
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        _tools: Option<&[ToolSchema]>,
    ) -> Result<StreamResult> {
        let (system_prompt, formatted_messages) = self.format_messages(messages);

        let mut body = json!({
            "model": self.model,
            "max_tokens": 4096,
            "messages": formatted_messages,
            "stream": true
        });

        if let Some(system) = system_prompt {
            body["system"] = json!(system);
        }

        debug!(
            "Anthropic streaming request: {}",
            serde_json::to_string_pretty(&body)?
        );

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        // Check for error status
        if !response.status().is_success() {
            let error_body = response.text().await?;
            anyhow::bail!("Anthropic API error: {}", error_body);
        }

        // Anthropic streams Server-Sent Events (SSE)
        let stream = async_stream::stream! {
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        // Process complete SSE events (lines starting with "data: ")
                        while let Some(pos) = buffer.find("\n\n") {
                            let event = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();

                            // Parse SSE event
                            for line in event.lines() {
                                if let Some(data) = line.strip_prefix("data: ") {
                                    if data == "[DONE]" {
                                        yield Ok(StreamChunk {
                                            delta: String::new(),
                                            done: true,
                                        });
                                        continue;
                                    }

                                    if let Ok(json) = serde_json::from_str::<Value>(data) {
                                        // Handle content_block_delta events
                                        if json["type"] == "content_block_delta" {
                                            if let Some(delta) = json["delta"]["text"].as_str() {
                                                yield Ok(StreamChunk {
                                                    delta: delta.to_string(),
                                                    done: false,
                                                });
                                            }
                                        }
                                        // Handle message_stop event
                                        else if json["type"] == "message_stop" {
                                            yield Ok(StreamChunk {
                                                delta: String::new(),
                                                done: true,
                                            });
                                        }
                                        // Handle errors
                                        else if json["type"] == "error" {
                                            let error_msg = json["error"]["message"]
                                                .as_str()
                                                .unwrap_or("Unknown error");
                                            yield Err(anyhow::anyhow!("Anthropic error: {}", error_msg));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(anyhow::anyhow!("Stream error: {}", e));
                        break;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

// Ollama Provider (for local models)
pub struct OllamaProvider {
    client: Client,
    endpoint: String,
    model: String,
}

impl OllamaProvider {
    pub fn new(endpoint: &str, model: &str) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            endpoint: endpoint.to_string(),
            model: model.to_string(),
        })
    }
}

#[async_trait]
impl LLMProvider for OllamaProvider {
    async fn chat(
        &self,
        messages: &[Message],
        _tools: Option<&[ToolSchema]>,
    ) -> Result<LLMResponse> {
        // Note: Ollama tool support is limited, so we format as plain chat
        let formatted_messages: Vec<Value> = messages
            .iter()
            .map(|m| {
                json!({
                    "role": match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "user", // Treat tool results as user messages
                    },
                    "content": m.content
                })
            })
            .collect();

        let body = json!({
            "model": self.model,
            "messages": formatted_messages,
            "stream": false
        });

        debug!("Ollama request: {}", serde_json::to_string_pretty(&body)?);

        let response = self
            .client
            .post(format!("{}/api/chat", self.endpoint))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let response_body: Value = response.json().await?;
        debug!(
            "Ollama response: {}",
            serde_json::to_string_pretty(&response_body)?
        );

        let content = response_body["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(LLMResponse::Text(content))
    }

    async fn summarize(&self, text: &str) -> Result<String> {
        let messages = vec![Message {
            role: Role::User,
            content: format!(
                "Summarize the following conversation concisely, preserving key information and context:\n\n{}",
                text
            ),
            tool_calls: None,
            tool_call_id: None,
        }];

        match self.chat(&messages, None).await? {
            LLMResponse::Text(summary) => Ok(summary),
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        _tools: Option<&[ToolSchema]>,
    ) -> Result<StreamResult> {
        let formatted_messages: Vec<Value> = messages
            .iter()
            .map(|m| {
                json!({
                    "role": match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "user",
                    },
                    "content": m.content
                })
            })
            .collect();

        let body = json!({
            "model": self.model,
            "messages": formatted_messages,
            "stream": true
        });

        debug!(
            "Ollama streaming request: {}",
            serde_json::to_string_pretty(&body)?
        );

        let response = self
            .client
            .post(format!("{}/api/chat", self.endpoint))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        // Ollama streams newline-delimited JSON
        let stream = async_stream::stream! {
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        // Process complete lines
                        while let Some(pos) = buffer.find('\n') {
                            let line = buffer[..pos].to_string();
                            buffer = buffer[pos + 1..].to_string();

                            if line.is_empty() {
                                continue;
                            }

                            if let Ok(json) = serde_json::from_str::<Value>(&line) {
                                let content = json["message"]["content"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();
                                let done = json["done"].as_bool().unwrap_or(false);

                                yield Ok(StreamChunk {
                                    delta: content,
                                    done,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(anyhow::anyhow!("Stream error: {}", e));
                        break;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Claude CLI Provider - invokes the `claude` CLI command
/// No tool support (text in → text out only)
/// No streaming (CLI output is collected then returned)
pub struct ClaudeCliProvider {
    command: String,
    model: String,
    /// Working directory for CLI execution
    workspace: std::path::PathBuf,
    /// Session key for the session store (e.g., "main")
    session_key: String,
    /// LocalGPT session ID (for session store tracking)
    localgpt_session_id: String,
    /// CLI session ID for multi-turn conversations (interior mutability for &self methods)
    cli_session_id: StdMutex<Option<String>>,
}

/// Provider name for CLI session storage
const CLAUDE_CLI_PROVIDER: &str = "claude-cli";

impl ClaudeCliProvider {
    pub fn new(command: &str, model: &str, workspace: std::path::PathBuf) -> Result<Self> {
        // Load existing CLI session from session store
        let session_key = "main".to_string();
        let existing_session = load_cli_session_from_store(&session_key, CLAUDE_CLI_PROVIDER);

        if let Some(ref sid) = existing_session {
            debug!("Loaded existing Claude CLI session: {}", sid);
        }

        Ok(Self {
            command: command.to_string(),
            model: normalize_claude_model(model),
            workspace,
            session_key,
            localgpt_session_id: uuid::Uuid::new_v4().to_string(),
            cli_session_id: StdMutex::new(existing_session),
        })
    }

}

/// Load CLI session ID from session store
fn load_cli_session_from_store(session_key: &str, provider: &str) -> Option<String> {
    use super::session_store::SessionStore;

    let store = SessionStore::load().ok()?;
    store.get_cli_session_id(session_key, provider)
}

/// Save CLI session ID to session store
fn save_cli_session_to_store(
    session_key: &str,
    session_id: &str,
    provider: &str,
    cli_session_id: &str,
) -> Result<()> {
    use super::session_store::SessionStore;

    let mut store = SessionStore::load()?;
    store.set_cli_session_id(session_key, session_id, provider, cli_session_id)?;
    Ok(())
}

fn normalize_claude_model(model: &str) -> String {
    match model.to_lowercase().as_str() {
        "opus" | "opus-4.5" | "opus-4" | "claude-opus-4-5" => "opus",
        "sonnet" | "sonnet-4.5" | "sonnet-4.1" | "claude-sonnet-4-5" => "sonnet",
        "haiku" | "haiku-3.5" | "claude-haiku-3-5" => "haiku",
        other => other,
    }
    .to_string()
}

fn build_prompt_from_messages(messages: &[Message]) -> String {
    // Get the last user message as the prompt
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

fn extract_system_prompt(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .find(|m| m.role == Role::System)
        .map(|m| m.content.clone())
}

/// Parse Claude CLI JSON output, returning (response_text, session_id)
fn parse_claude_cli_output(stdout: &str) -> Result<(String, Option<String>)> {
    // Claude CLI outputs JSON with message content and session info
    if let Ok(json) = serde_json::from_str::<Value>(stdout) {
        // Extract response text (try multiple field names)
        let text = json
            .get("result")
            .or_else(|| json.get("message"))
            .or_else(|| json.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| stdout.trim().to_string());

        // Extract session ID (try multiple field names per OpenClaw pattern)
        let session_id = json
            .get("session_id")
            .or_else(|| json.get("sessionId"))
            .or_else(|| json.get("conversation_id"))
            .or_else(|| json.get("conversationId"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        return Ok((text, session_id));
    }

    // Fallback: return raw output, no session
    Ok((stdout.trim().to_string(), None))
}

#[async_trait]
impl LLMProvider for ClaudeCliProvider {
    async fn chat(
        &self,
        messages: &[Message],
        _tools: Option<&[ToolSchema]>, // Ignored - no tool support
    ) -> Result<LLMResponse> {
        use std::process::Command;

        // Build prompt from messages (last user message)
        let prompt = build_prompt_from_messages(messages);
        let system_prompt = extract_system_prompt(messages);

        // Get current CLI session state
        let current_cli_session = self
            .cli_session_id
            .lock()
            .map_err(|e| anyhow::anyhow!("Session lock poisoned: {}", e))?
            .clone();
        let is_first_turn = current_cli_session.is_none();

        // Build command args
        let mut args = vec![
            "-p".to_string(),
            "--output-format".to_string(),
            "json".to_string(),
            "--dangerously-skip-permissions".to_string(),
        ];

        // Model (only on new sessions)
        if is_first_turn {
            args.push("--model".to_string());
            args.push(self.model.clone());
        }

        // System prompt (first turn only)
        if is_first_turn {
            if let Some(sys) = system_prompt {
                args.push("--append-system-prompt".to_string());
                args.push(sys);
            }
        }

        // CLI session handling
        if let Some(cli_sid) = &current_cli_session {
            // Resume existing CLI session
            args.push("--resume".to_string());
            args.push(cli_sid.clone());
        } else {
            // New CLI session - generate UUID
            let new_cli_session = uuid::Uuid::new_v4().to_string();
            args.push("--session-id".to_string());
            args.push(new_cli_session);
        }

        // Add prompt as final argument
        args.push(prompt);

        debug!(
            "Claude CLI: {} {:?} (cwd: {:?})",
            self.command, args, self.workspace
        );

        // Execute command (blocking - wrap in spawn_blocking for async)
        let output = tokio::task::spawn_blocking({
            let command = self.command.clone();
            let args = args.clone();
            let workspace = self.workspace.clone();
            move || {
                Command::new(&command)
                    .args(&args)
                    .current_dir(&workspace)
                    .output()
            }
        })
        .await??;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Claude CLI failed: {}", stderr);
        }

        // Parse JSON output and extract session ID
        let stdout = String::from_utf8_lossy(&output.stdout);
        let (response, new_session_id) = parse_claude_cli_output(&stdout)?;

        // Update CLI session ID for next turn and persist to session store
        if let Some(ref new_cli_sid) = new_session_id {
            let mut cli_session = self
                .cli_session_id
                .lock()
                .map_err(|e| anyhow::anyhow!("Session lock poisoned: {}", e))?;
            *cli_session = Some(new_cli_sid.clone());

            // Persist to session store for cross-restart continuity
            if let Err(e) = save_cli_session_to_store(
                &self.session_key,
                &self.localgpt_session_id,
                CLAUDE_CLI_PROVIDER,
                new_cli_sid,
            ) {
                debug!("Failed to persist CLI session: {}", e);
            }
        }

        Ok(LLMResponse::Text(response))
    }

    async fn summarize(&self, text: &str) -> Result<String> {
        let messages = vec![Message {
            role: Role::User,
            content: format!(
                "Summarize the following conversation concisely:\n\n{}",
                text
            ),
            tool_calls: None,
            tool_call_id: None,
        }];

        match self.chat(&messages, None).await? {
            LLMResponse::Text(summary) => Ok(summary),
            _ => anyhow::bail!("Unexpected response type"),
        }
    }

    // No streaming - uses default fallback (single chunk)
}

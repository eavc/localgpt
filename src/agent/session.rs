//! Session management with Pi-compatible JSONL format
//!
//! JSONL format matches Pi's SessionManager for OpenClaw compatibility:
//! - Header: {type: "session", version, id, timestamp, cwd}
//! - Messages: {type: "message", message: {role, content, ...}}

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};
use uuid::Uuid;

use super::providers::{LLMProvider, Message, Role, ToolCall, Usage};

/// Current session format version (matches Pi)
pub const CURRENT_SESSION_VERSION: u32 = 1;

/// Session state (internal representation)
#[derive(Debug, Clone)]
pub struct Session {
    id: String,
    created_at: DateTime<Utc>,
    cwd: String,
    messages: Vec<SessionMessage>,
    system_context: Option<String>,
    token_count: usize,
    compaction_count: u32,
    memory_flush_compaction_count: u32,
}

/// Message with metadata for persistence
#[derive(Debug, Clone)]
pub struct SessionMessage {
    pub message: Message,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api: Option<String>,
    pub usage: Option<MessageUsage>,
    pub stop_reason: Option<String>,
    pub timestamp: u64,
}

/// Per-message usage tracking (Pi-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageUsage {
    pub input: u64,
    pub output: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<u64>,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<MessageCost>,
}

/// Cost breakdown (Pi-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageCost {
    pub input: f64,
    pub output: f64,
    pub total: f64,
}

impl From<&Usage> for MessageUsage {
    fn from(usage: &Usage) -> Self {
        Self {
            input: usage.input_tokens,
            output: usage.output_tokens,
            cache_read: None,
            cache_write: None,
            total_tokens: usage.total(),
            cost: None, // Cost calculation not implemented
        }
    }
}

impl SessionMessage {
    pub fn new(message: Message) -> Self {
        Self {
            message,
            provider: None,
            model: None,
            api: None,
            usage: None,
            stop_reason: None,
            timestamp: Utc::now().timestamp_millis() as u64,
        }
    }

    pub fn with_metadata(
        message: Message,
        provider: Option<&str>,
        model: Option<&str>,
        usage: Option<&Usage>,
        stop_reason: Option<&str>,
    ) -> Self {
        Self {
            message,
            provider: provider.map(|s| s.to_string()),
            model: model.map(|s| s.to_string()),
            api: provider.map(|p| format!("{}-messages", p)), // e.g., "anthropic-messages"
            usage: usage.map(MessageUsage::from),
            stop_reason: stop_reason.map(|s| s.to_string()),
            timestamp: Utc::now().timestamp_millis() as u64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SessionStatus {
    pub id: String,
    pub message_count: usize,
    pub token_count: usize,
    pub compaction_count: u32,
    pub api_input_tokens: u64,
    pub api_output_tokens: u64,
}

impl Session {
    pub fn new() -> Self {
        Self::new_with_cwd(
            std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".to_string()),
        )
    }

    pub fn new_with_cwd(cwd: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now(),
            cwd,
            messages: Vec::new(),
            system_context: None,
            token_count: 0,
            compaction_count: 0,
            memory_flush_compaction_count: 0,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn compaction_count(&self) -> u32 {
        self.compaction_count
    }

    pub fn should_memory_flush(&self) -> bool {
        self.memory_flush_compaction_count <= self.compaction_count
    }

    pub fn mark_memory_flushed(&mut self) {
        self.memory_flush_compaction_count = self.compaction_count + 1;
    }

    pub fn set_system_context(&mut self, context: String) {
        self.system_context = Some(context);
        self.recalculate_tokens();
    }

    /// Add a message without metadata
    pub fn add_message(&mut self, message: Message) {
        let tokens = estimate_tokens(&message.content);
        self.token_count += tokens;
        self.messages.push(SessionMessage::new(message));
    }

    /// Add a message with provider/model metadata
    pub fn add_message_with_metadata(
        &mut self,
        message: Message,
        provider: Option<&str>,
        model: Option<&str>,
        usage: Option<&Usage>,
        stop_reason: Option<&str>,
    ) {
        let tokens = estimate_tokens(&message.content);
        self.token_count += tokens;
        self.messages.push(SessionMessage::with_metadata(
            message,
            provider,
            model,
            usage,
            stop_reason,
        ));
    }

    pub fn messages_for_llm(&self) -> Vec<Message> {
        let mut messages = Vec::new();

        if let Some(ref context) = self.system_context {
            messages.push(Message {
                role: Role::System,
                content: context.clone(),
                tool_calls: None,
                tool_call_id: None,
                images: Vec::new(),
            });
        }

        messages.extend(self.messages.iter().map(|sm| sm.message.clone()));
        messages
    }

    pub fn messages(&self) -> Vec<&Message> {
        self.messages.iter().map(|sm| &sm.message).collect()
    }

    /// Get raw session messages with metadata (for API responses)
    pub fn raw_messages(&self) -> &[SessionMessage] {
        &self.messages
    }

    pub fn user_assistant_messages(&self) -> Vec<Message> {
        self.messages
            .iter()
            .filter(|sm| matches!(sm.message.role, Role::User | Role::Assistant))
            .map(|sm| sm.message.clone())
            .collect()
    }

    pub async fn compact(&mut self, provider: &dyn LLMProvider) -> Result<()> {
        if self.messages.len() < 4 {
            return Ok(());
        }

        let keep_count = 4;
        let to_summarize = &self.messages[..self.messages.len() - keep_count];

        let text: String = to_summarize
            .iter()
            .map(|sm| format!("{:?}: {}", sm.message.role, sm.message.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let summary = provider.summarize(&text).await?;

        let mut new_messages = vec![SessionMessage::new(Message {
            role: Role::System,
            content: format!("Previous conversation summary:\n\n{}", summary),
            tool_calls: None,
            tool_call_id: None,
            images: Vec::new(),
        })];

        new_messages.extend(self.messages[self.messages.len() - keep_count..].to_vec());

        self.messages = new_messages;
        self.compaction_count += 1;
        self.recalculate_tokens();

        Ok(())
    }

    fn recalculate_tokens(&mut self) {
        self.token_count = 0;

        if let Some(ref context) = self.system_context {
            self.token_count += estimate_tokens(context);
        }

        for sm in &self.messages {
            self.token_count += estimate_tokens(&sm.message.content);
        }
    }

    /// Save session in Pi-compatible JSONL format
    pub fn save(&self) -> Result<PathBuf> {
        validate_session_id(&self.id)?;
        let dir = get_sessions_dir()?;
        fs::create_dir_all(&dir)?;

        let path = dir.join(format!("{}.jsonl", self.id));
        self.save_to_path(&path, &dir)?;
        Ok(path)
    }

    pub fn save_for_agent(&self, agent_id: &str) -> Result<PathBuf> {
        validate_session_id(&self.id)?;
        let dir = get_sessions_dir_for_agent(agent_id)?;
        fs::create_dir_all(&dir)?;

        let path = dir.join(format!("{}.jsonl", self.id));
        self.save_to_path(&path, &dir)?;
        Ok(path)
    }

    /// Write session data into `parent_dir`.  Only the filename component of
    /// `path` is used — directory components are discarded (defense-in-depth).
    fn save_to_path(&self, path: &Path, parent_dir: &Path) -> Result<()> {
        // Defense-in-depth: construct the write path from the canonical parent
        // directory and the filename component only.  This guarantees the
        // target stays inside the sessions directory even if `path` were
        // somehow manipulated (symlinks, extra components, etc.), and avoids
        // creating/truncating a file before verifying containment.
        let canonical_parent = parent_dir.canonicalize().with_context(|| {
            format!("Cannot canonicalize sessions dir: {}", parent_dir.display())
        })?;
        let filename = path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Session path has no filename component"))?;
        let safe_path = canonical_parent.join(filename);
        let mut file = File::create(&safe_path)?;

        // Write Pi-compatible header
        let header = json!({
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": self.id,
            "timestamp": self.created_at.to_rfc3339(),
            "cwd": self.cwd,
            // LocalGPT extensions (ignored by Pi but preserved)
            "compactionCount": self.compaction_count,
            "memoryFlushCompactionCount": self.memory_flush_compaction_count
        });
        writeln!(file, "{}", serde_json::to_string(&header)?)?;

        // Write system context as a system message
        if let Some(ref context) = self.system_context {
            let system_msg = self.format_message_entry(&SessionMessage::new(Message {
                role: Role::System,
                content: context.clone(),
                tool_calls: None,
                tool_call_id: None,
                images: Vec::new(),
            }));
            writeln!(file, "{}", serde_json::to_string(&system_msg)?)?;
        }

        // Write messages in Pi format
        for sm in &self.messages {
            let entry = self.format_message_entry(sm);
            writeln!(file, "{}", serde_json::to_string(&entry)?)?;
        }

        Ok(())
    }

    /// Format a message in Pi-compatible format
    fn format_message_entry(&self, sm: &SessionMessage) -> serde_json::Value {
        let role = match sm.message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => "toolResult",
        };

        // Build content array (Pi format)
        let mut content = Vec::new();

        // Add text content
        if !sm.message.content.is_empty() {
            content.push(json!({
                "type": "text",
                "text": sm.message.content
            }));
        }

        // Add images as image_url entries
        for img in &sm.message.images {
            content.push(json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", img.media_type, img.data)
                }
            }));
        }

        // Build message object
        let mut message = json!({
            "role": role,
            "content": content
        });

        // Add tool calls if present
        if let Some(ref tool_calls) = sm.message.tool_calls {
            let tc: Vec<serde_json::Value> = tool_calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments
                    })
                })
                .collect();
            message["toolCalls"] = json!(tc);
        }

        // Add tool call ID if present
        if let Some(ref id) = sm.message.tool_call_id {
            message["toolCallId"] = json!(id);
        }

        // Add metadata if available
        if let Some(ref provider) = sm.provider {
            message["provider"] = json!(provider);
        }
        if let Some(ref model) = sm.model {
            message["model"] = json!(model);
        }
        if let Some(ref api) = sm.api {
            message["api"] = json!(api);
        }
        if let Some(ref usage) = sm.usage {
            message["usage"] = serde_json::to_value(usage).unwrap_or(json!(null));
        }
        if let Some(ref reason) = sm.stop_reason {
            message["stopReason"] = json!(reason);
        }
        message["timestamp"] = json!(sm.timestamp);

        json!({
            "type": "message",
            "message": message
        })
    }

    /// Load session (supports both old and Pi formats)
    pub fn load(session_id: &str) -> Result<Self> {
        let normalized = validate_session_id(session_id)?;
        let dir = get_sessions_dir()?;
        let Some(path) = find_session_file_path_in_dir(&dir, &normalized)? else {
            anyhow::bail!("Session not found: {}", normalized);
        };

        Self::load_from_path(&path, &normalized)
    }

    fn load_from_path(path: &Path, session_id: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut session = Session {
            id: session_id.to_string(),
            created_at: Utc::now(),
            cwd: ".".to_string(),
            messages: Vec::new(),
            system_context: None,
            token_count: 0,
            compaction_count: 0,
            memory_flush_compaction_count: 0,
        };

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let entry: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue, // Skip malformed lines (session repair)
            };

            match entry["type"].as_str() {
                // Pi format header
                Some("session") => {
                    if let Some(ts) = entry["timestamp"].as_str() {
                        if let Ok(dt) = DateTime::parse_from_rfc3339(ts) {
                            session.created_at = dt.with_timezone(&Utc);
                        }
                    }
                    if let Some(cwd) = entry["cwd"].as_str() {
                        session.cwd = cwd.to_string();
                    }
                    if let Some(count) = entry["compactionCount"].as_u64() {
                        session.compaction_count = count as u32;
                    }
                    if let Some(count) = entry["memoryFlushCompactionCount"].as_u64() {
                        session.memory_flush_compaction_count = count as u32;
                    }
                }
                // Pi format message
                Some("message") => {
                    if let Some(msg_obj) = entry.get("message") {
                        if let Some(sm) = Self::parse_pi_message(msg_obj) {
                            // System messages become system_context
                            if sm.message.role == Role::System && session.system_context.is_none() {
                                session.system_context = Some(sm.message.content);
                            } else {
                                session.messages.push(sm);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        session.recalculate_tokens();
        Ok(session)
    }

    /// Parse Pi format message
    fn parse_pi_message(msg: &serde_json::Value) -> Option<SessionMessage> {
        let role = match msg["role"].as_str()? {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => Role::System,
            "toolResult" | "tool" => Role::Tool,
            _ => return None,
        };

        // Extract text content from content array
        let content = if let Some(arr) = msg["content"].as_array() {
            arr.iter()
                .filter_map(|item| {
                    if item["type"].as_str() == Some("text") {
                        item["text"].as_str().map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("")
        } else if let Some(s) = msg["content"].as_str() {
            s.to_string()
        } else {
            String::new()
        };

        // Parse tool calls
        let tool_calls = msg["toolCalls"].as_array().map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    Some(ToolCall {
                        id: tc["id"].as_str()?.to_string(),
                        name: tc["name"].as_str()?.to_string(),
                        arguments: tc["arguments"].as_str().unwrap_or("{}").to_string(),
                    })
                })
                .collect()
        });

        let tool_call_id = msg["toolCallId"].as_str().map(|s| s.to_string());

        // Parse usage
        let usage = serde_json::from_value(msg["usage"].clone()).ok();

        Some(SessionMessage {
            message: Message {
                role,
                content,
                tool_calls,
                tool_call_id,
                images: Vec::new(), // TODO: parse images from content array
            },
            provider: msg["provider"].as_str().map(|s| s.to_string()),
            model: msg["model"].as_str().map(|s| s.to_string()),
            api: msg["api"].as_str().map(|s| s.to_string()),
            usage,
            stop_reason: msg["stopReason"].as_str().map(|s| s.to_string()),
            timestamp: msg["timestamp"].as_u64().unwrap_or(0),
        })
    }

    pub fn status(&self) -> SessionStatus {
        SessionStatus {
            id: self.id.clone(),
            message_count: self.messages.len(),
            token_count: self.token_count,
            compaction_count: self.compaction_count,
            api_input_tokens: 0,
            api_output_tokens: 0,
        }
    }

    pub fn status_with_usage(&self, input_tokens: u64, output_tokens: u64) -> SessionStatus {
        SessionStatus {
            id: self.id.clone(),
            message_count: self.messages.len(),
            token_count: self.token_count,
            compaction_count: self.compaction_count,
            api_input_tokens: input_tokens,
            api_output_tokens: output_tokens,
        }
    }

    pub fn auto_save(&self) -> Result<()> {
        if self.messages.is_empty() {
            return Ok(());
        }
        self.save()?;
        Ok(())
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Default agent ID (matches OpenClaw's default)
pub const DEFAULT_AGENT_ID: &str = "main";

/// Validate that a session ID is a well-formed UUID and return the canonical
/// lowercase hyphenated representation.
///
/// Rejects path separators, `..`, and any non-UUID input to prevent path
/// traversal when the ID is used in file-system paths.
pub fn validate_session_id(id: &str) -> Result<String> {
    Uuid::parse_str(id)
        .map(|u| u.to_string()) // canonical lowercase hyphenated form
        .map_err(|_| anyhow::anyhow!("Invalid session ID: must be a valid UUID"))
}

fn get_sessions_dir() -> Result<PathBuf> {
    get_sessions_dir_for_agent(DEFAULT_AGENT_ID)
}

pub fn get_sessions_dir_for_agent(agent_id: &str) -> Result<PathBuf> {
    let base = directories::BaseDirs::new()
        .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;

    Ok(base
        .home_dir()
        .join(".localgpt")
        .join("agents")
        .join(agent_id)
        .join("sessions"))
}

pub fn get_state_dir() -> Result<PathBuf> {
    let base = directories::BaseDirs::new()
        .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;

    Ok(base.home_dir().join(".localgpt"))
}

fn find_session_file_path_in_dir(
    sessions_dir: &Path,
    normalized_id: &str,
) -> Result<Option<PathBuf>> {
    let canonical_path = sessions_dir.join(format!("{}.jsonl", normalized_id));
    if canonical_path.exists() {
        return Ok(Some(canonical_path));
    }

    if !sessions_dir.exists() {
        return Ok(None);
    }

    // Backward compatibility for legacy filenames:
    // find any UUID-parseable stem that normalizes to the same canonical ID.
    for entry in fs::read_dir(sessions_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e != "jsonl").unwrap_or(true) {
            continue;
        }

        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };

        let Ok(candidate) = validate_session_id(stem) else {
            continue;
        };

        if candidate == normalized_id {
            return Ok(Some(path));
        }
    }

    Ok(None)
}

pub fn find_session_file_path_for_agent(
    agent_id: &str,
    session_id: &str,
) -> Result<Option<PathBuf>> {
    let normalized = validate_session_id(session_id)?;
    let sessions_dir = get_sessions_dir_for_agent(agent_id)?;
    find_session_file_path_in_dir(&sessions_dir, &normalized)
}

fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub message_count: usize,
    pub file_size: u64,
}

pub fn list_sessions() -> Result<Vec<SessionInfo>> {
    list_sessions_for_agent(DEFAULT_AGENT_ID)
}

pub fn list_sessions_for_agent(agent_id: &str) -> Result<Vec<SessionInfo>> {
    let sessions_dir = get_sessions_dir_for_agent(agent_id)?;

    if !sessions_dir.exists() {
        return Ok(Vec::new());
    }

    let mut sessions = Vec::new();
    let mut seen_ids = HashSet::new();

    for entry in fs::read_dir(&sessions_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e != "jsonl").unwrap_or(true) {
            continue;
        }

        let filename = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

        // Only return sessions with valid UUID filenames.  This ensures
        // consistency: every listed session can also be opened via
        // `Session::load` / `get_saved_session` which now require UUIDs.
        let normalized_id = match validate_session_id(filename) {
            Ok(id) => id,
            Err(_) => {
                debug!("Skipping non-UUID session file: {}", path.display());
                continue;
            }
        };

        if !seen_ids.insert(normalized_id.clone()) {
            debug!(
                "Skipping duplicate session ID in listing: {}",
                normalized_id
            );
            continue;
        }

        let metadata = fs::metadata(&path)?;
        let file_size = metadata.len();

        if let Ok(file) = File::open(&path) {
            let reader = BufReader::new(file);
            if let Some(Ok(first_line)) = reader.lines().next() {
                if let Ok(header) = serde_json::from_str::<serde_json::Value>(&first_line) {
                    // Pi format header
                    if header["type"].as_str() == Some("session") {
                        let created_at = header["timestamp"]
                            .as_str()
                            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(Utc::now);

                        let message_count = fs::read_to_string(&path)
                            .map(|s| s.lines().count().saturating_sub(1))
                            .unwrap_or(0);

                        sessions.push(SessionInfo {
                            id: normalized_id,
                            created_at,
                            message_count,
                            file_size,
                        });
                    }
                }
            }
        }
    }

    sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(sessions)
}

pub fn get_last_session_id() -> Result<Option<String>> {
    get_last_session_id_for_agent(DEFAULT_AGENT_ID)
}

pub fn get_last_session_id_for_agent(agent_id: &str) -> Result<Option<String>> {
    let sessions = list_sessions_for_agent(agent_id)?;
    Ok(sessions.first().map(|s| s.id.clone()))
}

#[derive(Debug, Clone)]
pub struct SessionSearchResult {
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub message_preview: String,
    pub match_count: usize,
}

pub fn search_sessions(query: &str) -> Result<Vec<SessionSearchResult>> {
    search_sessions_for_agent(DEFAULT_AGENT_ID, query)
}

pub fn search_sessions_for_agent(agent_id: &str, query: &str) -> Result<Vec<SessionSearchResult>> {
    let sessions_dir = get_sessions_dir_for_agent(agent_id)?;

    if !sessions_dir.exists() {
        return Ok(Vec::new());
    }

    let query_lower = query.to_lowercase();
    let mut results = Vec::new();

    for entry in fs::read_dir(&sessions_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
            if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                if let Ok(content) = fs::read_to_string(&path) {
                    let content_lower = content.to_lowercase();
                    let match_count = content_lower.matches(&query_lower).count();

                    if match_count > 0 {
                        let preview = extract_match_preview(&content, &query_lower, 100);

                        let created_at = fs::metadata(&path)
                            .and_then(|m| m.created())
                            .map(DateTime::<Utc>::from)
                            .unwrap_or_else(|_| Utc::now());

                        results.push(SessionSearchResult {
                            session_id: filename.to_string(),
                            created_at,
                            message_preview: preview,
                            match_count,
                        });
                    }
                }
            }
        }
    }

    results.sort_by(|a, b| b.match_count.cmp(&a.match_count));
    Ok(results)
}

/// Purge session files older than `retention_days` for the given agent.
///
/// Returns the number of files deleted. Individual files that cannot be
/// removed (permissions, I/O errors) are logged and skipped. The function
/// can fail if the sessions directory cannot be read.
///
/// A `retention_days` of 0 is a no-op (keep forever).
pub fn purge_expired_sessions(agent_id: &str, retention_days: u32) -> Result<u32> {
    if retention_days == 0 {
        return Ok(0);
    }

    let sessions_dir = get_sessions_dir_for_agent(agent_id)?;
    let deleted = purge_expired_sessions_in_dir(&sessions_dir, retention_days)?;

    if deleted > 0 {
        tracing::info!(
            "Purged {} expired session(s) for agent '{}' (retention: {} days)",
            deleted,
            agent_id,
            retention_days
        );
    }

    Ok(deleted)
}

/// Core purge logic operating on a specific directory.
fn purge_expired_sessions_in_dir(sessions_dir: &Path, retention_days: u32) -> Result<u32> {
    if !sessions_dir.exists() {
        return Ok(0);
    }

    let cutoff = Utc::now() - chrono::Duration::days(i64::from(retention_days));
    let mut deleted = 0u32;

    for entry in fs::read_dir(sessions_dir)? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();

        if path.extension().map(|e| e != "jsonl").unwrap_or(true) {
            continue;
        }

        // Only purge files with valid UUID filenames — leave other files alone.
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if validate_session_id(stem).is_err() {
            continue;
        }

        // Use file modification time (last write) as the session age indicator.
        let modified = match fs::metadata(&path).and_then(|m| m.modified()) {
            Ok(t) => DateTime::<Utc>::from(t),
            Err(_) => continue,
        };

        if modified < cutoff {
            match fs::remove_file(&path) {
                Ok(()) => {
                    debug!(
                        "Purged expired session file: {} (modified {})",
                        path.display(),
                        modified.format("%Y-%m-%d %H:%M")
                    );
                    deleted += 1;
                }
                Err(e) => {
                    warn!("Failed to purge session file {}: {}", path.display(), e);
                }
            }
        }
    }

    Ok(deleted)
}

fn extract_match_preview(content: &str, query_lower: &str, max_len: usize) -> String {
    let content_lower = content.to_lowercase();

    if let Some(pos) = content_lower.find(query_lower) {
        let half_len = max_len / 2;
        let start = pos.saturating_sub(half_len);
        let end = (pos + query_lower.len() + half_len).min(content.len());

        let slice = &content[start..end];
        let cleaned: String = slice
            .chars()
            .map(|c| if c.is_whitespace() { ' ' } else { c })
            .collect();

        let trimmed = cleaned.trim();
        let prefix = if start > 0 { "..." } else { "" };
        let suffix = if end < content.len() { "..." } else { "" };

        format!("{}{}{}", prefix, trimmed, suffix)
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_new() {
        let session = Session::new();
        assert!(!session.id().is_empty());
        assert_eq!(session.token_count(), 0);
        assert_eq!(session.compaction_count(), 0);
    }

    #[test]
    fn test_message_usage_from() {
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 50,
        };
        let msg_usage = MessageUsage::from(&usage);
        assert_eq!(msg_usage.input, 100);
        assert_eq!(msg_usage.output, 50);
        assert_eq!(msg_usage.total_tokens, 150);
    }

    #[test]
    fn test_validate_session_id_accepts_valid_uuid() {
        let id = Uuid::new_v4().to_string();
        let result = validate_session_id(&id).unwrap();
        assert_eq!(result, id); // already lowercase
    }

    #[test]
    fn test_validate_session_id_normalizes_uppercase_to_lowercase() {
        let result = validate_session_id("A1B2C3D4-E5F6-7890-ABCD-EF1234567890").unwrap();
        assert_eq!(result, "a1b2c3d4-e5f6-7890-abcd-ef1234567890");
    }

    #[test]
    fn test_validate_session_id_rejects_path_traversal() {
        assert!(validate_session_id("../../etc/passwd").is_err());
        assert!(validate_session_id("../../../tmp/evil").is_err());
    }

    #[test]
    fn test_validate_session_id_rejects_path_separator() {
        assert!(validate_session_id("foo/bar").is_err());
        assert!(validate_session_id("foo\\bar").is_err());
    }

    #[test]
    fn test_validate_session_id_rejects_empty() {
        assert!(validate_session_id("").is_err());
    }

    #[test]
    fn test_validate_session_id_rejects_arbitrary_string() {
        assert!(validate_session_id("not-a-uuid").is_err());
        assert!(validate_session_id("my-session-name").is_err());
        assert!(validate_session_id("12345").is_err());
    }

    #[test]
    fn test_save_to_path_writes_inside_parent() {
        let dir = std::env::temp_dir().join(format!("localgpt-test-save-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let session = Session::new();
        let path = dir.join(format!("{}.jsonl", session.id()));
        let result = session.save_to_path(&path, &dir);
        assert!(result.is_ok());

        let canonical_dir = dir.canonicalize().unwrap();
        let created = canonical_dir.join(format!("{}.jsonl", session.id()));
        assert!(created.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_save_to_path_discards_traversal_components() {
        // Even if `path` contains ../ segments, save_to_path only uses the
        // filename component, so the file ends up inside `parent_dir`.
        let dir = std::env::temp_dir().join(format!("localgpt-test-traversal-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let session = Session::new();
        let evil_path = dir
            .join("..")
            .join("..")
            .join("tmp")
            .join(format!("{}.jsonl", session.id()));
        let result = session.save_to_path(&evil_path, &dir);
        assert!(result.is_ok());

        // File should be inside `dir`, not in /tmp
        let canonical_dir = dir.canonicalize().unwrap();
        let safe_file = canonical_dir.join(format!("{}.jsonl", session.id()));
        assert!(safe_file.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_purge_expired_sessions_zero_retention_is_noop() {
        let result = purge_expired_sessions("test-noop", 0).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_purge_expired_sessions_nonexistent_dir_returns_zero() {
        let result = purge_expired_sessions("nonexistent-agent-id-12345", 30).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_purge_expired_sessions_deletes_old_files() {
        use std::time::{Duration, SystemTime};

        let dir = std::env::temp_dir().join(format!("localgpt-test-purge-old-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        // Create an "old" session file and backdate its modification time
        let old_id = Uuid::new_v4().to_string();
        let old_file = dir.join(format!("{}.jsonl", old_id));
        std::fs::write(&old_file, "{}\n").unwrap();

        // Backdate to 60 days ago
        let sixty_days_ago = SystemTime::now() - Duration::from_secs(60 * 86400);
        filetime::set_file_mtime(
            &old_file,
            filetime::FileTime::from_system_time(sixty_days_ago),
        )
        .unwrap();

        // Create a "recent" session file (default mtime = now)
        let recent_id = Uuid::new_v4().to_string();
        let recent_file = dir.join(format!("{}.jsonl", recent_id));
        std::fs::write(&recent_file, "{}\n").unwrap();

        // Create a non-UUID file that should be left alone even if old
        let other_file = dir.join("notes.jsonl");
        std::fs::write(&other_file, "keep me\n").unwrap();
        filetime::set_file_mtime(
            &other_file,
            filetime::FileTime::from_system_time(sixty_days_ago),
        )
        .unwrap();

        // Call the real purge function with 30-day retention
        let deleted = purge_expired_sessions_in_dir(&dir, 30).unwrap();

        assert_eq!(deleted, 1, "should delete exactly the old UUID session");
        assert!(!old_file.exists(), "old session file should be deleted");
        assert!(recent_file.exists(), "recent session file should remain");
        assert!(other_file.exists(), "non-UUID file should remain");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_purge_expired_sessions_keeps_recent_files() {
        let dir = std::env::temp_dir().join(format!("localgpt-test-purge-keep-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        // Create three recent session files
        for _ in 0..3 {
            let id = Uuid::new_v4().to_string();
            let file = dir.join(format!("{}.jsonl", id));
            std::fs::write(&file, "{}\n").unwrap();
        }

        // Call the real purge function with 7-day retention — nothing should be deleted
        let deleted = purge_expired_sessions_in_dir(&dir, 7).unwrap();

        assert_eq!(deleted, 0, "no recent files should be deleted");

        let count = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(count, 3, "all three session files should remain");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_find_session_file_path_in_dir_matches_legacy_uppercase_filename() {
        let dir = std::env::temp_dir().join(format!(
            "localgpt-test-find-legacy-uppercase-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
        let legacy_upper = dir.join(format!("{}.jsonl", id.to_uppercase()));
        std::fs::write(&legacy_upper, "{}\n").unwrap();

        let resolved = find_session_file_path_in_dir(&dir, id).unwrap().unwrap();
        assert!(resolved.exists());
        let resolved_stem = resolved.file_stem().and_then(|s| s.to_str()).unwrap();
        assert_eq!(validate_session_id(resolved_stem).unwrap(), id);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_session_config_defaults_retention_days_zero() {
        use crate::config::Config;
        let config = Config::default();
        assert_eq!(
            config.session.retention_days, 0,
            "retention_days must default to 0 (keep forever)"
        );
    }

    #[test]
    fn test_session_config_retention_days_deserializes() {
        use crate::config::Config;
        let toml_str = r#"
[session]
retention_days = 90
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.session.retention_days, 90);
    }

    #[test]
    fn test_session_config_retention_days_omitted_defaults_zero() {
        use crate::config::Config;
        let toml_str = r#"
[logging]
level = "info"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.session.retention_days, 0);
    }

    #[test]
    fn test_session_config_get_set_retention_days() {
        use crate::config::Config;
        let mut config = Config::default();
        assert_eq!(config.get_value("session.retention_days").unwrap(), "0");

        config.set_value("session.retention_days", "45").unwrap();
        assert_eq!(config.session.retention_days, 45);
        assert_eq!(config.get_value("session.retention_days").unwrap(), "45");
    }
}

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, warn};

use super::providers::ToolSchema;
use crate::config::Config;
use crate::memory::MemoryManager;

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub call_id: String,
    pub output: String,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn schema(&self) -> ToolSchema;
    async fn execute(&self, arguments: &str) -> Result<String>;
}

/// Enforces that file tool operations stay within allowed directory roots.
///
/// The workspace directory is always permitted. Additional directories can be
/// configured via `tools.allowed_paths` in config.
#[derive(Debug, Clone)]
pub struct PathSandbox {
    /// Canonicalized roots that file tools may access.
    allowed_roots: Vec<PathBuf>,
    /// Workspace root used to resolve relative paths.
    workspace_root: PathBuf,
}

impl PathSandbox {
    /// Build a sandbox from the workspace path and any extra allowed paths from config.
    pub fn new(workspace: &Path, extra_allowed: &[String]) -> Self {
        let mut allowed_roots = Vec::new();
        let workspace_root = if let Ok(canonical) = fs::canonicalize(workspace) {
            canonical
        } else {
            workspace.to_path_buf()
        };

        // Workspace is always allowed — canonicalize if it exists, otherwise use as-is
        allowed_roots.push(workspace_root.clone());

        for raw in extra_allowed {
            let expanded = shellexpand::tilde(raw);
            let path = PathBuf::from(expanded.to_string());
            if let Ok(canonical) = fs::canonicalize(&path) {
                allowed_roots.push(canonical);
            } else {
                warn!(
                    "tools.allowed_paths entry does not exist, skipping: {}",
                    raw
                );
            }
        }

        Self {
            allowed_roots,
            workspace_root,
        }
    }

    /// Validate and expand a user-provided path string.
    ///
    /// For **read** operations the target file must already exist so we can
    /// `canonicalize()` the full path and compare against the allow-list.
    ///
    /// For **write** operations the file itself may not exist yet, so we
    /// canonicalize the nearest existing ancestor and verify *that* falls
    /// within an allowed root. The returned path is the canonicalized
    /// ancestor joined with the remaining unresolved components.
    pub fn validate(&self, raw_path: &str, must_exist: bool) -> Result<PathBuf> {
        let expanded = shellexpand::tilde(raw_path).to_string();
        let mut path = PathBuf::from(&expanded);
        if path.is_relative() {
            path = self.workspace_root.join(path);
        }

        if must_exist {
            // Canonicalize the full path — this resolves symlinks and `..`
            let canonical = fs::canonicalize(&path)
                .with_context(|| format!("Cannot resolve path: {}", raw_path))?;
            self.check_allowed(&canonical, raw_path)?;
            Ok(canonical)
        } else {
            // Walk up until we find an existing ancestor we can canonicalize
            let (canonical_ancestor, remainder) = self.canonicalize_ancestor(&path)?;
            self.check_allowed(&canonical_ancestor, raw_path)?;
            Ok(canonical_ancestor.join(remainder))
        }
    }

    /// Check whether a canonicalized path falls under any allowed root.
    fn check_allowed(&self, canonical: &Path, original: &str) -> Result<()> {
        for root in &self.allowed_roots {
            if canonical.starts_with(root) {
                return Ok(());
            }
        }
        warn!(
            "Path access denied — outside sandbox: {} (resolved: {})",
            original,
            canonical.display()
        );
        anyhow::bail!(
            "Access denied: path '{}' is outside the allowed directories. \
             File tools may only access the workspace and paths listed in tools.allowed_paths.",
            original
        )
    }

    /// Walk up the path to find the nearest existing ancestor and return
    /// `(canonicalized_ancestor, remaining_suffix)`.
    fn canonicalize_ancestor(&self, path: &Path) -> Result<(PathBuf, PathBuf)> {
        let abs_path = if path.is_relative() {
            self.workspace_root.join(path)
        } else {
            path.to_path_buf()
        };

        let mut current = abs_path.clone();
        let mut suffix_parts: Vec<std::ffi::OsString> = Vec::new();

        loop {
            match fs::canonicalize(&current) {
                Ok(canonical) => {
                    let mut remainder = PathBuf::new();
                    for part in suffix_parts.into_iter().rev() {
                        remainder.push(part);
                    }
                    return Ok((canonical, remainder));
                }
                Err(_) => {
                    if let Some(file_name) = current.file_name() {
                        suffix_parts.push(file_name.to_os_string());
                    }
                    if !current.pop() {
                        anyhow::bail!(
                            "Cannot resolve any ancestor of path: {}",
                            abs_path.display()
                        );
                    }
                }
            }
        }
    }
}

pub fn create_default_tools(
    config: &Config,
    memory: Option<Arc<MemoryManager>>,
) -> Result<Vec<Box<dyn Tool>>> {
    let workspace = config.workspace_path();
    let sandbox = Arc::new(PathSandbox::new(&workspace, &config.tools.allowed_paths));

    // Use indexed memory search if MemoryManager is provided, otherwise fallback to grep-based
    let memory_search_tool: Box<dyn Tool> = if let Some(ref mem) = memory {
        Box::new(MemorySearchToolWithIndex::new(Arc::clone(mem)))
    } else {
        Box::new(MemorySearchTool::new(workspace.clone()))
    };

    Ok(vec![
        Box::new(BashTool::new(config.tools.bash_timeout_ms)),
        Box::new(ReadFileTool::new(Arc::clone(&sandbox))),
        Box::new(WriteFileTool::new(Arc::clone(&sandbox))),
        Box::new(EditFileTool::new(Arc::clone(&sandbox))),
        memory_search_tool,
        Box::new(MemoryGetTool::new(workspace)),
        Box::new(WebFetchTool::new(config.tools.web_fetch_max_bytes)),
    ])
}

/// Create a restricted tool set for the heartbeat agent.
///
/// Only tools named in `config.heartbeat.allowed_tools` are instantiated.
/// Tools not on the allowlist simply do not exist in the returned vec —
/// this is a hard construction-time restriction, not a runtime gate.
///
/// Implementation note: we construct all default tools then filter down to the
/// allowlist. This is slightly wasteful (discarded tools are allocated then
/// dropped) but keeps the logic simple — a name→constructor factory would add
/// complexity that isn't justified for the small number of tools involved.
pub fn create_heartbeat_tools(
    config: &Config,
    memory: Option<Arc<MemoryManager>>,
) -> Result<Vec<Box<dyn Tool>>> {
    let all_tools = create_default_tools(config, memory)?;
    let total = all_tools.len();
    let allowed = &config.heartbeat.allowed_tools;

    let filtered: Vec<Box<dyn Tool>> = all_tools
        .into_iter()
        .filter(|t| allowed.iter().any(|a| a == t.name()))
        .collect();

    // Warn about allowlist entries that don't match any known tool (config typos)
    let matched_names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
    for name in allowed {
        if !matched_names.contains(&name.as_str()) {
            warn!(
                "Heartbeat allowed_tools entry {:?} does not match any known tool — check config for typos",
                name
            );
        }
    }

    debug!(
        "Heartbeat tools: {} of {} total allowed ({:?})",
        filtered.len(),
        total,
        matched_names
    );

    Ok(filtered)
}

// Bash Tool
pub struct BashTool {
    default_timeout_ms: u64,
}

impl BashTool {
    pub fn new(default_timeout_ms: u64) -> Self {
        Self { default_timeout_ms }
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "bash".to_string(),
            description: "Execute a bash command and return the output".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": format!("Optional timeout in milliseconds (default: {})", self.default_timeout_ms)
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let command = args["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing command"))?;

        let timeout_ms = args["timeout_ms"]
            .as_u64()
            .unwrap_or(self.default_timeout_ms);

        debug!(
            "Executing bash command (timeout: {}ms): {}",
            timeout_ms, command
        );

        // Run command with timeout
        let timeout_duration = std::time::Duration::from_millis(timeout_ms);
        let output = tokio::time::timeout(
            timeout_duration,
            tokio::process::Command::new("bash")
                .arg("-c")
                .arg(command)
                .output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Command timed out after {}ms", timeout_ms))??;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();

        if !stdout.is_empty() {
            result.push_str(&stdout);
        }

        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push_str("\n\nSTDERR:\n");
            }
            result.push_str(&stderr);
        }

        if result.is_empty() {
            result = format!(
                "Command completed with exit code: {}",
                output.status.code().unwrap_or(-1)
            );
        }

        Ok(result)
    }
}

// Read File Tool
pub struct ReadFileTool {
    sandbox: Arc<PathSandbox>,
}

impl ReadFileTool {
    pub fn new(sandbox: Arc<PathSandbox>) -> Self {
        Self { sandbox }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "read_file".to_string(),
            description: "Read the contents of a file within the workspace".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read (must be within workspace)"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-indexed)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let raw_path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing path"))?;

        let path = self.sandbox.validate(raw_path, true)?;

        debug!("Reading file: {}", path.display());

        let content = fs::read_to_string(&path)?;

        // Handle offset and limit
        let offset = args["offset"].as_u64().unwrap_or(0) as usize;
        let limit = args["limit"].as_u64().map(|l| l as usize);

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start = offset.min(total_lines);
        let end = limit
            .map(|l| (start + l).min(total_lines))
            .unwrap_or(total_lines);

        let selected: Vec<String> = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:4}\t{}", start + i + 1, line))
            .collect();

        Ok(selected.join("\n"))
    }
}

// Write File Tool
pub struct WriteFileTool {
    sandbox: Arc<PathSandbox>,
}

impl WriteFileTool {
    pub fn new(sandbox: Arc<PathSandbox>) -> Self {
        Self { sandbox }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "write_file".to_string(),
            description: "Write content to a file within the workspace (creates or overwrites)"
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write (must be within workspace)"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let raw_path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing path"))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing content"))?;

        // Validate path — file may not exist yet, so must_exist=false
        let path = self.sandbox.validate(raw_path, false)?;

        debug!("Writing file: {}", path.display());

        // Create parent directories if needed (only within validated path)
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&path, content)?;

        Ok(format!(
            "Successfully wrote {} bytes to {}",
            content.len(),
            path.display()
        ))
    }
}

// Edit File Tool
pub struct EditFileTool {
    sandbox: Arc<PathSandbox>,
}

impl EditFileTool {
    pub fn new(sandbox: Arc<PathSandbox>) -> Self {
        Self { sandbox }
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "edit_file".to_string(),
            description: "Edit a file within the workspace by replacing old_string with new_string"
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to edit (must be within workspace)"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The text to replace"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false)"
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        }
    }

    async fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let raw_path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing path"))?;
        let old_string = args["old_string"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing old_string"))?;
        let new_string = args["new_string"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing new_string"))?;
        let replace_all = args["replace_all"].as_bool().unwrap_or(false);

        let path = self.sandbox.validate(raw_path, true)?;

        debug!("Editing file: {}", path.display());

        let content = fs::read_to_string(&path)?;

        let (new_content, count) = if replace_all {
            let count = content.matches(old_string).count();
            (content.replace(old_string, new_string), count)
        } else if content.contains(old_string) {
            (content.replacen(old_string, new_string, 1), 1)
        } else {
            return Err(anyhow::anyhow!("old_string not found in file"));
        };

        fs::write(&path, &new_content)?;

        Ok(format!(
            "Replaced {} occurrence(s) in {}",
            count,
            path.display()
        ))
    }
}

// Memory Search Tool
pub struct MemorySearchTool {
    workspace: PathBuf,
}

impl MemorySearchTool {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }
}

#[async_trait]
impl Tool for MemorySearchTool {
    fn name(&self) -> &str {
        "memory_search"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "memory_search".to_string(),
            description: "Search the memory index for relevant information".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let query = args["query"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing query"))?;
        let limit = args["limit"].as_u64().unwrap_or(5) as usize;

        debug!("Memory search: {} (limit: {})", query, limit);

        // Simple grep-based search for now
        // TODO: Use proper memory index
        let mut results = Vec::new();

        let memory_file = self.workspace.join("MEMORY.md");
        if memory_file.exists() {
            if let Ok(content) = fs::read_to_string(&memory_file) {
                for (i, line) in content.lines().enumerate() {
                    if line.to_lowercase().contains(&query.to_lowercase()) {
                        results.push(format!("MEMORY.md:{}: {}", i + 1, line));
                        if results.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }

        // Search daily logs
        let memory_dir = self.workspace.join("memory");
        if memory_dir.exists() {
            if let Ok(entries) = fs::read_dir(&memory_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    if results.len() >= limit {
                        break;
                    }

                    let path = entry.path();
                    if path.extension().map(|e| e == "md").unwrap_or(false) {
                        if let Ok(content) = fs::read_to_string(&path) {
                            let filename = path.file_name().unwrap().to_string_lossy();
                            for (i, line) in content.lines().enumerate() {
                                if line.to_lowercase().contains(&query.to_lowercase()) {
                                    results.push(format!(
                                        "memory/{}:{}: {}",
                                        filename,
                                        i + 1,
                                        line
                                    ));
                                    if results.len() >= limit {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if results.is_empty() {
            Ok("No results found".to_string())
        } else {
            Ok(results.join("\n"))
        }
    }
}

// Memory Search Tool with Index - uses MemoryManager for hybrid FTS+vector search
pub struct MemorySearchToolWithIndex {
    memory: Arc<MemoryManager>,
}

impl MemorySearchToolWithIndex {
    pub fn new(memory: Arc<MemoryManager>) -> Self {
        Self { memory }
    }
}

#[async_trait]
impl Tool for MemorySearchToolWithIndex {
    fn name(&self) -> &str {
        "memory_search"
    }

    fn schema(&self) -> ToolSchema {
        let description = if self.memory.has_embeddings() {
            "Search the memory index using hybrid semantic + keyword search for relevant information"
        } else {
            "Search the memory index for relevant information"
        };

        ToolSchema {
            name: "memory_search".to_string(),
            description: description.to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let query = args["query"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing query"))?;
        let limit = args["limit"].as_u64().unwrap_or(5) as usize;

        let search_type = if self.memory.has_embeddings() {
            "hybrid"
        } else {
            "FTS"
        };
        debug!(
            "Memory search ({}): {} (limit: {})",
            search_type, query, limit
        );

        let results = self.memory.search(query, limit)?;

        if results.is_empty() {
            return Ok("No results found".to_string());
        }

        // Format results with relevance scores
        let formatted: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let preview: String = chunk.content.chars().take(200).collect();
                let preview = preview.replace('\n', " ");
                format!(
                    "{}. {} (lines {}-{}, score: {:.3})\n   {}{}",
                    i + 1,
                    chunk.file,
                    chunk.line_start,
                    chunk.line_end,
                    chunk.score,
                    preview,
                    if chunk.content.len() > 200 { "..." } else { "" }
                )
            })
            .collect();

        Ok(formatted.join("\n\n"))
    }
}

// Memory Get Tool - efficient snippet fetching after memory_search
pub struct MemoryGetTool {
    workspace: PathBuf,
}

impl MemoryGetTool {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }

    fn resolve_path(&self, path: &str) -> Result<PathBuf> {
        // Only allow workspace-relative memory paths
        if path.starts_with("memory/") || path == "MEMORY.md" || path == "HEARTBEAT.md" {
            // Reject paths containing parent-dir components to prevent traversal
            // (e.g., "memory/../../etc/passwd"). Uses component-based check so
            // filenames that legitimately contain ".." are not blocked.
            if Path::new(path)
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                warn!("memory_get: path traversal detected via '..': {}", path);
                anyhow::bail!(
                    "Access denied: path '{}' contains traversal components",
                    path
                );
            }
            let joined = self.workspace.join(path);
            // If the file exists, canonicalize and re-verify as defense-in-depth
            if joined.exists() {
                let canonical = fs::canonicalize(&joined)?;
                let workspace_canonical =
                    fs::canonicalize(&self.workspace).unwrap_or_else(|_| self.workspace.clone());
                if !canonical.starts_with(&workspace_canonical) {
                    warn!(
                        "memory_get: resolved path outside workspace: {} -> {}",
                        path,
                        canonical.display()
                    );
                    anyhow::bail!(
                        "Access denied: path '{}' resolves outside the workspace",
                        path
                    );
                }
                Ok(canonical)
            } else {
                // File doesn't exist yet — return the joined path for the
                // caller's "File not found" handling
                Ok(joined)
            }
        } else {
            // Reject any path that doesn't match known memory file patterns
            warn!(
                "memory_get: rejected path outside workspace memory: {}",
                path
            );
            anyhow::bail!(
                "Access denied: memory_get only supports workspace memory files \
                 (MEMORY.md, HEARTBEAT.md, memory/*.md). Got: {}",
                path
            )
        }
    }
}

#[async_trait]
impl Tool for MemoryGetTool {
    fn name(&self) -> &str {
        "memory_get"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "memory_get".to_string(),
            description: "Safe snippet read from MEMORY.md, HEARTBEAT.md, or memory/*.md with optional line range; use after memory_search to pull only the needed lines and keep context small.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (e.g., 'MEMORY.md' or 'memory/2024-01-15.md')"
                    },
                    "from": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed, default: 1)"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines to read (default: 50)"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing path"))?;

        let from = args["from"].as_u64().unwrap_or(1).max(1) as usize;
        let lines_count = args["lines"].as_u64().unwrap_or(50) as usize;

        let resolved_path = self.resolve_path(path)?;

        debug!(
            "Memory get: {} (from: {}, lines: {})",
            resolved_path.display(),
            from,
            lines_count
        );

        if !resolved_path.exists() {
            return Ok(format!("File not found: {}", path));
        }

        let content = fs::read_to_string(&resolved_path)?;
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        // Convert from 1-indexed to 0-indexed
        let start = (from - 1).min(total_lines);
        let end = (start + lines_count).min(total_lines);

        if start >= total_lines {
            return Ok(format!(
                "Line {} is past end of file ({} lines)",
                from, total_lines
            ));
        }

        let selected: Vec<String> = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:4}\t{}", start + i + 1, line))
            .collect();

        let header = format!(
            "# {} (lines {}-{} of {})\n",
            path,
            start + 1,
            end,
            total_lines
        );
        Ok(header + &selected.join("\n"))
    }
}

// Web Fetch Tool
pub struct WebFetchTool {
    client: reqwest::Client,
    max_bytes: usize,
}

impl WebFetchTool {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            max_bytes,
        }
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "web_fetch".to_string(),
            description: "Fetch content from a URL".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    async fn execute(&self, arguments: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let url = args["url"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing url"))?;

        debug!("Fetching URL: {}", url);

        let response = self
            .client
            .get(url)
            .header("User-Agent", "LocalGPT/0.1")
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        // Truncate if too long
        let truncated = if body.len() > self.max_bytes {
            format!(
                "{}...\n\n[Truncated, {} bytes total]",
                &body[..self.max_bytes],
                body.len()
            )
        } else {
            body
        };

        Ok(format!("Status: {}\n\n{}", status, truncated))
    }
}

/// Extract relevant detail from tool arguments for display.
/// Returns a human-readable summary of the key argument (file path, command, query, URL).
pub fn extract_tool_detail(tool_name: &str, arguments: &str) -> Option<String> {
    let args: Value = serde_json::from_str(arguments).ok()?;

    match tool_name {
        "edit_file" | "write_file" | "read_file" => args
            .get("path")
            .or_else(|| args.get("file_path"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        "bash" => args.get("command").and_then(|v| v.as_str()).map(|s| {
            if s.len() > 60 {
                format!("{}...", &s[..57])
            } else {
                s.to_string()
            }
        }),
        "memory_search" => args
            .get("query")
            .and_then(|v| v.as_str())
            .map(|s| format!("\"{}\"", s)),
        "web_fetch" => args
            .get("url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn setup_sandbox() -> (TempDir, PathSandbox) {
        let workspace = TempDir::new().unwrap();
        let sandbox = PathSandbox::new(workspace.path(), &[]);
        (workspace, sandbox)
    }

    fn setup_sandbox_with_extra(extra: &[String]) -> (TempDir, PathSandbox) {
        let workspace = TempDir::new().unwrap();
        let sandbox = PathSandbox::new(workspace.path(), extra);
        (workspace, sandbox)
    }

    #[test]
    fn test_sandbox_validate_allows_workspace_file() {
        let (workspace, sandbox) = setup_sandbox();
        let file = workspace.path().join("test.md");
        fs::write(&file, "hello").unwrap();

        let result = sandbox.validate(file.to_str().unwrap(), true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_validate_allows_workspace_subdir() {
        let (workspace, sandbox) = setup_sandbox();
        let subdir = workspace.path().join("memory");
        fs::create_dir_all(&subdir).unwrap();
        let file = subdir.join("2024-01-01.md");
        fs::write(&file, "log").unwrap();

        let result = sandbox.validate(file.to_str().unwrap(), true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_validate_denies_outside_workspace() {
        let (_workspace, sandbox) = setup_sandbox();

        let result = sandbox.validate("/etc/passwd", true);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Access denied"));
    }

    #[test]
    fn test_sandbox_validate_denies_traversal_attack() {
        let (workspace, sandbox) = setup_sandbox();
        // Create a file in workspace so traversal has something to resolve from
        let file = workspace.path().join("test.md");
        fs::write(&file, "hello").unwrap();

        // Try to escape via ../
        let traversal = format!("{}/../../../etc/passwd", workspace.path().display());
        let result = sandbox.validate(&traversal, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_sandbox_validate_write_new_file_in_workspace() {
        let (workspace, sandbox) = setup_sandbox();
        let new_file = workspace.path().join("new_file.md");

        // File doesn't exist yet — must_exist=false
        let result = sandbox.validate(new_file.to_str().unwrap(), false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_validate_write_relative_path_in_workspace() {
        let (workspace, sandbox) = setup_sandbox();

        let result = sandbox.validate("relative_new.md", false);
        assert!(result.is_ok());
        let resolved = result.unwrap();
        let workspace_canonical =
            fs::canonicalize(workspace.path()).unwrap_or_else(|_| workspace.path().to_path_buf());
        assert!(resolved.starts_with(&workspace_canonical));
    }

    #[test]
    fn test_sandbox_validate_write_new_file_outside_workspace() {
        let (_workspace, sandbox) = setup_sandbox();
        let result = sandbox.validate("/tmp/evil/backdoor.sh", false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Access denied"));
    }

    #[test]
    fn test_sandbox_validate_write_nested_new_dir() {
        let (workspace, sandbox) = setup_sandbox();
        let nested = workspace.path().join("a/b/c/new.md");

        let result = sandbox.validate(nested.to_str().unwrap(), false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_extra_allowed_path() {
        let extra_dir = TempDir::new().unwrap();
        let file = extra_dir.path().join("allowed.txt");
        fs::write(&file, "ok").unwrap();

        let (_workspace, sandbox) =
            setup_sandbox_with_extra(&[extra_dir.path().to_str().unwrap().to_string()]);

        let result = sandbox.validate(file.to_str().unwrap(), true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_nonexistent_extra_path_is_skipped() {
        let (_workspace, sandbox) =
            setup_sandbox_with_extra(&["/nonexistent/path/12345".to_string()]);
        // Should only have the workspace root
        assert_eq!(sandbox.allowed_roots.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn test_sandbox_validate_denies_symlink_escape() {
        let (workspace, sandbox) = setup_sandbox();
        let target = TempDir::new().unwrap();
        let secret = target.path().join("secret.txt");
        fs::write(&secret, "sensitive data").unwrap();

        // Create symlink inside workspace pointing outside
        let link = workspace.path().join("escape");
        std::os::unix::fs::symlink(target.path(), &link).unwrap();

        let link_target = link.join("secret.txt");
        let result = sandbox.validate(link_target.to_str().unwrap(), true);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Access denied"));
    }

    #[test]
    fn test_sandbox_validate_write_denies_traversal() {
        let (workspace, sandbox) = setup_sandbox();
        // Test the must_exist=false (write) path with `..` components,
        // exercising the canonicalize_ancestor codepath
        let traversal = format!("{}/../../etc/passwd", workspace.path().display());
        let result = sandbox.validate(&traversal, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_get_resolve_path_allows_memory_files() {
        let workspace = TempDir::new().unwrap();
        // Create the files so canonicalize succeeds
        fs::write(workspace.path().join("MEMORY.md"), "mem").unwrap();
        fs::write(workspace.path().join("HEARTBEAT.md"), "hb").unwrap();
        let mem_dir = workspace.path().join("memory");
        fs::create_dir_all(&mem_dir).unwrap();
        fs::write(mem_dir.join("2024-01-01.md"), "log").unwrap();

        let tool = MemoryGetTool::new(workspace.path().to_path_buf());

        assert!(tool.resolve_path("MEMORY.md").is_ok());
        assert!(tool.resolve_path("HEARTBEAT.md").is_ok());
        assert!(tool.resolve_path("memory/2024-01-01.md").is_ok());
    }

    #[test]
    fn test_memory_get_resolve_path_denies_traversal_via_memory_prefix() {
        let workspace = TempDir::new().unwrap();
        // Create memory dir so the prefix check passes
        let mem_dir = workspace.path().join("memory");
        fs::create_dir_all(&mem_dir).unwrap();

        let tool = MemoryGetTool::new(workspace.path().to_path_buf());

        // This passes the starts_with("memory/") check but traverses out
        let result = tool.resolve_path("memory/../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_get_resolve_path_denies_arbitrary_paths() {
        let workspace = TempDir::new().unwrap();
        let tool = MemoryGetTool::new(workspace.path().to_path_buf());

        let result = tool.resolve_path("/etc/passwd");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Access denied"));
    }

    #[test]
    fn test_memory_get_resolve_path_denies_home_dir_escape() {
        let workspace = TempDir::new().unwrap();
        let tool = MemoryGetTool::new(workspace.path().to_path_buf());

        let result = tool.resolve_path("~/.ssh/id_rsa");
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_get_resolve_path_denies_tilde_path() {
        let workspace = TempDir::new().unwrap();
        let tool = MemoryGetTool::new(workspace.path().to_path_buf());

        let result = tool.resolve_path("~/../../etc/passwd");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_read_file_tool_denies_outside_workspace() {
        let (_workspace, sandbox) = setup_sandbox();
        let tool = ReadFileTool::new(Arc::new(sandbox));

        let args = serde_json::json!({"path": "/etc/passwd"}).to_string();
        let result = tool.execute(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Access denied"));
    }

    #[tokio::test]
    async fn test_read_file_tool_allows_workspace_file() {
        let (workspace, sandbox) = setup_sandbox();
        let file = workspace.path().join("hello.txt");
        fs::write(&file, "world").unwrap();

        let tool = ReadFileTool::new(Arc::new(sandbox));
        let args = serde_json::json!({"path": file.to_str().unwrap()}).to_string();
        let result = tool.execute(&args).await;
        assert!(result.is_ok());
        assert!(result.unwrap().contains("world"));
    }

    #[tokio::test]
    async fn test_write_file_tool_denies_outside_workspace() {
        let (_workspace, sandbox) = setup_sandbox();
        let tool = WriteFileTool::new(Arc::new(sandbox));

        let args =
            serde_json::json!({"path": "/tmp/evil_write_test.txt", "content": "pwned"}).to_string();
        let result = tool.execute(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Access denied"));
    }

    #[tokio::test]
    async fn test_edit_file_tool_denies_outside_workspace() {
        let (_workspace, sandbox) = setup_sandbox();
        let tool = EditFileTool::new(Arc::new(sandbox));

        let args = serde_json::json!({
            "path": "/etc/hosts",
            "old_string": "localhost",
            "new_string": "evil"
        })
        .to_string();
        let result = tool.execute(&args).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Access denied"));
    }

    #[test]
    fn test_create_heartbeat_tools_uses_default_allowlist() {
        let config = Config::default();
        let tools = create_heartbeat_tools(&config, None).unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();

        // Default allowlist: memory_search, memory_get, read_file (read-only)
        assert!(names.contains(&"memory_search"), "missing memory_search");
        assert!(names.contains(&"memory_get"), "missing memory_get");
        assert!(names.contains(&"read_file"), "missing read_file");

        // bash, write_file, edit_file, and web_fetch must NOT be present
        assert!(!names.contains(&"bash"), "bash should be excluded");
        assert!(
            !names.contains(&"write_file"),
            "write_file should be excluded"
        );
        assert!(
            !names.contains(&"edit_file"),
            "edit_file should be excluded by default"
        );
        assert!(
            !names.contains(&"web_fetch"),
            "web_fetch should be excluded"
        );
    }

    #[test]
    fn test_create_heartbeat_tools_custom_allowlist() {
        let toml_str = r#"
            [heartbeat]
            allowed_tools = ["memory_search"]
        "#;
        let config: Config = toml::from_str(toml_str).unwrap();
        let tools = create_heartbeat_tools(&config, None).unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();

        assert_eq!(names, vec!["memory_search"]);
    }

    #[test]
    fn test_create_heartbeat_tools_empty_allowlist() {
        let toml_str = r#"
            [heartbeat]
            allowed_tools = []
        "#;
        let config: Config = toml::from_str(toml_str).unwrap();
        let tools = create_heartbeat_tools(&config, None).unwrap();

        assert!(tools.is_empty(), "empty allowlist should produce no tools");
    }
}

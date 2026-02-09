mod embeddings;
mod index;
mod search;
mod watcher;
mod workspace;

#[cfg(feature = "gguf")]
pub use embeddings::LlamaCppProvider;
pub use embeddings::{hash_text, EmbeddingProvider, FastEmbedProvider, OpenAIEmbeddingProvider};
pub use index::{MemoryIndex, ReindexStats};
pub use search::MemoryChunk;
pub use watcher::MemoryWatcher;
pub use workspace::{init_state_dir, init_workspace};

use anyhow::Result;
use chrono::Local;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Handle;
use tracing::{debug, info, warn};

use crate::config::{Config, MemoryConfig};

#[derive(Clone)]
pub struct MemoryManager {
    workspace: PathBuf,
    db_path: PathBuf,
    index: MemoryIndex,
    config: MemoryConfig,
    /// Optional embedding provider for semantic search
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// True if this was a brand new workspace (first run)
    is_brand_new: bool,
}

#[derive(Debug)]
pub struct MemoryStats {
    pub workspace: String,
    pub total_files: usize,
    pub total_chunks: usize,
    pub index_size_kb: u64,
    pub files: Vec<FileStats>,
}

#[derive(Debug)]
pub struct FileStats {
    pub name: String,
    pub chunks: usize,
    pub lines: usize,
}

#[derive(Debug)]
pub struct RecentEntry {
    pub timestamp: String,
    pub file: String,
    pub preview: String,
}

impl MemoryManager {
    /// Create a new MemoryManager with the default agent ID ("main")
    pub fn new(config: &MemoryConfig) -> Result<Self> {
        Self::new_with_agent(config, "main")
    }

    /// Create a new MemoryManager for a specific agent ID (OpenClaw-compatible)
    pub fn new_with_agent(config: &MemoryConfig, agent_id: &str) -> Result<Self> {
        Self::new_with_full_config(config, None, agent_id)
    }

    /// Create a new MemoryManager with full config (for OpenAI embedding provider)
    pub fn new_with_full_config(
        memory_config: &MemoryConfig,
        app_config: Option<&Config>,
        agent_id: &str,
    ) -> Result<Self> {
        let workspace = shellexpand::tilde(&memory_config.workspace).to_string();
        let workspace = PathBuf::from(workspace);

        // Initialize workspace with templates if needed, returns true if brand new
        let is_brand_new = init_workspace(&workspace)?;

        // Database goes in state_dir/memory/{agentId}.sqlite (OpenClaw-compatible)
        let state_dir = workspace
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Workspace has no parent directory"))?;
        let memory_dir = state_dir.join("memory");
        std::fs::create_dir_all(&memory_dir)?;
        let db_path = memory_dir.join(format!("{}.sqlite", agent_id));

        let index = MemoryIndex::new_with_db_path(&workspace, &db_path)?
            .with_chunk_config(memory_config.chunk_size, memory_config.chunk_overlap);

        // Create embedding provider based on config
        let embedding_provider: Option<Arc<dyn EmbeddingProvider>> = match memory_config
            .embedding_provider
            .as_str()
        {
            "local" => {
                let model_name = if memory_config.embedding_model.is_empty()
                    || memory_config.embedding_model == "text-embedding-3-small"
                {
                    None // Use default local model
                } else {
                    Some(memory_config.embedding_model.as_str())
                };
                let cache_dir = if memory_config.embedding_cache_dir.is_empty() {
                    None
                } else {
                    Some(memory_config.embedding_cache_dir.as_str())
                };
                match FastEmbedProvider::new_with_cache_dir(model_name, cache_dir) {
                    Ok(provider) => {
                        info!("Using local embedding provider: {}", provider.model());
                        Some(Arc::new(provider))
                    }
                    Err(e) => {
                        warn!("Failed to initialize local embeddings: {}. Falling back to FTS-only search.", e);
                        None
                    }
                }
            }
            "openai" => {
                // Need OpenAI config for API key
                if let Some(config) = app_config {
                    if let Some(ref openai) = config.providers.openai {
                        match OpenAIEmbeddingProvider::new(
                            &openai.api_key,
                            &openai.base_url,
                            &memory_config.embedding_model,
                        ) {
                            Ok(provider) => {
                                info!("Using OpenAI embedding provider: {}", provider.model());
                                Some(Arc::new(provider))
                            }
                            Err(e) => {
                                warn!("Failed to initialize OpenAI embeddings: {}. Falling back to FTS-only search.", e);
                                None
                            }
                        }
                    } else {
                        warn!("OpenAI embedding provider requested but no OpenAI config found. Falling back to FTS-only search.");
                        None
                    }
                } else {
                    warn!("OpenAI embedding provider requested but no app config provided. Falling back to FTS-only search.");
                    None
                }
            }
            #[cfg(feature = "gguf")]
            "gguf" => {
                let cache_dir = if memory_config.embedding_cache_dir.is_empty() {
                    None
                } else {
                    Some(memory_config.embedding_cache_dir.as_str())
                };
                match LlamaCppProvider::new(&memory_config.embedding_model, cache_dir) {
                    Ok(provider) => {
                        info!("Using GGUF embedding provider: {}", provider.model());
                        Some(Arc::new(provider))
                    }
                    Err(e) => {
                        warn!("Failed to initialize GGUF embeddings: {}. Falling back to FTS-only search.", e);
                        None
                    }
                }
            }
            #[cfg(not(feature = "gguf"))]
            "gguf" => {
                warn!("GGUF embedding provider requested but 'gguf' feature is not enabled. Build with --features gguf. Falling back to FTS-only search.");
                None
            }
            "none" => {
                debug!("Embeddings disabled, using FTS-only search");
                None
            }
            other => {
                warn!(
                    "Unknown embedding provider '{}'. Falling back to FTS-only search.",
                    other
                );
                None
            }
        };

        Ok(Self {
            workspace,
            db_path,
            index,
            config: memory_config.clone(),
            embedding_provider,
            is_brand_new,
        })
    }

    /// Create a minimal MemoryManager for testing (uses temp directory, no embeddings).
    #[cfg(test)]
    pub fn new_stub() -> Self {
        let dir = std::env::temp_dir().join(format!("localgpt-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let db_path = dir.join("test.sqlite");
        let index = MemoryIndex::new_with_db_path(&dir, &db_path).expect("create test index");
        Self {
            workspace: dir,
            db_path,
            index,
            config: MemoryConfig::default(),
            embedding_provider: None,
            is_brand_new: true,
        }
    }

    /// Set embedding provider for semantic search (requires OpenAI API key)
    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    /// Check if semantic search is available
    pub fn has_embeddings(&self) -> bool {
        self.embedding_provider.is_some()
    }

    pub fn workspace(&self) -> &PathBuf {
        &self.workspace
    }

    /// Read the main MEMORY.md file
    pub fn read_memory_file(&self) -> Result<String> {
        let path = self.workspace.join("MEMORY.md");
        if path.exists() {
            Ok(fs::read_to_string(&path)?)
        } else {
            Ok(String::new())
        }
    }

    /// Read the HEARTBEAT.md file
    pub fn read_heartbeat_file(&self) -> Result<String> {
        let path = self.workspace.join("HEARTBEAT.md");
        if path.exists() {
            Ok(fs::read_to_string(&path)?)
        } else {
            Ok(String::new())
        }
    }

    /// Read the SOUL.md file (persona/tone guidance)
    pub fn read_soul_file(&self) -> Result<String> {
        let path = self.workspace.join("SOUL.md");
        if path.exists() {
            Ok(fs::read_to_string(&path)?)
        } else {
            Ok(String::new())
        }
    }

    /// Read the USER.md file (OpenClaw-compatible: user info)
    pub fn read_user_file(&self) -> Result<String> {
        let path = self.workspace.join("USER.md");
        if path.exists() {
            Ok(fs::read_to_string(&path)?)
        } else {
            Ok(String::new())
        }
    }

    /// Read the IDENTITY.md file (OpenClaw-compatible: agent identity context)
    pub fn read_identity_file(&self) -> Result<String> {
        let path = self.workspace.join("IDENTITY.md");
        if path.exists() {
            Ok(fs::read_to_string(&path)?)
        } else {
            Ok(String::new())
        }
    }

    /// Read the AGENTS.md file (OpenClaw-compatible: list of agents)
    pub fn read_agents_file(&self) -> Result<String> {
        let path = self.workspace.join("AGENTS.md");
        if path.exists() {
            Ok(fs::read_to_string(&path)?)
        } else {
            Ok(String::new())
        }
    }

    /// Check if this is a brand new workspace (first run)
    pub fn is_brand_new(&self) -> bool {
        self.is_brand_new
    }

    /// Read the TOOLS.md file (OpenClaw-compatible: local tool notes)
    pub fn read_tools_file(&self) -> Result<String> {
        let path = self.workspace.join("TOOLS.md");
        if path.exists() {
            Ok(fs::read_to_string(&path)?)
        } else {
            Ok(String::new())
        }
    }

    /// Read recent daily log files
    pub fn read_recent_daily_logs(&self, days: usize) -> Result<String> {
        let memory_dir = self.workspace.join("memory");
        if !memory_dir.exists() {
            return Ok(String::new());
        }

        let today = Local::now().date_naive();
        let mut content = String::new();

        for i in 0..days {
            let date = today - chrono::Duration::days(i as i64);
            let filename = format!("{}.md", date.format("%Y-%m-%d"));
            let path = memory_dir.join(&filename);

            if path.exists() {
                if let Ok(file_content) = fs::read_to_string(&path) {
                    if !content.is_empty() {
                        content.push_str("\n---\n\n");
                    }
                    content.push_str(&format!("## {}\n\n", filename));
                    content.push_str(&file_content);
                }
            }
        }

        Ok(content)
    }

    /// Search memory using hybrid search (FTS + semantic if available)
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryChunk>> {
        // If we have an embedding provider, try hybrid search
        if let Some(ref provider) = self.embedding_provider {
            // Try to get query embedding (may fail if no API key, rate limited, etc.)
            if let Ok(handle) = Handle::try_current() {
                let provider = provider.clone();
                let query_string = query.to_string();
                let model = provider.model().to_string();

                // Run embedding in blocking context
                let embedding_result = std::thread::spawn(move || {
                    handle.block_on(async { provider.embed(&query_string).await })
                })
                .join()
                .map_err(|_| anyhow::anyhow!("Thread panicked"))?;

                if let Ok(embedding) = embedding_result {
                    debug!("Using hybrid search with {} dimensions", embedding.len());
                    return self.index.search_hybrid(
                        query,
                        Some(&embedding),
                        &model,
                        limit,
                        0.3, // FTS weight
                        0.7, // Vector weight
                    );
                }
            }
        }

        // Fallback to FTS-only search
        self.index.search(query, limit)
    }

    /// Search memory using FTS only (faster, no API calls)
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<MemoryChunk>> {
        self.index.search(query, limit)
    }

    /// Get total chunk count
    pub fn chunk_count(&self) -> Result<usize> {
        self.index.chunk_count()
    }

    /// Reindex all memory files
    pub fn reindex(&self, force: bool) -> Result<ReindexStats> {
        let start = std::time::Instant::now();
        let mut stats = ReindexStats {
            files_processed: 0,
            files_updated: 0,
            chunks_indexed: 0,
            duration: Duration::default(),
        };

        // First, clean up deleted files from the index
        let files_removed = self.cleanup_deleted_files()?;
        if files_removed > 0 {
            info!("Removed {} deleted files from index", files_removed);
        }

        // Index all .md files recursively under workspace
        let pattern = format!("{}/**/*.md", self.workspace.display());
        for entry in glob::glob(&pattern)
            .into_iter()
            .flatten()
            .filter_map(|r| r.ok())
        {
            if entry.is_file() {
                stats.files_processed += 1;
                if self.index.index_file(&entry, force)? {
                    stats.files_updated += 1;
                }
            }
        }

        // Index configured external paths (outside workspace)
        for index_path in &self.config.paths {
            let base_path = if index_path.path.starts_with('~') || index_path.path.starts_with('/')
            {
                PathBuf::from(shellexpand::tilde(&index_path.path).to_string())
            } else {
                self.workspace.join(&index_path.path)
            };

            // Skip paths inside workspace (already covered by recursive glob above)
            if base_path.starts_with(&self.workspace) {
                continue;
            }

            if !base_path.exists() {
                debug!("Skipping non-existent index path: {}", base_path.display());
                continue;
            }

            let pattern = format!("{}/{}", base_path.display(), index_path.pattern);
            debug!("Indexing external path with pattern: {}", pattern);

            for entry in glob::glob(&pattern)
                .into_iter()
                .flatten()
                .filter_map(|r| r.ok())
            {
                if entry.is_file() {
                    stats.files_processed += 1;
                    if self.index.index_file(&entry, force)? {
                        stats.files_updated += 1;
                    }
                }
            }
        }

        stats.chunks_indexed = self.index.chunk_count()?;
        stats.duration = start.elapsed();

        info!("Reindex complete: {:?}", stats);
        Ok(stats)
    }

    /// Remove files from index that no longer exist on disk
    fn cleanup_deleted_files(&self) -> Result<usize> {
        let indexed_files = self.index.indexed_files()?;
        let mut removed = 0;

        for relative_path in indexed_files {
            let full_path = self.workspace.join(&relative_path);
            if !full_path.exists() {
                debug!("Cleaning up deleted file: {}", relative_path);
                self.index.remove_file(&relative_path)?;
                removed += 1;
            }
        }

        Ok(removed)
    }

    /// Get memory statistics
    pub fn stats(&self) -> Result<MemoryStats> {
        let mut files = Vec::new();
        let mut total_chunks = 0;

        // Get stats for all .md files recursively under workspace
        let pattern = format!("{}/**/*.md", self.workspace.display());
        for entry in glob::glob(&pattern)
            .into_iter()
            .flatten()
            .filter_map(|r| r.ok())
        {
            if entry.is_file() {
                let content = fs::read_to_string(&entry)?;
                let lines = content.lines().count();
                let chunks = self.index.file_chunk_count(&entry)?;
                total_chunks += chunks;

                let display_name = entry
                    .strip_prefix(&self.workspace)
                    .map(|rel| rel.display().to_string())
                    .unwrap_or_else(|_| entry.display().to_string());

                files.push(FileStats {
                    name: display_name,
                    chunks,
                    lines,
                });
            }
        }

        // Configured external paths (outside workspace)
        for index_path in &self.config.paths {
            let base_path = if index_path.path.starts_with('~') || index_path.path.starts_with('/')
            {
                PathBuf::from(shellexpand::tilde(&index_path.path).to_string())
            } else {
                self.workspace.join(&index_path.path)
            };

            // Skip paths inside workspace (already covered above)
            if base_path.starts_with(&self.workspace) {
                continue;
            }

            if !base_path.exists() {
                continue;
            }

            let pattern = format!("{}/{}", base_path.display(), index_path.pattern);

            for entry in glob::glob(&pattern)
                .into_iter()
                .flatten()
                .filter_map(|r| r.ok())
            {
                if entry.is_file() {
                    let content = fs::read_to_string(&entry)?;
                    let lines = content.lines().count();
                    let chunks = self.index.file_chunk_count(&entry)?;
                    total_chunks += chunks;

                    let display_name = if let Ok(rel) = entry.strip_prefix(&base_path) {
                        format!("{}/{}", index_path.path, rel.display())
                    } else {
                        entry.display().to_string()
                    };

                    files.push(FileStats {
                        name: display_name,
                        chunks,
                        lines,
                    });
                }
            }
        }

        let index_size = self.index.size_bytes()? / 1024;

        Ok(MemoryStats {
            workspace: self.workspace.display().to_string(),
            total_files: files.len(),
            total_chunks,
            index_size_kb: index_size,
            files,
        })
    }

    /// Get recent memory entries
    pub fn recent_entries(&self, count: usize) -> Result<Vec<RecentEntry>> {
        let mut entries = Vec::new();

        let memory_dir = self.workspace.join("memory");
        if !memory_dir.exists() {
            return Ok(entries);
        }

        // Get all daily log files sorted by date (newest first)
        let mut files: Vec<_> = fs::read_dir(&memory_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|e| e == "md").unwrap_or(false))
            .collect();

        files.sort_by_key(|f| std::cmp::Reverse(f.file_name()));

        for entry in files.into_iter().take(count) {
            let path = entry.path();
            let filename = path.file_name().unwrap().to_string_lossy().to_string();

            if let Ok(content) = fs::read_to_string(&path) {
                // Get last non-empty line as preview
                let preview = content
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("")
                    .chars()
                    .take(100)
                    .collect();

                entries.push(RecentEntry {
                    timestamp: filename.replace(".md", ""),
                    file: format!("memory/{}", filename),
                    preview,
                });
            }
        }

        Ok(entries)
    }

    /// Start file watcher for automatic reindexing
    pub fn start_watcher(&self) -> Result<MemoryWatcher> {
        MemoryWatcher::new(
            self.workspace.clone(),
            self.db_path.clone(),
            self.config.clone(),
        )
    }

    /// Generate embeddings for chunks that don't have them
    /// Returns (chunks_processed, chunks_embedded)
    /// Uses embedding cache to avoid regenerating identical content
    pub async fn generate_embeddings(&self, batch_size: usize) -> Result<(usize, usize)> {
        let provider = match &self.embedding_provider {
            Some(p) => p,
            None => {
                debug!("No embedding provider configured, skipping embedding generation");
                return Ok((0, 0));
            }
        };

        let provider_id = provider.id().to_string();
        let model = provider.model().to_string();
        let mut total_processed = 0;
        let mut total_embedded = 0;
        let mut cache_hits = 0;

        loop {
            // Get chunks without embeddings
            let chunks = self.index.chunks_without_embeddings(batch_size)?;
            if chunks.is_empty() {
                break;
            }

            total_processed += chunks.len();

            // Separate chunks into cached and uncached
            let mut to_embed: Vec<(String, String, String)> = Vec::new(); // (id, text, hash)
            let mut from_cache: Vec<(String, Vec<f32>)> = Vec::new(); // (id, embedding)

            for (chunk_id, text) in &chunks {
                let text_hash = hash_text(text);

                // Check cache first
                if let Ok(Some(cached)) =
                    self.index
                        .get_cached_embedding(&provider_id, &model, &text_hash)
                {
                    from_cache.push((chunk_id.clone(), cached));
                    cache_hits += 1;
                } else {
                    to_embed.push((chunk_id.clone(), text.clone(), text_hash));
                }
            }

            // Store cached embeddings
            for (chunk_id, embedding) in from_cache {
                if let Err(e) = self.index.store_embedding(&chunk_id, &embedding, &model) {
                    warn!(
                        "Failed to store cached embedding for chunk {}: {}",
                        chunk_id, e
                    );
                } else {
                    total_embedded += 1;
                }
            }

            // Generate new embeddings for uncached chunks
            if !to_embed.is_empty() {
                let texts: Vec<String> = to_embed.iter().map(|(_, text, _)| text.clone()).collect();

                match provider.embed_batch(&texts).await {
                    Ok(embeddings) => {
                        for ((chunk_id, _text, text_hash), embedding) in
                            to_embed.iter().zip(embeddings.iter())
                        {
                            // Store in chunk
                            if let Err(e) = self.index.store_embedding(chunk_id, embedding, &model)
                            {
                                warn!("Failed to store embedding for chunk {}: {}", chunk_id, e);
                            } else {
                                total_embedded += 1;
                            }

                            // Store in cache for future reuse
                            if let Err(e) = self.index.cache_embedding(
                                &provider_id,
                                &model,
                                "", // provider_key (API key identifier, can be empty)
                                text_hash,
                                embedding,
                            ) {
                                debug!("Failed to cache embedding: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to generate embeddings: {}", e);
                        break;
                    }
                }
            }

            debug!(
                "Generated embeddings: {}/{} chunks ({} from cache)",
                total_embedded, total_processed, cache_hits
            );

            // Break if we processed fewer than batch_size (last batch)
            if chunks.len() < batch_size {
                break;
            }
        }

        info!(
            "Embedding generation complete: {} chunks, {} embedded, {} cache hits",
            total_processed, total_embedded, cache_hits
        );

        Ok((total_processed, total_embedded))
    }

    /// Get count of chunks with embeddings
    pub fn embedded_chunk_count(&self) -> Result<usize> {
        let model = self
            .embedding_provider
            .as_ref()
            .map(|p| p.model().to_string())
            .unwrap_or_default();
        self.index.embedded_chunk_count(&model)
    }
}

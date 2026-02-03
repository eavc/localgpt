mod index;
mod search;
mod watcher;
mod workspace;

pub use index::{MemoryIndex, ReindexStats};
pub use search::MemoryChunk;
pub use watcher::MemoryWatcher;
pub use workspace::{init_state_dir, init_workspace};

use anyhow::Result;
use chrono::Local;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tracing::info;

use crate::config::MemoryConfig;

#[derive(Clone)]
pub struct MemoryManager {
    workspace: PathBuf,
    index: MemoryIndex,
    config: MemoryConfig,
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
    pub fn new(config: &MemoryConfig) -> Result<Self> {
        let workspace = shellexpand::tilde(&config.workspace).to_string();
        let workspace = PathBuf::from(workspace);

        // Initialize workspace with templates if needed
        init_workspace(&workspace)?;

        let index = MemoryIndex::new(&workspace)?;

        Ok(Self {
            workspace,
            index,
            config: config.clone(),
        })
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

    /// Search memory using FTS
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryChunk>> {
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

        // Index MEMORY.md
        let memory_file = self.workspace.join("MEMORY.md");
        if memory_file.exists() {
            stats.files_processed += 1;
            if self.index.index_file(&memory_file, force)? {
                stats.files_updated += 1;
            }
        }

        // Index HEARTBEAT.md
        let heartbeat_file = self.workspace.join("HEARTBEAT.md");
        if heartbeat_file.exists() {
            stats.files_processed += 1;
            if self.index.index_file(&heartbeat_file, force)? {
                stats.files_updated += 1;
            }
        }

        // Index daily logs
        let memory_dir = self.workspace.join("memory");
        if memory_dir.exists() {
            for entry in fs::read_dir(&memory_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().map(|e| e == "md").unwrap_or(false) {
                    stats.files_processed += 1;
                    if self.index.index_file(&path, force)? {
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

    /// Get memory statistics
    pub fn stats(&self) -> Result<MemoryStats> {
        let mut files = Vec::new();
        let mut total_chunks = 0;

        // Get stats for each file
        let memory_file = self.workspace.join("MEMORY.md");
        if memory_file.exists() {
            let content = fs::read_to_string(&memory_file)?;
            let lines = content.lines().count();
            let chunks = self.index.file_chunk_count(&memory_file)?;
            total_chunks += chunks;
            files.push(FileStats {
                name: "MEMORY.md".to_string(),
                chunks,
                lines,
            });
        }

        let heartbeat_file = self.workspace.join("HEARTBEAT.md");
        if heartbeat_file.exists() {
            let content = fs::read_to_string(&heartbeat_file)?;
            let lines = content.lines().count();
            let chunks = self.index.file_chunk_count(&heartbeat_file)?;
            total_chunks += chunks;
            files.push(FileStats {
                name: "HEARTBEAT.md".to_string(),
                chunks,
                lines,
            });
        }

        // Daily logs
        let memory_dir = self.workspace.join("memory");
        if memory_dir.exists() {
            for entry in fs::read_dir(&memory_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().map(|e| e == "md").unwrap_or(false) {
                    let content = fs::read_to_string(&path)?;
                    let lines = content.lines().count();
                    let chunks = self.index.file_chunk_count(&path)?;
                    total_chunks += chunks;
                    files.push(FileStats {
                        name: format!("memory/{}", path.file_name().unwrap().to_string_lossy()),
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

        files.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

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
        MemoryWatcher::new(self.workspace.clone(), self.config.clone())
    }
}

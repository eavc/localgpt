//! Skills system for LocalGPT (OpenClaw-compatible)
//!
//! Skills are SKILL.md files that provide specialized instructions for specific tasks.
//! Supports multiple sources, requirements gating, and slash command invocation.

use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, warn};

/// Skill requirements for eligibility gating
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SkillRequirements {
    /// Required binaries (all must exist on PATH)
    #[serde(default)]
    pub bins: Vec<String>,

    /// At least one of these binaries must exist
    #[serde(default, rename = "anyBins")]
    pub any_bins: Vec<String>,

    /// Required environment variables
    #[serde(default)]
    pub env: Vec<String>,
}

/// OpenClaw metadata in frontmatter
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SkillMetadata {
    /// Emoji for display
    pub emoji: Option<String>,

    /// Skip eligibility checks if true
    #[serde(default)]
    pub always: bool,

    /// Requirements for this skill
    #[serde(default)]
    pub requires: SkillRequirements,
}

/// Frontmatter parsed from SKILL.md
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SkillFrontmatter {
    /// Skill name (overrides directory name)
    pub name: Option<String>,

    /// Brief description
    pub description: Option<String>,

    /// Whether this skill can be invoked via slash command (default: true)
    #[serde(default = "default_true", rename = "user-invocable")]
    pub user_invocable: bool,

    /// Whether to exclude from model's system prompt (default: false)
    #[serde(default, rename = "disable-model-invocation")]
    pub disable_model_invocation: bool,

    /// Dispatch slash command directly to a tool
    #[serde(rename = "command-dispatch")]
    pub command_dispatch: Option<String>,

    /// Tool name for command dispatch
    #[serde(rename = "command-tool")]
    pub command_tool: Option<String>,

    /// OpenClaw-specific metadata
    #[serde(default)]
    pub metadata: Option<SkillMetadataWrapper>,
}

/// Wrapper for nested metadata (handles both flat and nested openclaw key)
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SkillMetadataWrapper {
    #[serde(default)]
    pub openclaw: Option<SkillMetadata>,

    // Flat fields (fallback)
    pub emoji: Option<String>,
    #[serde(default)]
    pub requires: Option<SkillRequirements>,
}

fn default_true() -> bool {
    true
}

/// Source of a skill
#[derive(Debug, Clone, PartialEq)]
pub enum SkillSource {
    /// Workspace-level skill (highest priority)
    Workspace,
    /// User-managed skill (~/.localgpt/skills/)
    Managed,
    /// Bundled with the application
    Bundled,
}

/// Eligibility status for a skill
#[derive(Debug, Clone)]
pub enum SkillEligibility {
    /// Skill is ready to use
    Ready,
    /// Missing required binaries
    MissingBins(Vec<String>),
    /// Missing required environment variables
    MissingEnv(Vec<String>),
    /// Missing at least one of anyBins
    MissingAnyBins(Vec<String>),
}

impl SkillEligibility {
    pub fn is_ready(&self) -> bool {
        matches!(self, SkillEligibility::Ready)
    }
}

/// A skill loaded from SKILL.md
#[derive(Debug, Clone)]
pub struct Skill {
    /// Skill name (from frontmatter or directory name)
    pub name: String,

    /// Sanitized command name for slash commands
    pub command_name: String,

    /// Path to SKILL.md
    pub path: PathBuf,

    /// Brief description
    pub description: String,

    /// Emoji for display
    pub emoji: Option<String>,

    /// Source of the skill
    pub source: SkillSource,

    /// Whether this skill can be invoked via slash command
    pub user_invocable: bool,

    /// Whether to exclude from model's system prompt
    pub disable_model_invocation: bool,

    /// Direct tool dispatch configuration
    pub command_dispatch: Option<CommandDispatch>,

    /// Requirements for eligibility
    pub requires: SkillRequirements,

    /// Current eligibility status
    pub eligibility: SkillEligibility,
}

/// Command dispatch configuration for direct tool execution
#[derive(Debug, Clone)]
pub struct CommandDispatch {
    /// Dispatch type (currently only "tool")
    pub kind: String,
    /// Tool name to dispatch to
    pub tool_name: String,
}

impl Skill {
    /// Check if this skill should be included in the model's system prompt
    pub fn include_in_prompt(&self) -> bool {
        !self.disable_model_invocation && self.eligibility.is_ready()
    }

    /// Check if this skill can be invoked via slash command
    pub fn can_invoke(&self) -> bool {
        self.user_invocable && self.eligibility.is_ready()
    }
}

/// Load all skills from multiple sources
/// Returns skills sorted by name with workspace skills taking priority over managed
pub fn load_skills(workspace: &Path) -> Result<Vec<Skill>> {
    let mut skills_map: HashMap<String, Skill> = HashMap::new();

    // Load from managed directory first (lower priority)
    if let Some(managed_dir) = get_managed_skills_dir() {
        if managed_dir.exists() {
            for skill in load_skills_from_dir(&managed_dir, SkillSource::Managed)? {
                skills_map.insert(skill.name.clone(), skill);
            }
        }
    }

    // Load from workspace (higher priority, overwrites managed)
    let workspace_skills_dir = workspace.join("skills");
    if workspace_skills_dir.exists() {
        for skill in load_skills_from_dir(&workspace_skills_dir, SkillSource::Workspace)? {
            skills_map.insert(skill.name.clone(), skill);
        }
    }

    // Convert to vec and sort
    let mut skills: Vec<Skill> = skills_map.into_values().collect();
    skills.sort_by(|a, b| a.name.cmp(&b.name));

    debug!("Loaded {} skills", skills.len());
    Ok(skills)
}

/// Get the managed skills directory (~/.localgpt/skills/)
fn get_managed_skills_dir() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.home_dir().join(".localgpt").join("skills"))
}

/// Load skills from a single directory
fn load_skills_from_dir(dir: &Path, source: SkillSource) -> Result<Vec<Skill>> {
    let mut skills = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let skill_file = path.join("SKILL.md");
        if !skill_file.exists() {
            continue;
        }

        let dir_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        match load_skill(&skill_file, &dir_name, source.clone()) {
            Ok(skill) => skills.push(skill),
            Err(e) => {
                warn!("Failed to load skill from {:?}: {}", skill_file, e);
            }
        }
    }

    Ok(skills)
}

/// Load a single skill from a SKILL.md file
fn load_skill(path: &Path, dir_name: &str, source: SkillSource) -> Result<Skill> {
    let content = fs::read_to_string(path)?;
    let (frontmatter, body) = parse_frontmatter(&content);

    // Get name from frontmatter or directory
    let name = frontmatter
        .name
        .clone()
        .unwrap_or_else(|| dir_name.to_string());

    // Generate sanitized command name
    let command_name = sanitize_command_name(&name);

    // Get description from frontmatter or body
    let description = frontmatter
        .description
        .clone()
        .unwrap_or_else(|| extract_description_from_body(&body));

    // Extract metadata
    let (emoji, requires, always) = if let Some(ref meta) = frontmatter.metadata {
        if let Some(ref oc) = meta.openclaw {
            (oc.emoji.clone(), oc.requires.clone(), oc.always)
        } else {
            (
                meta.emoji.clone(),
                meta.requires.clone().unwrap_or_default(),
                false,
            )
        }
    } else {
        (None, SkillRequirements::default(), false)
    };

    // Check eligibility (skip if always=true)
    let eligibility = if always {
        SkillEligibility::Ready
    } else {
        check_eligibility(&requires)
    };

    // Parse command dispatch
    let command_dispatch = if frontmatter.command_dispatch.as_deref() == Some("tool") {
        frontmatter.command_tool.map(|tool_name| CommandDispatch {
            kind: "tool".to_string(),
            tool_name,
        })
    } else {
        None
    };

    Ok(Skill {
        name,
        command_name,
        path: path.to_path_buf(),
        description,
        emoji,
        source,
        user_invocable: frontmatter.user_invocable,
        disable_model_invocation: frontmatter.disable_model_invocation,
        command_dispatch,
        requires,
        eligibility,
    })
}

/// Parse YAML frontmatter from content
fn parse_frontmatter(content: &str) -> (SkillFrontmatter, String) {
    let lines: Vec<&str> = content.lines().collect();

    // Check for frontmatter
    if lines.first().map(|l| l.trim()) != Some("---") {
        return (SkillFrontmatter::default(), content.to_string());
    }

    // Find closing ---
    let end_idx = lines
        .iter()
        .skip(1)
        .position(|l| l.trim() == "---")
        .map(|i| i + 1);

    let Some(end_idx) = end_idx else {
        return (SkillFrontmatter::default(), content.to_string());
    };

    // Extract frontmatter YAML
    let yaml_content: String = lines[1..end_idx].join("\n");
    let body: String = lines[end_idx + 1..].join("\n");

    // Parse YAML
    match serde_yaml::from_str(&yaml_content) {
        Ok(fm) => (fm, body),
        Err(e) => {
            debug!("Failed to parse frontmatter: {}", e);
            (SkillFrontmatter::default(), content.to_string())
        }
    }
}

/// Extract description from markdown body
fn extract_description_from_body(body: &str) -> String {
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        return trimmed.chars().take(100).collect();
    }
    String::new()
}

/// Sanitize skill name to command name (lowercase, special chars to hyphens)
fn sanitize_command_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
        .chars()
        .take(32)
        .collect()
}

/// Check if a skill meets its requirements
fn check_eligibility(requires: &SkillRequirements) -> SkillEligibility {
    // Check required binaries
    let missing_bins: Vec<String> = requires
        .bins
        .iter()
        .filter(|bin| !has_binary(bin))
        .cloned()
        .collect();

    if !missing_bins.is_empty() {
        return SkillEligibility::MissingBins(missing_bins);
    }

    // Check anyBins (at least one must exist)
    if !requires.any_bins.is_empty() {
        let has_any = requires.any_bins.iter().any(|bin| has_binary(bin));
        if !has_any {
            return SkillEligibility::MissingAnyBins(requires.any_bins.clone());
        }
    }

    // Check environment variables
    let missing_env: Vec<String> = requires
        .env
        .iter()
        .filter(|var| env::var(var).is_err())
        .cloned()
        .collect();

    if !missing_env.is_empty() {
        return SkillEligibility::MissingEnv(missing_env);
    }

    SkillEligibility::Ready
}

/// Check if a binary exists on PATH
fn has_binary(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Result of parsing a slash command
#[derive(Debug, Clone)]
pub struct SkillInvocation {
    /// The skill being invoked
    pub skill_name: String,
    /// Arguments passed to the skill
    pub args: String,
    /// Direct dispatch configuration (if skill has it)
    pub dispatch: Option<CommandDispatch>,
}

/// Parse a slash command and find matching skill
/// Returns None if not a skill command or skill not found
pub fn parse_skill_command(input: &str, skills: &[Skill]) -> Option<SkillInvocation> {
    let input = input.trim();

    // Must start with /
    if !input.starts_with('/') {
        return None;
    }

    // Extract command and args
    let without_slash = &input[1..];
    let (cmd, args) = match without_slash.split_once(char::is_whitespace) {
        Some((c, a)) => (c.trim(), a.trim().to_string()),
        None => (without_slash.trim(), String::new()),
    };

    // Normalize command (lowercase, hyphens)
    let normalized_cmd = cmd.to_lowercase().replace('_', "-");

    // Find matching skill
    for skill in skills {
        if !skill.can_invoke() {
            continue;
        }

        // Match by command_name or name
        let skill_cmd = skill.command_name.replace('_', "-");
        let skill_name_normalized = skill.name.to_lowercase().replace('_', "-");

        if normalized_cmd == skill_cmd || normalized_cmd == skill_name_normalized {
            return Some(SkillInvocation {
                skill_name: skill.name.clone(),
                args,
                dispatch: skill.command_dispatch.clone(),
            });
        }
    }

    None
}

/// Build skills prompt section for the system prompt
pub fn build_skills_prompt(skills: &[Skill]) -> String {
    // Filter to skills that should be in the prompt
    let prompt_skills: Vec<&Skill> = skills.iter().filter(|s| s.include_in_prompt()).collect();

    if prompt_skills.is_empty() {
        return String::new();
    }

    let mut lines = vec![
        "## Skills".to_string(),
        String::new(),
        "Before replying: scan available skills below. If one clearly applies, \
         read its SKILL.md with read_file, then follow it."
            .to_string(),
        String::new(),
        "<available_skills>".to_string(),
    ];

    for skill in &prompt_skills {
        let emoji_prefix = skill
            .emoji
            .as_ref()
            .map(|e| format!("{} ", e))
            .unwrap_or_default();

        let command_info = if skill.user_invocable {
            format!(" (or use /{} command)", skill.command_name)
        } else {
            String::new()
        };

        lines.push(format!(
            "- {}{}: {}{}",
            emoji_prefix, skill.name, skill.description, command_info
        ));
        lines.push(format!("  location: {}", skill.path.display()));
    }

    lines.push("</available_skills>".to_string());
    lines.push(String::new());

    // List user-invocable skills
    let invocable: Vec<&Skill> = skills.iter().filter(|s| s.can_invoke()).collect();
    if !invocable.is_empty() {
        lines.push("Available slash commands:".to_string());
        for skill in &invocable {
            let emoji = skill
                .emoji
                .as_ref()
                .map(|e| format!(" {}", e))
                .unwrap_or_default();
            lines.push(format!(
                "- /{}{} - {}",
                skill.command_name, emoji, skill.description
            ));
        }
        lines.push(String::new());
    }

    lines.push("Rules:".to_string());
    lines.push(
        "- If exactly one skill clearly applies: read its SKILL.md, then follow it.".to_string(),
    );
    lines.push("- If multiple could apply: choose the most specific one.".to_string());
    lines.push("- If none clearly apply: do not read any SKILL.md.".to_string());
    lines.push(String::new());

    lines.join("\n")
}

/// Get skill status summary for CLI display
pub fn get_skills_summary(skills: &[Skill]) -> String {
    let ready: Vec<&Skill> = skills.iter().filter(|s| s.eligibility.is_ready()).collect();
    let blocked: Vec<&Skill> = skills
        .iter()
        .filter(|s| !s.eligibility.is_ready())
        .collect();

    let mut lines = Vec::new();
    lines.push(format!(
        "Skills: {} ready, {} blocked",
        ready.len(),
        blocked.len()
    ));

    if !ready.is_empty() {
        lines.push(String::new());
        lines.push("Ready:".to_string());
        for skill in &ready {
            let emoji = skill
                .emoji
                .as_ref()
                .map(|e| format!(" {}", e))
                .unwrap_or_default();
            let source = match skill.source {
                SkillSource::Workspace => "[workspace]",
                SkillSource::Managed => "[managed]",
                SkillSource::Bundled => "[bundled]",
            };
            lines.push(format!(
                "  /{}{} - {} {}",
                skill.command_name, emoji, skill.description, source
            ));
        }
    }

    if !blocked.is_empty() {
        lines.push(String::new());
        lines.push("Blocked:".to_string());
        for skill in &blocked {
            let reason = match &skill.eligibility {
                SkillEligibility::Ready => "ready".to_string(),
                SkillEligibility::MissingBins(bins) => format!("missing bins: {}", bins.join(", ")),
                SkillEligibility::MissingEnv(vars) => format!("missing env: {}", vars.join(", ")),
                SkillEligibility::MissingAnyBins(bins) => {
                    format!("need one of: {}", bins.join(", "))
                }
            };
            lines.push(format!("  {} - {}", skill.name, reason));
        }
    }

    lines.join("\n")
}

/// Extract description from skill content (used by tests)
#[allow(dead_code)]
fn extract_description(content: &str) -> String {
    let (fm, body) = parse_frontmatter(content);
    fm.description
        .unwrap_or_else(|| extract_description_from_body(&body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_frontmatter() {
        let content = r#"---
name: test-skill
description: "A test skill"
user-invocable: true
disable-model-invocation: false
---
# Test Skill

This is the body.
"#;
        let (fm, body) = parse_frontmatter(content);
        assert_eq!(fm.name, Some("test-skill".to_string()));
        assert_eq!(fm.description, Some("A test skill".to_string()));
        assert!(fm.user_invocable);
        assert!(!fm.disable_model_invocation);
        assert!(body.contains("This is the body"));
    }

    #[test]
    fn test_parse_frontmatter_with_metadata() {
        let content = r#"---
name: github-pr
description: "Create GitHub PRs"
metadata:
  openclaw:
    emoji: "🐙"
    requires:
      bins: ["gh", "git"]
      env: ["GITHUB_TOKEN"]
---
"#;
        let (fm, _) = parse_frontmatter(content);
        assert_eq!(fm.name, Some("github-pr".to_string()));

        let meta = fm.metadata.unwrap();
        let oc = meta.openclaw.unwrap();
        assert_eq!(oc.emoji, Some("🐙".to_string()));
        assert_eq!(oc.requires.bins, vec!["gh", "git"]);
        assert_eq!(oc.requires.env, vec!["GITHUB_TOKEN"]);
    }

    #[test]
    fn test_sanitize_command_name() {
        assert_eq!(sanitize_command_name("GitHub PR"), "github-pr");
        assert_eq!(sanitize_command_name("test_skill"), "test-skill");
        assert_eq!(sanitize_command_name("My Cool Skill!"), "my-cool-skill");
    }

    #[test]
    fn test_extract_description() {
        let content = r#"---
name: test
---
# Test Skill

This is a test skill that does something useful.
"#;
        let desc = extract_description(content);
        assert_eq!(desc, "This is a test skill that does something useful.");
    }

    #[test]
    fn test_extract_description_no_frontmatter() {
        let content = r#"# My Skill

A skill for doing things.
"#;
        let desc = extract_description(content);
        assert_eq!(desc, "A skill for doing things.");
    }

    #[test]
    fn test_parse_skill_command() {
        let skills = vec![Skill {
            name: "github-pr".to_string(),
            command_name: "github-pr".to_string(),
            path: PathBuf::from("/test/SKILL.md"),
            description: "Create PRs".to_string(),
            emoji: Some("🐙".to_string()),
            source: SkillSource::Workspace,
            user_invocable: true,
            disable_model_invocation: false,
            command_dispatch: None,
            requires: SkillRequirements::default(),
            eligibility: SkillEligibility::Ready,
        }];

        // Match by command name
        let result = parse_skill_command("/github-pr create feature", &skills);
        assert!(result.is_some());
        let inv = result.unwrap();
        assert_eq!(inv.skill_name, "github-pr");
        assert_eq!(inv.args, "create feature");

        // No match
        let result = parse_skill_command("/unknown-skill", &skills);
        assert!(result.is_none());

        // Not a command
        let result = parse_skill_command("hello", &skills);
        assert!(result.is_none());
    }
}

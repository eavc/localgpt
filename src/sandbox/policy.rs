use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::config::SandboxConfig;

/// High-level sandbox mode (user-facing setting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxMode {
    /// R/W in workspace + /tmp; R/O system dirs; deny credentials; deny network.
    WorkspaceWrite,
    /// R/O everywhere allowed; no writes anywhere; deny network.
    ReadOnly,
    /// Unrestricted — requires explicit opt-in.
    FullAccess,
}

/// Enforcement level based on detected kernel capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SandboxLevel {
    /// No kernel support — rlimits + timeout only.
    None,
    /// seccomp only — network blocking only.
    Minimal,
    /// Landlock V1+ + seccomp — filesystem + network isolation.
    Standard,
    /// Landlock V4+ + seccomp + userns — full isolation.
    Full,
}

/// Network access policy for sandboxed commands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkPolicy {
    /// No network connectivity at all.
    Deny,
    /// Allow through a proxy socket (future).
    AllowProxy(PathBuf),
}

/// Serializable sandbox policy passed to the re-exec'd child process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxPolicy {
    /// Workspace directory — gets R/W access.
    pub workspace_path: PathBuf,

    /// Additional read-only paths (system dirs, user-specified).
    pub read_only_paths: Vec<PathBuf>,

    /// Additional writable paths (user-specified).
    pub extra_write_paths: Vec<PathBuf>,

    /// Paths to explicitly deny (credential directories).
    pub deny_paths: Vec<PathBuf>,

    /// Network access policy.
    pub network: NetworkPolicy,

    /// Kill command after this many seconds.
    pub timeout_secs: u64,

    /// Maximum stdout+stderr bytes.
    pub max_output_bytes: u64,

    /// RLIMIT_FSIZE in bytes.
    pub max_file_size_bytes: u64,

    /// RLIMIT_NPROC.
    pub max_processes: u32,

    /// Enforcement level.
    pub level: SandboxLevel,
}

/// Default credential directories to deny access to.
fn default_deny_paths() -> Vec<PathBuf> {
    let home = dirs_home();
    vec![
        home.join(".ssh"),
        home.join(".aws"),
        home.join(".gnupg"),
        home.join(".config"),
        home.join(".docker"),
        home.join(".kube"),
        home.join(".npmrc"),
        home.join(".pypirc"),
        home.join(".netrc"),
    ]
}

/// Default system read-only paths.
fn default_read_only_paths() -> Vec<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        vec![
            PathBuf::from("/usr"),
            PathBuf::from("/lib"),
            PathBuf::from("/lib64"),
            PathBuf::from("/bin"),
            PathBuf::from("/sbin"),
            PathBuf::from("/etc"),
            PathBuf::from("/dev"),
            PathBuf::from("/proc/self"),
        ]
    }
    #[cfg(target_os = "macos")]
    {
        vec![
            PathBuf::from("/usr"),
            PathBuf::from("/bin"),
            PathBuf::from("/sbin"),
            PathBuf::from("/Library"),
            PathBuf::from("/System"),
            PathBuf::from("/etc"),
            PathBuf::from("/dev"),
            PathBuf::from("/var/folders"),
            PathBuf::from("/private/tmp"),
            PathBuf::from("/private/var"),
            PathBuf::from("/opt/homebrew"),
            PathBuf::from("/Applications"),
        ]
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        vec![]
    }
}

fn dirs_home() -> PathBuf {
    directories::BaseDirs::new()
        .map(|b| b.home_dir().to_path_buf())
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
        .unwrap_or_else(|| {
            eprintln!(
                "localgpt-sandbox: WARNING: cannot determine home directory, \
                 credential deny paths will not be effective"
            );
            PathBuf::from("/nonexistent")
        })
}

/// Check if a candidate path overlaps any deny path.
///
/// An "overlap" means the candidate is an ancestor of (or equal to) a deny path,
/// which would grant Landlock access to the deny subtree. Also catches the case
/// where the candidate IS a deny path.
fn overlaps_deny_path(candidate: &std::path::Path, deny_paths: &[PathBuf]) -> bool {
    let candidate_canonical = candidate
        .canonicalize()
        .unwrap_or_else(|_| candidate.to_path_buf());
    for deny in deny_paths {
        let deny_canonical = deny.canonicalize().unwrap_or_else(|_| deny.to_path_buf());
        // Candidate is ancestor of or equal to a deny path
        if deny_canonical.starts_with(&candidate_canonical) {
            return true;
        }
        // Candidate is inside a deny path
        if candidate_canonical.starts_with(&deny_canonical) {
            return true;
        }
    }
    false
}

/// Build a `SandboxPolicy` from sandbox config and workspace path.
pub fn build_policy(
    config: &SandboxConfig,
    workspace: &std::path::Path,
    level: SandboxLevel,
) -> SandboxPolicy {
    let mode = if config.level == crate::config::SandboxLevelConfig::None || !config.enabled {
        SandboxMode::FullAccess
    } else {
        SandboxMode::WorkspaceWrite
    };

    // Network is always deny for now. Proxy support is a future extension.
    let network = NetworkPolicy::Deny;

    let deny_paths = if mode == SandboxMode::FullAccess {
        vec![]
    } else {
        default_deny_paths()
    };

    let mut read_only = default_read_only_paths();
    for p in &config.allow_paths.read {
        let expanded = shellexpand::tilde(p);
        let candidate = PathBuf::from(expanded.to_string());
        if !overlaps_deny_path(&candidate, &deny_paths) {
            read_only.push(candidate);
        } else {
            eprintln!(
                "localgpt-sandbox: WARNING: ignoring allow_paths.read entry {:?} — \
                 it overlaps a credential deny path",
                p
            );
        }
    }

    let mut extra_write = vec![PathBuf::from("/tmp")];
    for p in &config.allow_paths.write {
        let expanded = shellexpand::tilde(p);
        let candidate = PathBuf::from(expanded.to_string());
        if !overlaps_deny_path(&candidate, &deny_paths) {
            extra_write.push(candidate);
        } else {
            eprintln!(
                "localgpt-sandbox: WARNING: ignoring allow_paths.write entry {:?} — \
                 it overlaps a credential deny path",
                p
            );
        }
    }

    SandboxPolicy {
        workspace_path: workspace.to_path_buf(),
        read_only_paths: read_only,
        extra_write_paths: extra_write,
        deny_paths,
        network,
        timeout_secs: config.timeout_secs,
        max_output_bytes: config.max_output_bytes,
        max_file_size_bytes: config.max_file_size_bytes,
        max_processes: config.max_processes,
        level,
    }
}

/// Check if a path falls within any of the credential deny paths.
pub fn is_path_denied(path: &std::path::Path, policy: &SandboxPolicy) -> bool {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    for deny in &policy.deny_paths {
        let deny_canonical = deny.canonicalize().unwrap_or_else(|_| deny.to_path_buf());
        if canonical.starts_with(&deny_canonical) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SandboxConfig;

    #[test]
    fn test_build_policy_workspace_write() {
        let config = SandboxConfig::default();
        let workspace = PathBuf::from("/home/user/project");
        let policy = build_policy(&config, &workspace, SandboxLevel::Standard);

        assert_eq!(policy.workspace_path, workspace);
        assert_eq!(policy.network, NetworkPolicy::Deny);
        assert_eq!(policy.level, SandboxLevel::Standard);
        assert!(!policy.deny_paths.is_empty());
        assert!(policy.extra_write_paths.contains(&PathBuf::from("/tmp")));
    }

    #[test]
    fn test_build_policy_disabled() {
        let mut config = SandboxConfig::default();
        config.enabled = false;
        let workspace = PathBuf::from("/home/user/project");
        let policy = build_policy(&config, &workspace, SandboxLevel::None);

        // When disabled, deny_paths is empty (full access mode)
        assert!(policy.deny_paths.is_empty());
    }

    #[test]
    fn test_sandbox_level_ordering() {
        assert!(SandboxLevel::None < SandboxLevel::Minimal);
        assert!(SandboxLevel::Minimal < SandboxLevel::Standard);
        assert!(SandboxLevel::Standard < SandboxLevel::Full);
    }

    #[test]
    fn test_policy_serialization_roundtrip() {
        let config = SandboxConfig::default();
        let workspace = PathBuf::from("/tmp/test-workspace");
        let policy = build_policy(&config, &workspace, SandboxLevel::Standard);

        let json = serde_json::to_string(&policy).unwrap();
        let deserialized: SandboxPolicy = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.workspace_path, policy.workspace_path);
        assert_eq!(deserialized.level, policy.level);
        assert_eq!(deserialized.network, policy.network);
        assert_eq!(deserialized.timeout_secs, policy.timeout_secs);
    }

    #[test]
    fn test_deny_paths_include_credentials() {
        let config = SandboxConfig::default();
        let workspace = PathBuf::from("/tmp/test");
        let policy = build_policy(&config, &workspace, SandboxLevel::Standard);

        let home = dirs_home();
        assert!(policy.deny_paths.contains(&home.join(".ssh")));
        assert!(policy.deny_paths.contains(&home.join(".aws")));
        assert!(policy.deny_paths.contains(&home.join(".gnupg")));
    }

    #[test]
    fn test_is_path_denied() {
        let config = SandboxConfig::default();
        let workspace = PathBuf::from("/tmp/test");
        let policy = build_policy(&config, &workspace, SandboxLevel::Standard);

        let home = dirs_home();
        // These use the deny list which contains home-relative paths
        // We test with raw paths since canonicalize may fail for non-existent paths
        let ssh_key = home.join(".ssh/id_rsa");
        // is_path_denied uses canonicalize which may fail for non-existent paths,
        // but it falls back to the raw path — so starts_with still works
        assert!(is_path_denied(&ssh_key, &policy));
    }

    #[test]
    fn test_overlaps_deny_path_ancestor() {
        let home = dirs_home();
        let deny_paths = vec![home.join(".ssh"), home.join(".aws")];

        // Home directory is an ancestor of .ssh — overlaps
        assert!(overlaps_deny_path(&home, &deny_paths));
    }

    #[test]
    fn test_overlaps_deny_path_exact() {
        let home = dirs_home();
        let deny_paths = vec![home.join(".ssh")];

        // Exact match — overlaps
        assert!(overlaps_deny_path(&home.join(".ssh"), &deny_paths));
    }

    #[test]
    fn test_overlaps_deny_path_child() {
        let home = dirs_home();
        let deny_paths = vec![home.join(".ssh")];

        // Inside a deny path — overlaps
        assert!(overlaps_deny_path(&home.join(".ssh/keys"), &deny_paths));
    }

    #[test]
    fn test_overlaps_deny_path_unrelated() {
        let deny_paths = vec![PathBuf::from("/home/user/.ssh")];

        // Unrelated path — no overlap
        assert!(!overlaps_deny_path(
            &PathBuf::from("/usr/share/dict"),
            &deny_paths
        ));
    }

    #[test]
    fn test_allow_paths_overlapping_deny_are_pruned() {
        let home = dirs_home();
        let mut config = SandboxConfig::default();
        // Try to add the home directory as a read-allowed path — should be pruned
        config
            .allow_paths
            .read
            .push(home.to_string_lossy().to_string());

        let workspace = PathBuf::from("/tmp/test");
        let policy = build_policy(&config, &workspace, SandboxLevel::Standard);

        // The home dir should NOT be in read_only_paths (it was pruned)
        assert!(
            !policy.read_only_paths.contains(&home),
            "Home directory should be pruned from allow_paths since it overlaps deny paths"
        );
    }
}

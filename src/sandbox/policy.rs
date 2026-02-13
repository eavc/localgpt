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
            // /proc/self intentionally excluded — exposes /proc/self/environ
            // which contains the parent process's full environment including
            // API keys, bypassing env_clear(). See SEC-15.
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

pub(super) fn dirs_home() -> PathBuf {
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

/// Check if a path is `/proc` or any subtree under it.
///
/// `/proc` exposes `/proc/self/environ` (parent process's full environment
/// including API keys), `/proc/<pid>/cmdline`, and other sensitive kernel
/// state. User-configured allow_paths must never grant access to it.
/// See SEC-15.
///
/// Handles non-canonical forms (`//proc`, `/./proc`, `/proc/../proc/self`)
/// and symlinks (via `canonicalize()` for paths that exist on disk).
fn is_proc_path(candidate: &std::path::Path) -> bool {
    // Try filesystem-level resolution first (follows symlinks).
    // Fall back to lexical normalization for non-existent paths.
    let normalized = candidate
        .canonicalize()
        .unwrap_or_else(|_| lexical_normalize(candidate));
    let s = normalized.to_string_lossy();
    s == "/proc" || s.starts_with("/proc/")
}

/// Lexical path normalization: collapse `.`, `..`, and redundant separators
/// without touching the filesystem.
///
/// Root-stable for absolute paths: `PathBuf::pop()` on root `/` is a no-op
/// (returns false), so repeated `..` cannot escape above root.
/// E.g. `/../../../proc` normalizes to `/proc`.
fn lexical_normalize(path: &std::path::Path) -> PathBuf {
    use std::path::Component;
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {} // skip `.`
            Component::ParentDir => {
                result.pop(); // no-op at root — cannot escape above /
            }
            other => result.push(other),
        }
    }
    result
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
        if is_proc_path(&candidate) {
            eprintln!(
                "localgpt-sandbox: WARNING: ignoring allow_paths.read entry {:?} — \
                 /proc access is denied (credential leak risk, see SEC-15)",
                p
            );
        } else if overlaps_deny_path(&candidate, &deny_paths) {
            eprintln!(
                "localgpt-sandbox: WARNING: ignoring allow_paths.read entry {:?} — \
                 it overlaps a credential deny path",
                p
            );
        } else {
            read_only.push(candidate);
        }
    }

    let mut extra_write = vec![PathBuf::from("/tmp")];
    for p in &config.allow_paths.write {
        let expanded = shellexpand::tilde(p);
        let candidate = PathBuf::from(expanded.to_string());
        if is_proc_path(&candidate) {
            eprintln!(
                "localgpt-sandbox: WARNING: ignoring allow_paths.write entry {:?} — \
                 /proc access is denied (credential leak risk, see SEC-15)",
                p
            );
        } else if overlaps_deny_path(&candidate, &deny_paths) {
            eprintln!(
                "localgpt-sandbox: WARNING: ignoring allow_paths.write entry {:?} — \
                 it overlaps a credential deny path",
                p
            );
        } else {
            extra_write.push(candidate);
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
    #[cfg(target_os = "linux")]
    fn test_default_read_only_paths_excludes_proc_self() {
        // SEC-15: /proc/self must not be in the default read-only paths because
        // it exposes /proc/self/environ which contains the parent's full
        // environment including API keys. Linux-only since /proc is a Linux concept.
        let paths = default_read_only_paths();
        let has_proc_self = paths.iter().any(|p| {
            let s = p.to_string_lossy();
            s == "/proc/self" || s.starts_with("/proc/self/")
        });
        assert!(
            !has_proc_self,
            "default_read_only_paths must not include /proc/self (credential leak via /proc/self/environ)"
        );
    }

    #[test]
    fn test_is_proc_path_exact() {
        assert!(is_proc_path(std::path::Path::new("/proc")));
    }

    #[test]
    fn test_is_proc_path_subtree() {
        assert!(is_proc_path(std::path::Path::new("/proc/self")));
        assert!(is_proc_path(std::path::Path::new("/proc/self/environ")));
        assert!(is_proc_path(std::path::Path::new("/proc/1/cmdline")));
    }

    #[test]
    fn test_is_proc_path_non_canonical() {
        // Double slash
        assert!(is_proc_path(std::path::Path::new("//proc")));
        assert!(is_proc_path(std::path::Path::new("//proc/self")));
        // Dot component
        assert!(is_proc_path(std::path::Path::new("/./proc")));
        assert!(is_proc_path(std::path::Path::new("/./proc/self/environ")));
        // Parent traversal back into /proc
        assert!(is_proc_path(std::path::Path::new("/proc/../proc/self")));
    }

    #[test]
    fn test_is_proc_path_negative() {
        assert!(!is_proc_path(std::path::Path::new("/usr")));
        assert!(!is_proc_path(std::path::Path::new("/tmp")));
        assert!(!is_proc_path(std::path::Path::new("/procfs")));
        assert!(!is_proc_path(std::path::Path::new("/home/proc")));
    }

    #[test]
    fn test_lexical_normalize_root_stable() {
        // Repeated `..` from root must not escape above `/`.
        // PathBuf::pop() on root is a no-op, so this normalizes to `/proc`.
        assert_eq!(
            lexical_normalize(std::path::Path::new("/../../../proc")),
            PathBuf::from("/proc")
        );
        assert_eq!(
            lexical_normalize(std::path::Path::new("/../../proc/self")),
            PathBuf::from("/proc/self")
        );
        // Normal absolute path passes through unchanged
        assert_eq!(
            lexical_normalize(std::path::Path::new("/usr/local/bin")),
            PathBuf::from("/usr/local/bin")
        );
        // Dot and double-slash collapse
        assert_eq!(
            lexical_normalize(std::path::Path::new("/./usr//local")),
            PathBuf::from("/usr/local")
        );
    }

    #[test]
    fn test_allow_paths_read_proc_is_rejected() {
        let mut config = SandboxConfig::default();
        config.allow_paths.read.push("/proc".to_string());

        let workspace = PathBuf::from("/tmp/test");
        let policy = build_policy(&config, &workspace, SandboxLevel::Standard);

        assert!(
            !policy.read_only_paths.iter().any(|p| is_proc_path(p)),
            "/proc must not appear in read_only_paths"
        );
    }

    #[test]
    fn test_allow_paths_read_proc_subtree_is_rejected() {
        let mut config = SandboxConfig::default();
        config.allow_paths.read.push("/proc/self".to_string());
        config.allow_paths.read.push("/proc/1/environ".to_string());

        let workspace = PathBuf::from("/tmp/test");
        let policy = build_policy(&config, &workspace, SandboxLevel::Standard);

        assert!(
            !policy.read_only_paths.iter().any(|p| is_proc_path(p)),
            "/proc subtree must not appear in read_only_paths"
        );
    }

    #[test]
    fn test_allow_paths_write_proc_is_rejected() {
        let mut config = SandboxConfig::default();
        config.allow_paths.write.push("/proc".to_string());
        config.allow_paths.write.push("/proc/self".to_string());

        let workspace = PathBuf::from("/tmp/test");
        let policy = build_policy(&config, &workspace, SandboxLevel::Standard);

        assert!(
            !policy.extra_write_paths.iter().any(|p| is_proc_path(p)),
            "/proc must not appear in extra_write_paths"
        );
    }

    #[test]
    fn test_allow_paths_non_proc_still_accepted() {
        let mut config = SandboxConfig::default();
        config.allow_paths.read.push("/opt/data".to_string());

        let workspace = PathBuf::from("/tmp/test");
        let policy = build_policy(&config, &workspace, SandboxLevel::Standard);

        assert!(
            policy.read_only_paths.contains(&PathBuf::from("/opt/data")),
            "non-/proc path should be accepted in read_only_paths"
        );
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

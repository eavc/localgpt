use super::policy::{NetworkPolicy, SandboxPolicy};

/// Apply macOS sandbox enforcement using Seatbelt SBPL profiles.
///
/// Instead of applying sandbox restrictions in the current process,
/// we modify the exec target to use `sandbox-exec -p <profile> bash -c <command>`.
/// This is called from the child process before exec.
pub fn apply_sandbox(policy: &SandboxPolicy) -> Result<(), String> {
    let profile = generate_sbpl_profile(policy);

    // Store profile in env for the child's exec_bash to pick up.
    // On macOS, the child execs sandbox-exec instead of bash directly.
    // SAFETY: set_var is used in a single-threaded child process before exec.
    #[allow(unused_unsafe)]
    unsafe {
        std::env::set_var("_LOCALGPT_SBPL_PROFILE", &profile);
    }

    Ok(())
}

/// Execute a command under sandbox-exec with the given SBPL profile.
/// This replaces the standard exec_bash on macOS when sandbox is active.
pub fn exec_sandboxed(command: &str) -> ! {
    use std::os::unix::process::CommandExt;

    let profile = std::env::var("_LOCALGPT_SBPL_PROFILE").unwrap_or_default();

    if profile.is_empty() {
        let err = std::process::Command::new("/bin/bash")
            .arg("-c")
            .arg(command)
            .exec();
        eprintln!("localgpt-sandbox: failed to exec bash: {}", err);
        std::process::exit(1);
    }

    let err = std::process::Command::new("/usr/bin/sandbox-exec")
        .arg("-p")
        .arg(&profile)
        .arg("/bin/bash")
        .arg("-c")
        .arg(command)
        .exec();

    eprintln!("localgpt-sandbox: failed to exec sandbox-exec: {}", err);
    std::process::exit(1);
}

/// Generate a Seatbelt SBPL profile string from a SandboxPolicy.
///
/// Strategy: allow reads broadly (bash needs access to system libraries,
/// dyld cache, etc.), then deny credential paths specifically. Restrict
/// writes to workspace and /tmp only. Deny all network access.
pub fn generate_sbpl_profile(policy: &SandboxPolicy) -> String {
    let mut rules = vec![
        "(version 1)".to_string(),
        "(deny default)".to_string(),
        // Process operations — bash needs these
        "(allow process*)".to_string(),
        "(allow signal)".to_string(),
        // Mach IPC — required for dyld, libSystem, and most macOS subsystems
        "(allow mach*)".to_string(),
        // IPC — shared memory, semaphores
        "(allow ipc*)".to_string(),
        // Sysctl — process info, hw.ncpu, etc.
        "(allow sysctl*)".to_string(),
        // Pseudo-terminals — needed for interactive commands
        "(allow pseudo-tty)".to_string(),
        // File reads — allow broadly, then deny credentials
        "(allow file-read*)".to_string(),
        "(allow file-read-metadata)".to_string(),
        // /dev devices — bash needs /dev/null, /dev/urandom, tty
        "(allow file-write* (subpath \"/dev\"))".to_string(),
    ];

    // Allow writes to workspace
    let workspace_escaped = escape_sbpl_path(&policy.workspace_path.to_string_lossy());
    rules.push(format!(
        "(allow file-write* (subpath \"{}\"))",
        workspace_escaped
    ));

    // Allow writes to extra writable paths (/tmp, user-configured)
    for path in &policy.extra_write_paths {
        let escaped = escape_sbpl_path(&path.to_string_lossy());
        rules.push(format!("(allow file-write* (subpath \"{}\"))", escaped));
    }

    // Deny credential directories (overrides the broad file-read* allow above)
    for path in &policy.deny_paths {
        let escaped = escape_sbpl_path(&path.to_string_lossy());
        rules.push(format!(
            "(deny file-read* file-write* (subpath \"{}\"))",
            escaped
        ));
    }

    // Network policy — deny all network by default
    match &policy.network {
        NetworkPolicy::Deny => {
            rules.push("(deny network*)".to_string());
        }
        NetworkPolicy::AllowProxy(socket_path) => {
            let escaped = escape_sbpl_path(&socket_path.to_string_lossy());
            rules.push("(deny network*)".to_string());
            rules.push(format!(
                "(allow network-outbound (local file \"{}\"))",
                escaped
            ));
        }
    }

    rules.join("\n")
}

/// Escape a path string for use in SBPL profiles.
fn escape_sbpl_path(path: &str) -> String {
    path.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::policy::{NetworkPolicy, SandboxLevel, SandboxPolicy};
    use std::path::PathBuf;

    fn test_policy() -> SandboxPolicy {
        SandboxPolicy {
            workspace_path: PathBuf::from("/Users/test/project"),
            read_only_paths: vec![
                PathBuf::from("/usr"),
                PathBuf::from("/bin"),
                PathBuf::from("/Library"),
            ],
            extra_write_paths: vec![PathBuf::from("/tmp")],
            deny_paths: vec![
                PathBuf::from("/Users/test/.ssh"),
                PathBuf::from("/Users/test/.aws"),
            ],
            network: NetworkPolicy::Deny,
            timeout_secs: 120,
            max_output_bytes: 1_048_576,
            max_file_size_bytes: 52_428_800,
            max_processes: 64,
            level: SandboxLevel::Standard,
        }
    }

    #[test]
    fn test_generate_sbpl_profile_contains_deny_default() {
        let policy = test_policy();
        let profile = generate_sbpl_profile(&policy);
        assert!(profile.contains("(deny default)"));
    }

    #[test]
    fn test_generate_sbpl_profile_allows_workspace_writes() {
        let policy = test_policy();
        let profile = generate_sbpl_profile(&policy);
        assert!(profile.contains("(allow file-write* (subpath \"/Users/test/project\"))"));
    }

    #[test]
    fn test_generate_sbpl_profile_allows_broad_reads() {
        let policy = test_policy();
        let profile = generate_sbpl_profile(&policy);
        assert!(profile.contains("(allow file-read*)"));
    }

    #[test]
    fn test_generate_sbpl_profile_denies_network() {
        let policy = test_policy();
        let profile = generate_sbpl_profile(&policy);
        assert!(profile.contains("(deny network*)"));
    }

    #[test]
    fn test_generate_sbpl_profile_denies_credentials() {
        let policy = test_policy();
        let profile = generate_sbpl_profile(&policy);
        assert!(profile.contains("(deny file-read* file-write* (subpath \"/Users/test/.ssh\"))"));
        assert!(profile.contains("(deny file-read* file-write* (subpath \"/Users/test/.aws\"))"));
    }

    #[test]
    fn test_escape_sbpl_path() {
        assert_eq!(escape_sbpl_path("/simple/path"), "/simple/path");
        assert_eq!(
            escape_sbpl_path("/path/with \"quotes\""),
            "/path/with \\\"quotes\\\""
        );
        assert_eq!(
            escape_sbpl_path("/path\\with\\backslashes"),
            "/path\\\\with\\\\backslashes"
        );
    }

    #[test]
    fn test_generate_sbpl_profile_allows_tmp_writes() {
        let policy = test_policy();
        let profile = generate_sbpl_profile(&policy);
        assert!(profile.contains("(allow file-write* (subpath \"/tmp\"))"));
    }
}

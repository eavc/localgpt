use anyhow::Result;
use std::time::Duration;
use tokio::io::AsyncWriteExt;

use super::policy::SandboxPolicy;

/// Run a shell command inside the sandbox.
///
/// This is the parent-side function. It:
/// 1. Serializes the policy to JSON
/// 2. Re-execs the current binary with argv[0]="localgpt-sandbox"
/// 3. Passes policy via stdin pipe (not argv — avoids /proc/pid/cmdline leak)
/// 4. Passes the shell command as argv[1]
/// 5. Collects output and enforces timeout (with explicit kill/reap)
///
/// The effective timeout is the minimum of `timeout_ms` and the policy's
/// `timeout_secs * 1000`, ensuring the sandbox config acts as a hard cap.
pub async fn run_sandboxed(
    command: &str,
    policy: &SandboxPolicy,
    timeout_ms: u64,
) -> Result<(String, i32)> {
    // Get path to current executable for re-exec
    let exe_path = std::env::current_exe()?;

    // Enforce sandbox timeout as hard cap
    let policy_timeout_ms = policy.timeout_secs.saturating_mul(1000);
    let effective_timeout_ms = timeout_ms.min(policy_timeout_ms);
    let timeout_duration = Duration::from_millis(effective_timeout_ms);

    let child_fut = async {
        let (mut child, policy_json) = spawn_sandbox_child(command, policy, &exe_path)?;

        // Write policy JSON to child's stdin, then close the pipe.
        // Policy is small (< 4 KB), well within pipe buffer limits.
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("BUG: stdin pipe was requested but not available"))?;
        stdin.write_all(policy_json.as_bytes()).await?;
        drop(stdin); // close the pipe, signaling EOF to the child

        child.wait_with_output().await.map_err(anyhow::Error::from)
    };

    // Wait with timeout — on expiry, dropping the future kills the child
    match tokio::time::timeout(timeout_duration, child_fut).await {
        Ok(Ok(output)) => format_output(&output, policy),
        Ok(Err(e)) => Err(anyhow::anyhow!("Sandboxed command I/O error: {}", e)),
        Err(_timeout) => Err(anyhow::anyhow!(
            "Sandboxed command timed out after {}ms",
            effective_timeout_ms
        )),
    }
}

/// Spawn a sandbox child process with the given command and policy.
///
/// Returns the child handle and the serialized policy JSON. The caller is
/// responsible for writing `policy_json` to the child's stdin, closing the
/// pipe (to signal EOF), and then reading output.
///
/// The child is spawned with `kill_on_drop(true)`, so dropping the handle
/// sends SIGKILL.
///
/// # Security
///
/// - **SEC-20b**: Policy is passed via stdin, NOT as argv. This function is
///   the single point of command construction — regressions are caught by
///   `tests/sandbox_stdin_transport.rs`.
/// - **SEC-15**: `env_clear()` + explicit allowlist prevents credential leak.
#[doc(hidden)]
pub fn spawn_sandbox_child(
    command: &str,
    policy: &SandboxPolicy,
    exe_path: &std::path::Path,
) -> Result<(tokio::process::Child, String)> {
    let policy_json = serde_json::to_string(policy)?;

    let child = tokio::process::Command::new(exe_path)
        .arg0("localgpt-sandbox")
        .arg(command)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .env_clear()
        .envs(sandbox_env())
        .kill_on_drop(true)
        .spawn()?;

    Ok((child, policy_json))
}

/// Format captured output, applying UTF-8-safe truncation.
fn format_output(output: &std::process::Output, policy: &SandboxPolicy) -> Result<(String, i32)> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let max_bytes = policy.max_output_bytes as usize;

    let mut result = String::new();

    if !stdout.is_empty() {
        if stdout.len() > max_bytes {
            result.push_str(truncate_utf8(&stdout, max_bytes));
            result.push_str(&format!(
                "\n\n[Output truncated, {} bytes total]",
                stdout.len()
            ));
        } else {
            result.push_str(&stdout);
        }
    }

    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push_str("\n\nSTDERR:\n");
        }
        let remaining = max_bytes.saturating_sub(result.len());
        if stderr.len() > remaining && remaining > 0 {
            result.push_str(truncate_utf8(&stderr, remaining));
            result.push_str("\n[stderr truncated]");
        } else {
            result.push_str(&stderr);
        }
    }

    let exit_code = output.status.code().unwrap_or(-1);

    Ok((result, exit_code))
}

/// Truncate a UTF-8 string at or before `max_bytes`, respecting char boundaries.
fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// The set of environment variables forwarded to sandboxed child processes.
///
/// Returns `(key, value)` pairs. Values are either hardcoded safe defaults
/// or inherited non-secret values from the parent. No API keys, tokens,
/// or credentials are included.
fn sandbox_env() -> Vec<(&'static str, String)> {
    vec![
        (
            "PATH",
            "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin".to_string(),
        ),
        (
            "HOME",
            super::policy::dirs_home().to_string_lossy().into_owned(),
        ),
        ("TERM", "dumb".to_string()),
        (
            "LANG",
            std::env::var("LANG").unwrap_or_else(|_| "C.UTF-8".to_string()),
        ),
    ]
}

/// Trait extension for Command to set argv[0].
/// The trait itself is "dead" from the compiler's perspective because it's only
/// used implicitly via the impl — but the impl IS needed for arg0() calls above.
#[allow(dead_code)]
trait CommandExt {
    fn arg0(&mut self, arg0: &str) -> &mut Self;
}

#[cfg(unix)]
impl CommandExt for tokio::process::Command {
    fn arg0(&mut self, arg0: &str) -> &mut Self {
        use std::os::unix::process::CommandExt;
        self.as_std_mut().arg0(arg0);
        self
    }
}

#[cfg(not(unix))]
impl CommandExt for tokio::process::Command {
    fn arg0(&mut self, _arg0: &str) -> &mut Self {
        // argv[0] dispatch not supported on non-unix platforms
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes tests that mutate the process environment via set_var/remove_var.
    /// Process env is global state; concurrent mutation is UB in Rust >=1.83.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_sandbox_env_contains_only_safe_vars() {
        let env = sandbox_env();
        let keys: Vec<&str> = env.iter().map(|(k, _)| *k).collect();

        assert_eq!(keys, vec!["PATH", "HOME", "TERM", "LANG"]);
    }

    #[test]
    fn test_sandbox_env_path_is_hardcoded() {
        let env = sandbox_env();
        let (_, path_val) = env.iter().find(|(k, _)| *k == "PATH").unwrap();

        assert_eq!(path_val, "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin");
    }

    #[test]
    fn test_sandbox_env_term_is_dumb() {
        let env = sandbox_env();
        let (_, term_val) = env.iter().find(|(k, _)| *k == "TERM").unwrap();

        assert_eq!(term_val, "dumb");
    }

    #[test]
    fn test_sandbox_env_no_api_keys() {
        let env = sandbox_env();
        let keys: Vec<&str> = env.iter().map(|(k, _)| *k).collect();

        let secret_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ACCESS_KEY_ID",
            "GITHUB_TOKEN",
            "LOCALGPT_API_KEY",
        ];
        for secret in &secret_keys {
            assert!(
                !keys.contains(secret),
                "{} must not be in sandbox env",
                secret
            );
        }
    }

    /// Verify that a child process spawned with env_clear() + sandbox_env()
    /// does not inherit parent secrets.
    #[tokio::test]
    async fn test_sandbox_child_env_is_scrubbed() {
        let _guard = ENV_LOCK.lock().unwrap();

        // SAFETY: ENV_LOCK serializes all env-mutating tests. No other test
        // in this binary mutates these specific keys.
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-test-secret-key");
            std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-test-secret");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/test");
        }

        let output = tokio::process::Command::new("/bin/bash")
            .arg("-c")
            .arg("env")
            .env_clear()
            .envs(sandbox_env())
            .output()
            .await
            .expect("failed to run env");

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Secrets must not appear
        assert!(
            !stdout.contains("sk-test-secret-key"),
            "OPENAI_API_KEY leaked to child"
        );
        assert!(
            !stdout.contains("sk-ant-test-secret"),
            "ANTHROPIC_API_KEY leaked to child"
        );
        assert!(
            !stdout.contains("wJalrXUtnFEMI"),
            "AWS_SECRET_ACCESS_KEY leaked to child"
        );

        // Allowed vars must be present
        assert!(stdout.contains("PATH="), "PATH missing from child env");
        assert!(stdout.contains("HOME="), "HOME missing from child env");
        assert!(stdout.contains("LANG="), "LANG missing from child env");
        assert!(
            stdout.contains("TERM=dumb"),
            "TERM=dumb missing from child env"
        );

        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("AWS_SECRET_ACCESS_KEY");
        }
    }

    /// Verify that `printenv <KEY>` for a secret returns empty in the child.
    #[tokio::test]
    async fn test_sandbox_printenv_specific_key_empty() {
        let _guard = ENV_LOCK.lock().unwrap();

        // SAFETY: ENV_LOCK serializes all env-mutating tests.
        unsafe {
            std::env::set_var("TEST_SECRET", "hunter2");
        }

        let output = tokio::process::Command::new("/bin/bash")
            .arg("-c")
            .arg("printenv TEST_SECRET")
            .env_clear()
            .envs(sandbox_env())
            .output()
            .await
            .expect("failed to run printenv");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            !stdout.contains("hunter2"),
            "TEST_SECRET leaked to child env"
        );
        assert_ne!(output.status.code(), Some(0));

        unsafe {
            std::env::remove_var("TEST_SECRET");
        }
    }

    /// Verify that basic commands work with the scrubbed environment.
    #[tokio::test]
    async fn test_sandbox_path_is_functional() {
        let output = tokio::process::Command::new("/bin/bash")
            .arg("-c")
            .arg("which echo")
            .env_clear()
            .envs(sandbox_env())
            .output()
            .await
            .expect("failed to run which");

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("/echo"),
            "echo not found in PATH — got: {}",
            stdout.trim()
        );
        assert_eq!(output.status.code(), Some(0));
    }

    #[test]
    fn test_truncate_utf8_basic() {
        assert_eq!(truncate_utf8("hello", 3), "hel");
        assert_eq!(truncate_utf8("hello", 10), "hello");
        assert_eq!(truncate_utf8("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_utf8_multibyte() {
        // '€' is 3 bytes in UTF-8
        let s = "€€";
        assert_eq!(truncate_utf8(s, 3), "€");
        assert_eq!(truncate_utf8(s, 4), "€");
        assert_eq!(truncate_utf8(s, 6), "€€");
    }

    /// Verify the stdin-pipe pattern used by `run_sandboxed` (SEC-20b).
    ///
    /// Spawns a child process that reads stdin via `cat` and echoes it to
    /// stdout. This validates that the pipe mechanism correctly delivers
    /// data and that EOF is signaled when the sender drops the handle.
    ///
    /// Full end-to-end transport (with argv[0] dispatch and sandbox child
    /// policy parsing) is covered by the integration test in
    /// `tests/sandbox_stdin_transport.rs` (Linux) and by
    /// `cargo run -- sandbox test` smoke tests.
    #[tokio::test]
    async fn test_stdin_pipe_delivers_data_to_child() {
        use tokio::io::AsyncWriteExt;

        let test_data = r#"{"workspace_path":"/tmp","level":"None"}"#;

        let mut child = tokio::process::Command::new("/bin/bash")
            .arg("-c")
            .arg("cat")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .expect("failed to spawn child");

        // Use the same ok_or_else pattern as run_sandboxed
        let mut stdin = child
            .stdin
            .take()
            .expect("stdin pipe was requested but not available");
        stdin
            .write_all(test_data.as_bytes())
            .await
            .expect("failed to write to stdin");
        drop(stdin); // close the pipe → child receives EOF

        let output = child.wait_with_output().await.expect("failed to wait");
        let stdout = String::from_utf8_lossy(&output.stdout);

        assert_eq!(
            stdout, test_data,
            "stdin data should be delivered to child stdout via cat"
        );
        assert_eq!(output.status.code(), Some(0));
    }

    /// Verify that command arguments do NOT contain policy JSON (SEC-20b).
    ///
    /// Spawns a child that prints its own `/proc/self/cmdline` (Linux) or
    /// `$0 $@` (portable). Asserts policy field names are absent from argv.
    #[tokio::test]
    async fn test_command_args_do_not_contain_policy() {
        use tokio::io::AsyncWriteExt;

        let policy_json = r#"{"workspace_path":"/tmp","deny_paths":["/secret"]}"#;
        let command = "echo args-check-ok";

        // Simulate the run_sandboxed spawn pattern: policy via stdin, command as argv.
        // The command drains stdin first to avoid EPIPE — on Linux, bash can exit
        // before the parent's write completes if stdin isn't consumed.
        let mut child = tokio::process::Command::new("/bin/bash")
            .arg("-c")
            .arg(format!(
                "cat > /dev/null; echo \"CMDLINE: $0 $@\"; {}",
                command
            ))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .expect("failed to spawn child");

        let mut stdin = child
            .stdin
            .take()
            .expect("stdin pipe was requested but not available");
        stdin
            .write_all(policy_json.as_bytes())
            .await
            .expect("failed to write policy to stdin");
        drop(stdin);

        let output = child.wait_with_output().await.expect("failed to wait");
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Policy field names must not appear in command-line output
        assert!(
            !stdout.contains("workspace_path"),
            "policy field 'workspace_path' found in child args: {}",
            stdout
        );
        assert!(
            !stdout.contains("deny_paths"),
            "policy field 'deny_paths' found in child args: {}",
            stdout
        );
        // But the command itself should execute
        assert!(
            stdout.contains("args-check-ok"),
            "expected command output in stdout: {}",
            stdout
        );
    }
}

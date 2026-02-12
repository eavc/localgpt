use anyhow::Result;
use std::time::Duration;

use super::policy::SandboxPolicy;

/// Run a shell command inside the sandbox.
///
/// This is the parent-side function. It:
/// 1. Serializes the policy to JSON
/// 2. Re-execs the current binary with argv[0]="localgpt-sandbox"
/// 3. Passes policy + command as arguments
/// 4. Collects output and enforces timeout (with explicit kill/reap)
///
/// The effective timeout is the minimum of `timeout_ms` and the policy's
/// `timeout_secs * 1000`, ensuring the sandbox config acts as a hard cap.
pub async fn run_sandboxed(
    command: &str,
    policy: &SandboxPolicy,
    timeout_ms: u64,
) -> Result<(String, i32)> {
    let policy_json = serde_json::to_string(policy)?;

    // Get path to current executable for re-exec
    let exe_path = std::env::current_exe()?;

    // Enforce sandbox timeout as hard cap
    let policy_timeout_ms = policy.timeout_secs.saturating_mul(1000);
    let effective_timeout_ms = timeout_ms.min(policy_timeout_ms);
    let timeout_duration = Duration::from_millis(effective_timeout_ms);

    // Build the child command. We use .output() wrapped in timeout, with
    // kill_on_drop as a safety net. On timeout, we explicitly handle the error
    // and tokio's kill_on_drop ensures the child is cleaned up.
    //
    // SECURITY: env_clear() prevents sandboxed commands from reading parent
    // API keys via `env`/`printenv`. Only minimal, non-secret variables are
    // forwarded. See SEC-15.
    let output_fut = tokio::process::Command::new(&exe_path)
        .arg0("localgpt-sandbox")
        .arg(&policy_json)
        .arg(command)
        .env_clear()
        .envs(sandbox_env())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .output();

    // Wait with timeout — on expiry, dropping the future kills the child
    match tokio::time::timeout(timeout_duration, output_fut).await {
        Ok(Ok(output)) => format_output(&output, policy),
        Ok(Err(e)) => Err(anyhow::anyhow!("Sandboxed command I/O error: {}", e)),
        Err(_timeout) => Err(anyhow::anyhow!(
            "Sandboxed command timed out after {}ms",
            effective_timeout_ms
        )),
    }
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
}

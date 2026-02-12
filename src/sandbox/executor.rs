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
    let output_fut = tokio::process::Command::new(&exe_path)
        .arg0("localgpt-sandbox")
        .arg(&policy_json)
        .arg(command)
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

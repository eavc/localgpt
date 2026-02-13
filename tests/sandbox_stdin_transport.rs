//! Integration test for SEC-20b: sandbox policy transport via stdin.
//!
//! Exercises the REAL sandbox child path (argv[0] dispatch →
//! `sandbox_child_main` → stdin policy parse → exec bash) and verifies
//! that `/proc/<pid>/cmdline` never contains policy fields.
//!
//! Two tests:
//! 1. Direct binary spawn — validates the child dispatch path.
//! 2. Via `spawn_sandbox_child` — validates the production executor's
//!    Command construction (the exact code `run_sandboxed` uses).
//!
//! Linux-only: requires `/proc` filesystem for cmdline inspection.

#![cfg(target_os = "linux")]

use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;

use localgpt::sandbox::{spawn_sandbox_child, NetworkPolicy, SandboxLevel, SandboxPolicy};

/// Construct a minimal `SandboxPolicy` with `level=None` (skips Landlock/seccomp).
fn test_policy() -> SandboxPolicy {
    SandboxPolicy {
        workspace_path: PathBuf::from("/tmp"),
        read_only_paths: vec![],
        extra_write_paths: vec![],
        deny_paths: vec![PathBuf::from("/secret/creds")],
        network: NetworkPolicy::Deny,
        timeout_secs: 30,
        max_output_bytes: 65536,
        max_file_size_bytes: 1_048_576,
        max_processes: 64,
        level: SandboxLevel::None,
    }
}

/// Spawn the actual `localgpt` binary directly with `argv[0]="localgpt-sandbox"`,
/// read `/proc/<pid>/cmdline` while the child is blocked on stdin, then
/// deliver the policy and confirm the command executes.
///
/// This validates the child dispatch path (main.rs argv[0] → sandbox_child_main).
#[test]
fn test_sandbox_child_proc_cmdline_excludes_policy() {
    let exe = env!("CARGO_BIN_EXE_localgpt");
    let policy_json = serde_json::to_string(&test_policy()).expect("serialize policy");

    let mut child = std::process::Command::new(exe)
        .arg0("localgpt-sandbox")
        .arg("echo cmdline-transport-ok")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn localgpt-sandbox child");

    let child_pid = child.id();

    // The child is alive and blocked on stdin `read_to_string()`.
    // `/proc/<pid>/cmdline` is kernel-populated at exec time, so it's
    // already available. Entries are null-byte separated.
    //
    // Save the Result — don't unwrap yet. If this fails, we still need
    // to deliver policy and reap the child to avoid leaking the process.
    let cmdline_path = format!("/proc/{}/cmdline", child_pid);
    let cmdline_result = std::fs::read(&cmdline_path);

    // Always deliver policy and wait before any assertions or unwraps.
    // This ensures the child is cleaned up (no leaked processes).
    {
        let stdin = child.stdin.as_mut().expect("stdin pipe");
        stdin
            .write_all(policy_json.as_bytes())
            .expect("write policy to stdin");
    }
    drop(child.stdin.take()); // close pipe → child receives EOF

    let output = child.wait_with_output().expect("wait for child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Now unwrap and assert — child is already reaped regardless of outcome
    let cmdline_bytes =
        cmdline_result.unwrap_or_else(|e| panic!("failed to read {}: {}", cmdline_path, e));
    let cmdline = String::from_utf8_lossy(&cmdline_bytes);

    assert!(
        !cmdline.contains("workspace_path"),
        "policy field 'workspace_path' leaked to /proc/{}/cmdline: {:?}",
        child_pid,
        cmdline
    );
    assert!(
        !cmdline.contains("deny_paths"),
        "policy field 'deny_paths' leaked to /proc/{}/cmdline: {:?}",
        child_pid,
        cmdline
    );
    assert!(
        !cmdline.contains("read_only_paths"),
        "policy field 'read_only_paths' leaked to /proc/{}/cmdline: {:?}",
        child_pid,
        cmdline
    );
    assert!(
        !cmdline.contains("/secret/creds"),
        "deny path value leaked to /proc/{}/cmdline: {:?}",
        child_pid,
        cmdline
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "sandbox child should exit 0; stdout: {}, stderr: {}",
        stdout.trim(),
        stderr.trim()
    );
    assert!(
        stdout.contains("cmdline-transport-ok"),
        "command should have executed; stdout: {}, stderr: {}",
        stdout.trim(),
        stderr.trim()
    );
}

/// Exercise the production executor's `spawn_sandbox_child` — the exact
/// function that `run_sandboxed` uses to construct and spawn the Command.
///
/// Reads `/proc/<pid>/cmdline` while the child is blocked on stdin, then
/// delivers policy and verifies execution. A regression in `executor.rs`
/// (e.g. re-adding `.arg(&policy_json)`) would fail this test.
#[tokio::test]
async fn test_spawn_sandbox_child_cmdline_excludes_policy() {
    use tokio::io::AsyncWriteExt;

    let exe = std::path::Path::new(env!("CARGO_BIN_EXE_localgpt"));
    let policy = test_policy();

    let (mut child, policy_json) = spawn_sandbox_child("echo executor-path-ok", &policy, exe)
        .expect("spawn_sandbox_child failed");

    let child_pid = child.id().expect("child should have a PID");

    // Child is blocked on stdin. Read /proc/<pid>/cmdline from parent.
    // Save the Result — don't unwrap yet. If this fails, we still need
    // to deliver policy and reap the child to avoid leaking the process.
    // (Note: spawn_sandbox_child sets kill_on_drop(true), so the child
    // would be killed on panic-unwind, but explicit cleanup is cleaner.)
    let cmdline_path = format!("/proc/{}/cmdline", child_pid);
    let cmdline_result = std::fs::read(&cmdline_path);

    // Deliver policy and wait before any assertions or unwraps.
    let mut stdin = child
        .stdin
        .take()
        .expect("stdin pipe was requested but not available");
    stdin
        .write_all(policy_json.as_bytes())
        .await
        .expect("write policy to stdin");
    drop(stdin); // close pipe → child receives EOF

    let output = child.wait_with_output().await.expect("wait for child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Now unwrap and assert — child is already reaped regardless of outcome
    let cmdline_bytes =
        cmdline_result.unwrap_or_else(|e| panic!("failed to read {}: {}", cmdline_path, e));
    let cmdline = String::from_utf8_lossy(&cmdline_bytes);

    assert!(
        !cmdline.contains("workspace_path"),
        "policy field 'workspace_path' in executor cmdline: {:?}",
        cmdline
    );
    assert!(
        !cmdline.contains("deny_paths"),
        "policy field 'deny_paths' in executor cmdline: {:?}",
        cmdline
    );
    assert!(
        !cmdline.contains("read_only_paths"),
        "policy field 'read_only_paths' in executor cmdline: {:?}",
        cmdline
    );
    assert!(
        !cmdline.contains("/secret/creds"),
        "deny path value in executor cmdline: {:?}",
        cmdline
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "executor child should exit 0; stdout: {}, stderr: {}",
        stdout.trim(),
        stderr.trim()
    );
    assert!(
        stdout.contains("executor-path-ok"),
        "command should have executed; stdout: {}, stderr: {}",
        stdout.trim(),
        stderr.trim()
    );
}

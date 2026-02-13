use super::policy::{SandboxLevel, SandboxPolicy};

/// Entry point for the sandbox child process.
///
/// Called when the binary detects argv[0] ends with "localgpt-sandbox".
/// This function never returns — it either execs bash or exits.
///
/// argv layout:
///   argv[0] = "localgpt-sandbox" (already consumed by dispatch)
///   argv[1] = shell command to execute
///
/// Policy is read from stdin (JSON). This avoids leaking sandbox
/// configuration through /proc/<pid>/cmdline. See SEC-20b.
pub fn sandbox_child_main() -> ! {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("localgpt-sandbox: expected command argument");
        std::process::exit(1);
    }

    // Read policy JSON from stdin (closed by parent after write).
    // Cap at 64 KiB to prevent memory exhaustion from malicious input.
    // Valid policies are < 4 KiB in practice.
    const MAX_POLICY_BYTES: u64 = 64 * 1024;
    let policy_json = {
        use std::io::Read;
        let mut buf = String::new();
        let bytes_read = match std::io::stdin()
            .take(MAX_POLICY_BYTES)
            .read_to_string(&mut buf)
        {
            Ok(n) => n,
            Err(e) => {
                eprintln!("localgpt-sandbox: failed to read policy from stdin: {}", e);
                std::process::exit(1);
            }
        };
        // `>=` is intentional: `Take` reads at most MAX_POLICY_BYTES, so
        // we cannot distinguish "exactly MAX_POLICY_BYTES" from "truncated
        // at MAX_POLICY_BYTES". Fail-closed: reject anything that *might*
        // have been truncated. This is safe because valid policies are well
        // below 64 KiB.
        if bytes_read as u64 >= MAX_POLICY_BYTES {
            eprintln!(
                "localgpt-sandbox: policy JSON too large (read {} bytes, limit {})",
                bytes_read, MAX_POLICY_BYTES
            );
            std::process::exit(1);
        }
        buf
    };

    let policy: SandboxPolicy = match serde_json::from_str(&policy_json) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("localgpt-sandbox: failed to parse policy: {}", e);
            std::process::exit(1);
        }
    };

    let command = &args[1];

    // Apply resource limits first (works on all platforms)
    if let Err(e) = apply_rlimits(&policy) {
        eprintln!("localgpt-sandbox: failed to apply rlimits: {}", e);
        std::process::exit(1);
    }

    // Apply platform-specific sandbox enforcement
    if policy.level > SandboxLevel::None {
        if let Err(e) = apply_platform_sandbox(&policy) {
            eprintln!("localgpt-sandbox: failed to apply sandbox: {}", e);
            std::process::exit(1);
        }
    }

    // exec bash -c <command>
    exec_bash(command);
}

/// Apply resource limits using setrlimit.
fn apply_rlimits(policy: &SandboxPolicy) -> Result<(), String> {
    use nix::sys::resource::{setrlimit, Resource};

    // RLIMIT_FSIZE — max file size
    setrlimit(
        Resource::RLIMIT_FSIZE,
        policy.max_file_size_bytes,
        policy.max_file_size_bytes,
    )
    .map_err(|e| format!("RLIMIT_FSIZE: {}", e))?;

    // RLIMIT_NPROC — max processes (Linux only; not available on macOS)
    #[cfg(target_os = "linux")]
    {
        let nproc = policy.max_processes as u64;
        setrlimit(Resource::RLIMIT_NPROC, nproc, nproc)
            .map_err(|e| format!("RLIMIT_NPROC: {}", e))?;
    }

    // RLIMIT_NOFILE — max open files (256)
    setrlimit(Resource::RLIMIT_NOFILE, 256, 256).map_err(|e| format!("RLIMIT_NOFILE: {}", e))?;

    Ok(())
}

/// Apply platform-specific sandbox enforcement.
fn apply_platform_sandbox(policy: &SandboxPolicy) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        super::linux::apply_sandbox(policy)
    }

    #[cfg(target_os = "macos")]
    {
        super::macos::apply_sandbox(policy)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = policy;
        // No platform-specific sandbox — rlimits only
        Ok(())
    }
}

/// Exec bash with the given command, replacing the current process.
/// On macOS, if sandbox-exec profile is set, uses sandbox-exec instead.
fn exec_bash(command: &str) -> ! {
    #[cfg(target_os = "macos")]
    {
        // On macOS, apply_sandbox sets _LOCALGPT_SBPL_PROFILE env var.
        // If set, exec through sandbox-exec instead of plain bash.
        if std::env::var("_LOCALGPT_SBPL_PROFILE").is_ok() {
            super::macos::exec_sandboxed(command);
        }
    }

    use std::os::unix::process::CommandExt;

    let err = std::process::Command::new("/bin/bash")
        .arg("-c")
        .arg(command)
        .exec();

    // exec() only returns on error
    eprintln!("localgpt-sandbox: failed to exec bash: {}", err);
    std::process::exit(1);
}

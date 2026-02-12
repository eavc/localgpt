use super::policy::{NetworkPolicy, SandboxLevel, SandboxPolicy};
use nix::libc;

/// Apply Linux sandbox enforcement: rlimits → NO_NEW_PRIVS → Landlock → seccomp.
///
/// Order matters: seccomp must be last because it blocks syscalls that
/// Landlock setup requires.
pub fn apply_sandbox(policy: &SandboxPolicy) -> Result<(), String> {
    // 1. Set PR_SET_NO_NEW_PRIVS (required for both Landlock and seccomp)
    set_no_new_privs()?;

    // 2. Apply Landlock filesystem rules
    // Fail-closed: Standard/Full levels require Landlock to succeed.
    match apply_landlock(policy) {
        Ok(()) => {}
        Err(e) if policy.level >= SandboxLevel::Standard => {
            return Err(format!(
                "Landlock required for {:?} level but failed: {}",
                policy.level, e
            ));
        }
        Err(e) => {
            eprintln!(
                "localgpt-sandbox: landlock not applied (non-fatal at {:?}): {}",
                policy.level, e
            );
        }
    }

    // 3. Apply seccomp network deny filter (must be last)
    // Fail-closed: Minimal+ levels require seccomp when network=deny.
    if policy.network == NetworkPolicy::Deny {
        match apply_seccomp_network_deny() {
            Ok(()) => {}
            Err(e) if policy.level >= SandboxLevel::Minimal => {
                return Err(format!(
                    "seccomp required for {:?} level but failed: {}",
                    policy.level, e
                ));
            }
            Err(e) => {
                eprintln!("localgpt-sandbox: seccomp not applied (non-fatal): {}", e);
            }
        }
    }

    Ok(())
}

fn set_no_new_privs() -> Result<(), String> {
    // SAFETY: prctl(PR_SET_NO_NEW_PRIVS, 1) is a well-defined Linux syscall
    // that restricts the calling thread from gaining elevated privileges.
    // It takes no pointers and has no memory safety implications.
    let ret = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    if ret != 0 {
        return Err(format!(
            "PR_SET_NO_NEW_PRIVS failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

/// Apply Landlock filesystem rules.
///
/// Uses BestEffort ABI negotiation (V5→V1) so rules degrade gracefully
/// on older kernels.
fn apply_landlock(policy: &SandboxPolicy) -> Result<(), String> {
    use landlock::{
        Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
        RulesetStatus, ABI,
    };

    // Try best available ABI
    let abi = ABI::V5;

    let read_access = AccessFs::ReadFile | AccessFs::ReadDir | AccessFs::Execute;
    let write_access = read_access
        | AccessFs::WriteFile
        | AccessFs::RemoveFile
        | AccessFs::RemoveDir
        | AccessFs::MakeReg
        | AccessFs::MakeDir
        | AccessFs::MakeSym;

    let mut ruleset = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))
        .map_err(|e| format!("Landlock ruleset creation: {}", e))?
        .create()
        .map_err(|e| format!("Landlock ruleset create: {}", e))?;

    // Read-only paths (system dirs)
    for path in &policy.read_only_paths {
        if path.exists() {
            if let Ok(fd) = PathFd::new(path) {
                let _ = (&mut ruleset).add_rule(PathBeneath::new(fd, read_access));
            }
        }
    }

    // Workspace — read+write (critical: failure here is fatal)
    if policy.workspace_path.exists() {
        let fd = PathFd::new(&policy.workspace_path).map_err(|e| {
            format!(
                "Landlock: cannot open workspace {}: {}",
                policy.workspace_path.display(),
                e
            )
        })?;
        (&mut ruleset)
            .add_rule(PathBeneath::new(fd, write_access))
            .map_err(|e| format!("Landlock: cannot grant workspace write access: {}", e))?;
    }

    // Extra writable paths (/tmp, user-configured)
    for path in &policy.extra_write_paths {
        if path.exists() {
            if let Ok(fd) = PathFd::new(path) {
                let _ = (&mut ruleset).add_rule(PathBeneath::new(fd, write_access));
            }
        }
    }

    // Note: deny_paths are enforced by omission — we simply don't add rules for them.
    // Since Landlock is deny-by-default once a ruleset handles an access type,
    // any path not explicitly allowed is automatically denied.

    let status = ruleset
        .restrict_self()
        .map_err(|e| format!("Landlock restrict_self: {}", e))?;

    match status.ruleset {
        RulesetStatus::FullyEnforced => {}
        RulesetStatus::PartiallyEnforced => {
            eprintln!("localgpt-sandbox: Landlock partially enforced (ABI downgrade)");
        }
        RulesetStatus::NotEnforced => {
            return Err("Landlock not enforced by kernel".to_string());
        }
    }

    Ok(())
}

/// Apply seccomp-bpf filter that denies network-related syscalls with EPERM.
fn apply_seccomp_network_deny() -> Result<(), String> {
    use seccompiler::{BpfProgram, SeccompAction, SeccompFilter, SeccompRule, TargetArch};
    use std::collections::BTreeMap;

    // Network-related syscalls to deny
    let denied_syscalls: Vec<i64> = vec![
        libc::SYS_socket,
        libc::SYS_connect,
        libc::SYS_accept,
        libc::SYS_accept4,
        libc::SYS_bind,
        libc::SYS_listen,
        libc::SYS_sendto,
        libc::SYS_sendmsg,
        libc::SYS_sendmmsg,
        libc::SYS_recvfrom,
        libc::SYS_recvmsg,
        libc::SYS_recvmmsg,
        libc::SYS_ptrace,
    ];

    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
    for syscall in denied_syscalls {
        let rule = SeccompRule::new(vec![])
            .map_err(|e| format!("seccomp rule for syscall {}: {}", syscall, e))?;
        rules.insert(syscall, vec![rule]);
    }

    let target_arch: TargetArch = std::env::consts::ARCH
        .try_into()
        .map_err(|e: seccompiler::BackendError| format!("seccomp unsupported arch: {}", e))?;

    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow, // Default: allow all other syscalls
        SeccompAction::Errno(libc::EPERM as u32), // Denied syscalls get EPERM
        target_arch,
    )
    .map_err(|e| format!("seccomp filter creation: {}", e))?;

    let bpf: BpfProgram = filter
        .try_into()
        .map_err(|e: seccompiler::BackendError| format!("seccomp BPF compilation: {}", e))?;

    seccompiler::apply_filter(&bpf).map_err(|e| format!("seccomp apply_filter: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use nix::libc;

    // Note: We can't fully test sandbox enforcement in unit tests because
    // they would restrict the test process itself. These tests verify
    // that the construction logic doesn't panic.

    #[test]
    fn test_seccomp_syscall_list_is_valid() {
        // Verify all syscall numbers are positive (valid)
        let syscalls = [
            libc::SYS_socket,
            libc::SYS_connect,
            libc::SYS_accept,
            libc::SYS_accept4,
            libc::SYS_bind,
            libc::SYS_listen,
            libc::SYS_sendto,
            libc::SYS_sendmsg,
            libc::SYS_sendmmsg,
            libc::SYS_recvfrom,
            libc::SYS_recvmsg,
            libc::SYS_recvmmsg,
            libc::SYS_ptrace,
        ];

        for syscall in syscalls {
            assert!(syscall > 0, "Invalid syscall number: {}", syscall);
        }
    }
}

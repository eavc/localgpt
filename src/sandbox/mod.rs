//! Shell sandbox module — kernel-enforced execution isolation for LLM-generated commands.
//!
//! Uses the argv[0] re-exec pattern: when spawning a sandboxed command, the binary
//! re-executes itself with argv[0]="localgpt-sandbox", triggering sandbox setup in
//! a clean, single-threaded child process before exec'ing bash.
//!
//! The sandbox policy is passed via stdin pipe (not argv) to avoid leaking
//! configuration through `/proc/<pid>/cmdline`. See SEC-20b.
//!
//! Platform enforcement:
//! - Linux: Landlock LSM (filesystem) + seccomp-bpf (network syscall deny)
//! - macOS: Seatbelt SBPL profiles via sandbox-exec

pub mod child;
pub mod detect;
pub mod executor;
#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
pub mod policy;

pub use child::sandbox_child_main;
pub use detect::{detect_capabilities, SandboxCapabilities};
pub use executor::run_sandboxed;
#[doc(hidden)]
pub use executor::spawn_sandbox_child;
pub use policy::{build_policy, NetworkPolicy, SandboxLevel, SandboxMode, SandboxPolicy};

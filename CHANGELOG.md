# Changelog

## [Unreleased]

### Security

A comprehensive two-phase security audit was performed, covering all tool
execution paths, the HTTP server, session management, sandbox enforcement,
and the web UI. All findings have been resolved.

#### File tools sandboxed to prevent unrestricted filesystem access (SEC-01)

File tools (`read_file`, `write_file`, `edit_file`) now validate all paths
against a `PathSandbox` before any I/O. Paths are canonicalized to resolve
symlinks and `..` traversal, then checked against allowed roots (workspace
directory plus any `tools.allowed_paths` entries from config).

`memory_get` also received defense-in-depth fixes: component-based `..`
rejection and post-canonicalization containment checks prevent traversal
via prefixed paths like `memory/../../etc/passwd`.

Configure additional allowed directories in `config.toml`:

```toml
[tools]
allowed_paths = ["/tmp/localgpt", "~/projects"]
```

#### API key auth and CORS restriction for HTTP server (SEC-02)

The HTTP server previously exposed all API endpoints with no authentication
and wildcard CORS (`allow_origin(Any)`), allowing any website to silently
call the API via JavaScript.

Changes:
- All `/api/*` routes now require a bearer token (`Authorization: Bearer <key>`).
  The embedded web UI authenticates via a session cookie issued on page load
  (loopback bind only).
- CORS restricted to `localhost:<port>` origins only (was `Any`).
- API key auto-generated (UUID v4) on first run and persisted to config.
- Non-loopback bind address triggers a warning log.
- Rate limiting added: configurable via `server.rate_limit_per_minute`
  (default 120 requests/minute, 0 = unlimited).

#### Tool approval enforcement in non-interactive contexts (SEC-03)

Tools now require explicit approval in non-interactive contexts (daemon,
heartbeat). Desktop GUI approval flow wired up with a confirmation dialog.

#### Heartbeat tool restrictions (SEC-04)

Heartbeat mode restricts tool access to a safe subset. File operations are
sandboxed to the workspace via `PathSandbox::new_heartbeat` with symlink
defense.

#### SSRF and data exfiltration prevention in web_fetch (SEC-06, SEC-18)

`web_fetch` validates URLs against a private-IP blocklist (RFC 1918/4193,
link-local, loopback, and cloud metadata endpoints). DNS resolution is
pinned at validation time to prevent TOCTOU/rebinding attacks — each
request builds a per-URL client with `resolve_to_addrs()` using
pre-validated addresses. Redirect hops re-validate and re-pin at each hop.
Fail-closed guard when pinned addresses are present but the URL has no host.

#### Content delimiter spoofing prevention (SEC-07)

Tool output sanitization now detects and neutralizes delimiter token
spoofing in content delimiters.

#### Session ID path traversal prevention (SEC-08)

Session IDs are validated as UUIDs before use in filesystem paths,
preventing directory traversal via crafted session identifiers.

#### Safe SQLite extension loading (SEC-09)

Replaced unsafe `load_extension` with in-process `sqlite-vec`
initialization, eliminating the extension loading attack surface.

#### LLM body logging gated behind config (SEC-10)

Full LLM request/response body logging is now gated behind a config flag
and requires TRACE log level, preventing accidental exposure of sensitive
content in production logs.

#### Session retention and purge (SEC-11)

Session files are purged after a configurable retention period. Stale
entries in `sessions.json` are automatically pruned after session file
purge, with crash-recovery safety.

#### Memory content sanitization (SEC-12)

Memory content loaded into `build_memory_context` is sanitized to prevent
prompt injection via stored memory files.

#### External provider data egress warnings (SEC-13)

Runtime and config-time warnings when data is sent to external LLM
providers. Centralized egress classifier in `src/config/egress.rs` with
warnings surfaced in both the web and desktop UIs.

#### Sandbox credential exfiltration channels closed (SEC-15)

Sandboxed processes now start with a cleared environment (`env_clear()`)
and a minimal allowlist of safe variables. `/proc/self` removed from
Linux read-only sandbox paths to prevent credential harvesting via
`/proc/self/environ`.

#### Sandbox fail-closed enforcement (SEC-16)

When the sandbox is enabled but the platform lacks kernel support (e.g.,
old Linux kernel without Landlock), execution is now refused rather than
silently falling back to unsandboxed mode. `SandboxEnforcement` enum
(Active/FailClosed/Disabled) replaces the previous optional wrapper.
Daemon startup banner shows enforcement status.

#### Session-token cookie auth replacing Sec-Fetch-Site (SEC-17)

Removed query-parameter API key auth fallback and Sec-Fetch-Site header
trust. The web UI now authenticates via `HttpOnly; SameSite=Strict`
session cookies issued on page load (loopback only). Configurable session
TTL via `server.session_ttl_secs` (default 86400). In-memory session store
with periodic cleanup and 1000-session cap.

#### XSS prevention in web UI (SEC-19)

Applied `escapeHtml()` to all dynamic-value `innerHTML` sinks in the
embedded web UI.

#### Sandbox policy transport via stdin (SEC-20)

Sandbox policy is now passed to child processes via stdin pipe instead of
command-line arguments, preventing policy field exposure in
`/proc/<pid>/cmdline`. API key prefix removed from daemon startup banner.
`/proc` entries rejected from sandbox `allow_paths` to prevent
re-introduction of credential exfiltration paths.

#### Kernel-enforced shell sandbox for bash tool

Shell commands executed by the `bash` tool now run inside a kernel-enforced
sandbox. On Linux, Landlock LSM restricts filesystem access (workspace +
`/tmp` writable, system dirs read-only, credential directories denied) and
seccomp-bpf blocks network syscalls. On macOS, Seatbelt profiles provide
equivalent compile-time support.

The sandbox uses an argv[0] re-exec pattern: the parent forks itself as
`localgpt-sandbox`, and the child applies rlimits, Landlock rules, and
seccomp filters before exec'ing bash. Enforcement is fail-closed at
Standard+ (Landlock) and Minimal+ (seccomp) levels.

Configure in `config.toml`:

```toml
[sandbox]
enabled = true
level = "auto"        # auto | full | standard | minimal | none
timeout_secs = 30
```

Credential directories (`~/.ssh`, `~/.aws`, `~/.gnupg`, etc.) are always
denied. User-provided `allow_paths` that overlap credential directories
are automatically pruned.

### Changed

- CI now runs on both `main` and `development` branches, with an explicit
  step for Linux-only SEC-20b integration tests (`sandbox_stdin_transport`).

### Fixed

#### OpenAI-compatible provider: tool calls silently dropped during streaming

**Problem**

When using the OpenAI provider with a local llama-server backend (or any
OpenAI-compatible endpoint), tool calls are never executed. The model
reports its available tools correctly, but when asked to use one it emits
raw XML-like text instead of producing structured tool calls:

```
LocalGPT: <tool_call>
<bash>
pwd
</tool_call>
</tool_call>
```

The tools are never executed. The session transcript confirms the
response arrives as plain text content, not as a `tool_calls` array:

```json
{
  "content": [
    {
      "text": "<tool_call>\n<bash>\npwd\n</tool_call>\n</tool_call>",
      "type": "text"
    }
  ],
  "role": "assistant"
}
```

However, hitting llama-server directly via curl with the same tool
schemas returns a correctly structured response with
`"finish_reason": "tool_calls"` and a valid `tool_calls` array.

**Diagnosis**

The `OpenAIProvider` implements `chat()` and `summarize()` but does not
implement `chat_stream()`. The interactive chat CLI uses the streaming
path (`agent.chat_stream_with_images()`), which falls through to the
default `chat_stream` implementation on the `LLMProvider` trait.

Two bugs in the default fallback (`src/agent/providers.rs`, lines
152-173):

1. **Tools are dropped.** The fallback calls `self.chat(messages, None)`
   — passing `None` for the tools parameter instead of forwarding the
   tools it received. The model never sees tool schemas in the API
   request, so it cannot produce structured `tool_calls` responses. It
   falls back to emitting its training-time tool format as raw text.

2. **ToolCalls response is treated as an error.** If `chat()` were to
   return `LLMResponseContent::ToolCalls`, the fallback returns
   `Err("Tool calls not supported in streaming")` instead of converting
   the tool calls into a `StreamChunk` with the `tool_calls` field
   populated.

The combination means: the model never receives tool schemas (bug 1),
and even if it did, the response would be discarded (bug 2).

**Impact**

Any provider that relies on the default `chat_stream` fallback — which
currently includes `OpenAIProvider` — cannot execute tools when used via
the interactive chat CLI. This affects all OpenAI-compatible local
backends (llama-server, vLLM, LM Studio, etc.) and the OpenAI API
itself.

The Anthropic and Ollama providers are unaffected because they implement
their own `chat_stream`. (Ollama separately drops tools intentionally.)

**Fix**

1. Forward the `tools` parameter in the default `chat_stream` fallback.
2. Convert `ToolCalls` responses into a `StreamChunk` with `tool_calls`
   populated instead of returning an error.

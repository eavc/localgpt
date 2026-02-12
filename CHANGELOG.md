# Changelog

## [Unreleased]

### Security

#### API key auth and CORS restriction for HTTP server (SEC-02)

The HTTP server previously exposed all API endpoints with no authentication
and wildcard CORS (`allow_origin(Any)`), allowing any website to silently
call the API via JavaScript.

Changes:
- All `/api/*` routes now require a bearer token (`Authorization: Bearer <key>`)
  or a valid `api_key` query parameter. The embedded web UI bypasses auth
  automatically via `Sec-Fetch-Site: same-origin` (loopback bind only).
- CORS restricted to `localhost:<port>` origins only (was `Any`).
- API key auto-generated (UUID v4) on first run and persisted to config.
- Non-loopback bind address triggers a warning log.
- Rate limiting added: configurable via `server.rate_limit_per_minute`
  (default 120 requests/minute, 0 = unlimited).

#### Kernel-enforced shell sandbox for bash tool (localgpt-359)

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

# hermes-agent-acp-bridge

A Rust acp-bridge of [hermes-agent](https://github.com/NousResearch/hermes-agent), providing an OpenAI-compatible HTTP API
that talks to ACP (Agent Client Protocol) agents like CodeBuddy and Copilot via stdio.

## Features vs TypeScript version

- Same config format (`~/.config/acp-bridge/config.json`)
- Same API endpoints
- Lower memory, faster startup (~5ms vs ~300ms)
- Real usage token counts in responses (4-char/token estimate)
- Concurrent model listing via `tokio::task::spawn_blocking`
- Session state in `DashMap` (lock-free concurrent hashmap)

## Build

```bash
cargo build --release
# binary: target/release/hab
```

## Config (~/.config/hermes-agent-acp-bridge/config.json)

```json
{
  "port": 7800,
  "agents": {
    "codebuddy": { "command": "codebuddy", "args": ["--acp"] },
    "copilot": {
      "command": "copilot",
      "args": ["--acp", "--stdio"],
      "env": {
        "HTTPS_PROXY": "http://127.0.0.1:1087",
        "HTTP_PROXY": "http://127.0.0.1:1087"
      }
    }
  }
}
```

## API

```
GET  /health
GET  /v1/models
POST /v1/chat/completions          streaming SSE, OpenAI chunk format
POST /v1/completions               non-streaming, returns usage tokens
POST /v1/sessions                  create persistent session (body: {"model": "acp/codebuddy"})
GET  /v1/sessions                  list active sessions
DELETE /v1/sessions/{id}           close session
GET  /v1/sessions/{id}/permission  check for pending permission request
POST /v1/sessions/{id}/approve     approve permission
POST /v1/sessions/{id}/deny        deny permission
```

## Model format

```
acp/{agent}                    e.g. acp/codebuddy
acp/{agent}/{model-name}       e.g. acp/codebuddy/Claude-Sonnet-4.6
```

## Optimization notes

1. Each ACP call runs in `spawn_blocking` + single-thread tokio + `LocalSet`
   (required by ACP SDK's `?Send` Client trait)
2. `DashMap` for session store — no global mutex
3. `/v1/models` fans out queries concurrently with `spawn_blocking` per agent
4. Streaming uses `async-stream` + `Body::from_stream` — zero-copy SSE
5. Token estimation: `ceil(len / 4)` — cheap, avoids tokenizer dependency

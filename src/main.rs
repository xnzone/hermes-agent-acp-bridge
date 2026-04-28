mod acp;
mod agent_pool;
mod config;
mod session;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

use axum::{
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

use crate::{
    acp::{estimate_tokens, query_models, ChatMessage, ExtractedToolCall},
    agent_pool::{AgentPool, StreamEvent},
    config::Config,
    session::{new_store, Session, SessionStore},
};

// ─── App State ───────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    pool: Arc<AgentPool>,
    sessions: SessionStore,
    // (timestamp_secs, cached_models) — TTL 60s
    models_cache: Arc<Mutex<Option<(u64, Vec<Value>)>>>,
}

// ─── Request/Response types ───────────────────────────────────────────────────

#[derive(Deserialize, Serialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Option<Vec<ChatMessageReq>>,
    prompt: Option<String>,
    session_id: Option<String>,
    stream: Option<bool>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
}

#[derive(Deserialize, Debug, Serialize)]
struct ChatMessageReq {
    role: String,
    #[allow(dead_code)]
    content: Option<Value>,
    tool_calls: Option<Vec<ToolCallReq>>,
    #[allow(dead_code)]
    tool_call_id: Option<String>,
    #[allow(dead_code)]
    name: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Serialize)]
struct ToolCallReq {
    id: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    tc_type: Option<String>,
    function: FunctionReq,
}

#[derive(Deserialize, Debug, Clone, Serialize)]
struct FunctionReq {
    name: String,
    arguments: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct CreateSessionRequest {
    model: String,
}

#[derive(Deserialize, Serialize)]
struct ApproveRequest {
    outcome: Option<String>,
}

// ─── Parse model string ───────────────────────────────────────────────────────

fn parse_model(model: &str) -> (String, Option<String>) {
    if let Some(rest) = model.strip_prefix("acp/") {
        if let Some(slash) = rest.find('/') {
            let agent = rest[..slash].to_string();
            let model_id = rest[slash + 1..].to_string();
            return (agent, Some(model_id));
        }
        return (rest.to_string(), None);
    }
    (model.to_string(), None)
}

/// 将请求中的 ChatMessageReq 转换为内部 ChatMessage
fn convert_messages(req_msgs: &[ChatMessageReq]) -> Vec<ChatMessage> {
    req_msgs
        .iter()
        .map(|m| {
            let content = m.content.clone();
            let tool_calls = m.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| ExtractedToolCall {
                        id: tc.id.clone(),
                        call_id: tc.id.clone(),
                        r#type: "function".to_string(),
                        function: crate::acp::ExtractedFunction {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone().unwrap_or_else(|| "{}".to_string()),
                        },
                    })
                    .collect()
            });
            ChatMessage {
                role: m.role.clone(),
                content,
                tool_calls,
                tool_call_id: m.tool_call_id.clone(),
                name: m.name.clone(),
            }
        })
        .collect()
}

fn extract_prompt(body: &ChatCompletionRequest) -> Option<Vec<ChatMessage>> {
    if let Some(messages) = &body.messages {
        if !messages.is_empty() {
            return Some(convert_messages(messages));
        }
    }
    if let Some(p) = &body.prompt {
        let s = p.trim();
        if !s.is_empty() {
            return Some(vec![ChatMessage {
                role: "user".to_string(),
                content: Some(Value::String(s.to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }]);
        }
    }
    None
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

async fn health() -> Json<Value> {
    Json(json!({ "ok": true }))
}

// GET /v1/models — 并发查询所有 agent 内部模型（结果缓存 60s）
async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let now = now_secs();

    // 检查缓存
    {
        let cache = state.models_cache.lock().await;
        if let Some((ts, ref data)) = *cache {
            if now - ts < 60 {
                return Json(json!({ "object": "list", "data": data }));
            }
        }
    }

    // 缓存未命中 — 并发查询所有 agent
    let cfg = state.config.clone();
    let agent_names = cfg.agent_names();

    let handles: Vec<_> = agent_names
        .into_iter()
        .map(|name| {
            let cfg_clone = cfg.clone();
            tokio::task::spawn_blocking(move || {
                let resolved = cfg_clone.resolve_agent(&name);
                query_models(&resolved, &name, 200)
            })
        })
        .collect();

    let results = futures::future::join_all(handles).await;

    let mut data: Vec<Value> = vec![];
    for res in results {
        if let Ok(models) = res {
            for (id, ctx) in models {
                data.push(json!({
                    "id": id,
                    "object": "model",
                    "created": now,
                    "owned_by": "haab",
                    "context_window": ctx,
                }));
            }
        }
    }

    // 写入缓存
    {
        let mut cache = state.models_cache.lock().await;
        *cache = Some((now, data.clone()));
    }

    Json(json!({ "object": "list", "data": data }))
}

// POST /v1/sessions — 创建持久 session（预热 agent）
async fn create_session(
    State(state): State<AppState>,
    Json(body): Json<CreateSessionRequest>,
) -> impl IntoResponse {
    tracing::info!("[create_session] request body: {}", serde_json::to_string(&body).unwrap_or_else(|e| format!("<serialize error: {e}>")));

    let (agent_type, model_id) = parse_model(&body.model);
    let sessions = state.sessions.clone();

    // 通过 AgentPool 触发 agent 启动和 session 创建
    let (chunk_tx, _chunk_rx) = tokio::sync::mpsc::unbounded_channel::<StreamEvent>();
    let result = state
        .pool
        .prompt(&agent_type, model_id.as_deref(), String::new(), chunk_tx)
        .await;

    match result {
        Ok(pr) => {
            // 获取 config_options
            let config_opts = state
                .pool
                .get_session_info(&agent_type, model_id.as_deref())
                .await
                .map(|(_, opts)| opts)
                .unwrap_or_default();

            let mut sess = Session::new(&agent_type, model_id);
            sess.acp_session_id = pr.session_id.clone();
            sess.config_options = config_opts;
            let sess_id = sess.id.clone();
            let created_at = sess.created_at;
            let display_model = sess.display_model();
            sessions.insert(sess_id.clone(), sess);

            let resp = json!({
                "id": sess_id,
                "object": "session",
                "model": display_model,
                "agent": agent_type,
                "created_at": created_at,
            });
            tracing::info!("[create_session] response body: {resp}");

            (StatusCode::CREATED, Json(resp)).into_response()
        }
        Err(e) => {
            let msg = e.to_string();
            let resp = json!({ "error": { "message": msg, "type": "server_error" } });
            tracing::error!("[create_session] response error: {resp}");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(resp)).into_response()
        }
    };
}

// GET /v1/sessions
async fn list_sessions(State(state): State<AppState>) -> Json<Value> {
    let data: Vec<Value> = state
        .sessions
        .iter()
        .map(|e| {
            let s = e.value();
            json!({
                "id": s.id,
                "object": "session",
                "agent": s.agent_type,
                "model": s.display_model(),
                "created_at": s.created_at,
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

// DELETE /v1/sessions/{id}
async fn delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if state.sessions.remove(&id).is_some() {
        Json(json!({ "id": id, "deleted": true })).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": { "message": "session not found", "type": "not_found" } })),
        )
            .into_response()
    }
}

// GET /v1/sessions/{id}/permission
async fn get_permission(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.sessions.get(&id) {
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": { "message": "session not found" } })),
        )
            .into_response(),
        Some(sess) => {
            if let Some((params, _)) = &sess.pending_permission {
                Json(json!({ "pending": true, "permission": params })).into_response()
            } else {
                Json(json!({ "pending": false })).into_response()
            }
        }
    }
}

// POST /v1/sessions/{id}/approve
async fn approve_permission(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: Option<Json<ApproveRequest>>,
) -> impl IntoResponse {
    let _outcome = body
        .as_ref()
        .and_then(|b| b.outcome.as_deref())
        .unwrap_or("approved");

    match state.sessions.get_mut(&id) {
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": { "message": "session not found" } })),
        )
            .into_response(),
        Some(mut sess) => {
            if let Some((_, tx)) = sess.pending_permission.take() {
                let _ = tx.send(true);
                Json(json!({ "ok": true, "outcome": "approved" })).into_response()
            } else {
                (
                    StatusCode::CONFLICT,
                    Json(json!({ "error": { "message": "no pending permission request" } })),
                )
                    .into_response()
            }
        }
    }
}

// POST /v1/sessions/{id}/deny
async fn deny_permission(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.sessions.get_mut(&id) {
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": { "message": "session not found" } })),
        )
            .into_response(),
        Some(mut sess) => {
            if let Some((_, tx)) = sess.pending_permission.take() {
                let _ = tx.send(false);
                Json(json!({ "ok": true, "outcome": "denied" })).into_response()
            } else {
                (
                    StatusCode::CONFLICT,
                    Json(json!({ "error": { "message": "no pending permission request" } })),
                )
                    .into_response()
            }
        }
    }
}

// POST /v1/chat/completions — stream=true 走 SSE，否则走非流式 JSON
async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    tracing::info!("[chat_completions] request body: {}", serde_json::to_string(&body).unwrap_or_else(|e| format!("<serialize error: {e}>")));

    let is_stream = body.stream.unwrap_or(false);

    let messages = match extract_prompt(&body) {
        Some(m) => m,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": { "message": "messages or prompt required" } })),
            )
                .into_response();
        }
    };

    let (agent_type, model_id) = if let Some(sid) = &body.session_id {
        match state.sessions.get(sid.as_str()) {
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({
                        "error": { "message": format!("session {sid} not found") }
                    })),
                )
                    .into_response();
            }
            Some(sess) => (sess.agent_type.clone(), sess.model_id.clone()),
        }
    } else {
        let raw_model = match &body.model {
            Some(m) if !m.trim().is_empty() => m.trim().to_string(),
            _ => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": { "message": "model or session_id required" } })),
                )
                    .into_response();
            }
        };
        let (agent, model) = parse_model(&raw_model);
        (agent, model)
    };

    // 构建 prompt 文本（复用 acp.rs 中的 build_prompt 逻辑）
    let prompt_text = crate::acp::build_prompt(&messages, body.tools.as_deref(), body.tool_choice.as_ref());

    if is_stream {
        chat_completions_stream(state, prompt_text, agent_type, model_id).await
    } else {
        chat_completions_json(state, prompt_text, agent_type, model_id).await
    }
}

// ── 非流式 JSON 响应 ──────────────────────────────────────────────────────────
async fn chat_completions_json(
    state: AppState,
    prompt_text: String,
    agent_type: String,
    model_id: Option<String>,
) -> axum::response::Response {
    let display_model = match &model_id {
        Some(m) => format!("acp/{agent_type}/{m}"),
        None => format!("acp/{agent_type}"),
    };

    let prompt_tokens = estimate_tokens(&prompt_text);

    // 创建 chunk channel（非流式不需要，但 AgentPool::prompt 要求一个）
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::unbounded_channel::<StreamEvent>();

    let result = state
        .pool
        .prompt(&agent_type, model_id.as_deref(), prompt_text, chunk_tx)
        .await;

    // 收集所有 chunk
    let mut collected_text = String::new();
    let mut collected_thought = String::new();
    while let Some(event) = chunk_rx.recv().await {
        match event {
            StreamEvent::TextChunk(chunk) => collected_text.push_str(&chunk),
            StreamEvent::ThoughtChunk(chunk) => collected_thought.push_str(&chunk),
        }
    }

    match result {
        Ok(_pr) => {
            let (tool_calls, cleaned_text) = crate::acp::extract_tool_calls(&collected_text);
            let ctokens = estimate_tokens(&cleaned_text);

            let has_tool_calls = !tool_calls.is_empty();
            let tool_calls_json: Vec<Value> = tool_calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    })
                })
                .collect();

            let finish_reason = if has_tool_calls { "tool_calls" } else { "stop" };

            let mut message = json!({
                "role": "assistant",
                "content": if cleaned_text.is_empty() { Value::Null } else { Value::String(cleaned_text) },
            });
            if has_tool_calls {
                message["tool_calls"] = Value::Array(tool_calls_json);
            }

            if !collected_thought.is_empty() {
                message["reasoning"] = Value::String(collected_thought.clone());
                message["reasoning_content"] = Value::String(collected_thought);
            }

            let resp = json!({
                "id": format!("chatcmpl-{}", Uuid::new_v4()),
                "object": "chat.completion",
                "created": now_secs(),
                "model": display_model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": ctokens,
                    "total_tokens": prompt_tokens + ctokens,
                },
            });

            Json(resp).into_response()
        }
        Err(e) => {
            let msg = e.to_string();
            let resp = json!({ "error": { "message": msg } });
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(resp),
            )
                .into_response()
        }
    }
}

// ── 流式 SSE 响应 ─────────────────────────────────────────────────────────────
async fn chat_completions_stream(
    state: AppState,
    prompt_text: String,
    agent_type: String,
    model_id: Option<String>,
) -> axum::response::Response {
    let cmpl_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_secs();
    let display_model = match &model_id {
        Some(m) => format!("acp/{agent_type}/{m}"),
        None => format!("acp/{agent_type}"),
    };
    let prompt_tokens = estimate_tokens(&prompt_text);

    // 创建 chunk channel — agent 线程通过这个 channel 实时推送 chunk
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::unbounded_channel::<StreamEvent>();

    // 异步发送 prompt（不等待完成，让 chunk 流过来）
    let pool = state.pool.clone();
    let _prompt_result_handle = tokio::spawn(async move {
        pool.prompt(&agent_type, model_id.as_deref(), prompt_text, chunk_tx)
            .await
    });

    // 构建 SSE stream
    let cmpl_id_clone = cmpl_id.clone();
    let display_model_clone = display_model.clone();

    let stream = async_stream::stream! {
        // 发送 role chunk
        let role_chunk = json!({
            "id": cmpl_id_clone,
            "object": "chat.completion.chunk",
            "created": created,
            "model": display_model_clone,
            "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "" }, "finish_reason": null }],
        });
        yield Ok::<String, std::convert::Infallible>(format!("data: {}\n\n", role_chunk));

        let mut total_chars = 0usize;
        let mut had_thinking = false;
        let mut thinking_ended = false;

        loop {
            match chunk_rx.recv().await {
                Some(StreamEvent::ThoughtChunk(chunk)) => {
                    if !had_thinking {
                        had_thinking = true;
                    }
                    let payload = json!({
                        "id": cmpl_id_clone,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": display_model_clone,
                        "choices": [{ "index": 0, "delta": { "reasoning_content": chunk }, "finish_reason": null }],
                    });
                    yield Ok(format!("data: {}\n\n", payload));
                }
                Some(StreamEvent::TextChunk(chunk)) => {
                    // 如果还有未结束的 thinking，先结束它
                    if had_thinking && !thinking_ended {
                        thinking_ended = true;
                        let thought_end = json!({
                            "id": cmpl_id_clone,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": display_model_clone,
                            "choices": [{ "index": 0, "delta": { "reasoning_content": "" }, "finish_reason": null }],
                        });
                        yield Ok(format!("data: {}\n\n", thought_end));
                    }
                    total_chars += chunk.len();
                    let payload = json!({
                        "id": cmpl_id_clone,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": display_model_clone,
                        "choices": [{ "index": 0, "delta": { "content": chunk }, "finish_reason": null }],
                    });
                    yield Ok(format!("data: {}\n\n", payload));
                }
                None => {
                    // channel 关闭 = agent 完成了输出
                    break;
                }
            }
        }

        // thinking 未关闭时补上结束标记
        if had_thinking && !thinking_ended {
            let thought_end = json!({
                "id": cmpl_id_clone,
                "object": "chat.completion.chunk",
                "created": created,
                "model": display_model_clone,
                "choices": [{ "index": 0, "delta": { "reasoning_content": "" }, "finish_reason": null }],
            });
            yield Ok(format!("data: {}\n\n", thought_end));
        }

        let completion_tokens = estimate_tokens(&"x".repeat(total_chars));
        let stop_chunk = json!({
            "id": cmpl_id_clone,
            "object": "chat.completion.chunk",
            "created": created,
            "model": display_model_clone,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        });
        yield Ok(format!("data: {}\n\n", stop_chunk));
        yield Ok("data: [DONE]\n\n".to_string());
    };

    let mut headers = HeaderMap::new();
    headers.insert(
        "content-type",
        "text/event-stream; charset=utf-8".parse().unwrap(),
    );
    headers.insert("cache-control", "no-cache".parse().unwrap());
    headers.insert("connection", "keep-alive".parse().unwrap());

    (headers, Body::from_stream(stream)).into_response()
}

// POST /v1/completions — 非流式（支持 tool_calls 提取）
async fn completions(
    State(state): State<AppState>,
    Json(body): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let messages = match extract_prompt(&body) {
        Some(m) => m,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": { "message": "messages or prompt required" } })),
            )
                .into_response();
        }
    };

    let (agent_type, model_id) = if let Some(sid) = &body.session_id {
        match state.sessions.get(sid.as_str()) {
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": { "message": format!("session {sid} not found") } })),
                )
                    .into_response();
            }
            Some(sess) => (sess.agent_type.clone(), sess.model_id.clone()),
        }
    } else {
        let raw_model = match &body.model {
            Some(m) if !m.trim().is_empty() => m.trim().to_string(),
            _ => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({ "error": { "message": "model or session_id required" } })),
                )
                    .into_response();
            }
        };
        let (agent, model) = parse_model(&raw_model);
        (agent, model)
    };

    let prompt_text = crate::acp::build_prompt(&messages, body.tools.as_deref(), body.tool_choice.as_ref());
    let display_model = match &model_id {
        Some(m) => format!("acp/{agent_type}/{m}"),
        None => format!("acp/{agent_type}"),
    };
    let prompt_tokens = estimate_tokens(&prompt_text);

    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::unbounded_channel::<StreamEvent>();
    let result = state
        .pool
        .prompt(&agent_type, model_id.as_deref(), prompt_text, chunk_tx)
        .await;

    let mut collected_text = String::new();
    let mut collected_thought = String::new();
    while let Some(event) = chunk_rx.recv().await {
        match event {
            StreamEvent::TextChunk(chunk) => collected_text.push_str(&chunk),
            StreamEvent::ThoughtChunk(chunk) => collected_thought.push_str(&chunk),
        }
    }

    match result {
        Ok(_pr) => {
            let (tool_calls, cleaned_text) = crate::acp::extract_tool_calls(&collected_text);
            let ctokens = estimate_tokens(&cleaned_text);

            let has_tool_calls = !tool_calls.is_empty();
            let tool_calls_json: Vec<Value> = tool_calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    })
                })
                .collect();

            let finish_reason = if has_tool_calls { "tool_calls" } else { "stop" };

            let mut message = json!({
                "role": "assistant",
                "content": if cleaned_text.is_empty() { Value::Null } else { Value::String(cleaned_text) },
            });
            if has_tool_calls {
                message["tool_calls"] = Value::Array(tool_calls_json);
            }

            if !collected_thought.is_empty() {
                message["reasoning"] = Value::String(collected_thought.clone());
                message["reasoning_content"] = Value::String(collected_thought);
            }

            let resp = json!({
                "id": format!("chatcmpl-{}", Uuid::new_v4()),
                "object": "chat.completion",
                "created": now_secs(),
                "model": display_model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": ctokens,
                    "total_tokens": prompt_tokens + ctokens,
                },
            });

            Json(resp).into_response()
        }
        Err(e) => {
            let msg = e.to_string();
            let resp = json!({ "error": { "message": msg } });
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(resp),
            )
                .into_response()
        }
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    let config = Arc::new(Config::load());
    let pool = Arc::new(AgentPool::new(config.clone()));
    let sessions = new_store();

    let state = AppState {
        config: config.clone(),
        pool,
        sessions,
        models_cache: Arc::new(Mutex::new(None)),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_headers(Any)
        .allow_methods(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/sessions", post(create_session).get(list_sessions))
        .route("/v1/sessions/{id}", delete(delete_session))
        .route("/v1/sessions/{id}/permission", get(get_permission))
        .route("/v1/sessions/{id}/approve", post(approve_permission))
        .route("/v1/sessions/{id}/deny", post(deny_permission))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(cors)
        .with_state(state);

    let port = std::env::var("HAAB_PORT")
        .or_else(|_| std::env::var("ACP_SERVE_PORT"))
        .or_else(|_| std::env::var("ACP_BRIDGE_PORT"))
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or_else(|| config.port.unwrap_or(7800));

    let host = std::env::var("HAB_HOST")
        .or_else(|_| std::env::var("ACP_SERVE_HOST"))
        .or_else(|_| std::env::var("ACP_BRIDGE_HOST"))
        .unwrap_or_else(|_| {
            config
                .host
                .clone()
                .unwrap_or_else(|| "127.0.0.1".to_string())
        });

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

    let agents = config.agent_names();
    eprintln!();
    eprintln!("╔══════════════════════════════════════╗");
    eprintln!("║   hermes-agent-acp-bridge  v0.2.0    ║");
    eprintln!("║   ACP → OpenAI Protocol Gateway      ║");
    eprintln!("╚══════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Listening:  http://{addr}");
    eprintln!("  Agents:     {}", agents.join(", "));
    eprintln!();
    eprintln!("  GET  /v1/models");
    eprintln!("  POST /v1/chat/completions   (stream=true→SSE, otherwise JSON; tools+skills support)");
    eprintln!("  POST /v1/completions        (non-streaming, tool_calls extraction)");
    eprintln!("  POST /v1/sessions           (create persistent session)");
    eprintln!("  GET  /v1/sessions           (list sessions)");
    eprintln!("  DELETE /v1/sessions/{{id}}    (close session)");
    eprintln!("  GET  /v1/sessions/{{id}}/permission");
    eprintln!("  POST /v1/sessions/{{id}}/approve");
    eprintln!("  POST /v1/sessions/{{id}}/deny");
    eprintln!();

    axum::serve(listener, app).await.unwrap();
}

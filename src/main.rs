mod acp;
mod config;
mod session;

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
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
    acp::{estimate_tokens, query_models, run_prompt, ChatMessage, ExtractedToolCall},
    config::Config,
    session::{new_store, Session, SessionStore},
};

// ─── App State ───────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
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
    let cfg = state.config.clone();
    let sessions = state.sessions.clone();

    let agent_type_clone = agent_type.clone();
    let model_id_clone = model_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        let resolved = cfg.resolve_agent(&agent_type_clone);
        run_prompt(
            &resolved,
            &agent_type_clone,
            None,
            &[],
            None,
            None,
            None,
            model_id_clone.as_deref(),
            None,
            None,
            None,
        )
    })
    .await;

    match result {
        Ok(Ok(run)) => {
            let mut sess = Session::new(&agent_type, model_id);
            sess.acp_session_id = run.session_id;
            sess.config_options = run.config_options;
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

            (
                StatusCode::CREATED,
                Json(resp),
            )
                .into_response()
        }
        Ok(Err(e)) => {
            let resp = json!({ "error": { "message": e.to_string(), "type": "server_error" } });
            tracing::error!("[create_session] response error: {resp}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(resp),
            )
                .into_response()
        }
        Err(e) => {
            let resp = json!({ "error": { "message": e.to_string(), "type": "server_error" } });
            tracing::error!("[create_session] response error: {resp}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(resp),
            )
                .into_response()
        }
    }
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

    let (agent_type, model_id, existing_sess, config_opts, acp_sid) =
        if let Some(sid) = &body.session_id {
            match state.sessions.get(sid.as_str()) {
                None => {
                    return (
                        StatusCode::NOT_FOUND,
                        Json(json!({ "error": { "message": format!("session {sid} not found") } })),
                    )
                        .into_response();
                }
                Some(sess) => (
                    sess.agent_type.clone(),
                    sess.model_id.clone(),
                    Some(sid.clone()),
                    sess.config_options.clone(),
                    Some(sess.acp_session_id.clone()),
                ),
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
            (agent, model, None, vec![], None)
        };

    if is_stream {
        chat_completions_stream(state, body, messages, agent_type, model_id, existing_sess, config_opts, acp_sid).await
    } else {
        chat_completions_json(state, body, messages, agent_type, model_id, existing_sess, config_opts, acp_sid).await
    }
}

// ── 非流式 JSON 响应 ──────────────────────────────────────────────────────────
async fn chat_completions_json(
    state: AppState,
    body: ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    agent_type: String,
    model_id: Option<String>,
    existing_sess: Option<String>,
    config_opts: Vec<serde_json::Value>,
    acp_sid: Option<String>,
) -> axum::response::Response {
    let cfg = state.config.clone();
    let sessions = state.sessions.clone();
    let display_model = match &model_id {
        Some(m) => format!("acp/{agent_type}/{m}"),
        None => format!("acp/{agent_type}"),
    };

    let tools_clone = body.tools.clone();
    let tool_choice_clone = body.tool_choice.clone();

    let result = tokio::task::spawn_blocking(move || {
        let resolved = cfg.resolve_agent(&agent_type);
        let run = run_prompt(
            &resolved,
            &agent_type,
            acp_sid,
            &messages,
            None,
            None,
            None,
            model_id.as_deref(),
            if config_opts.is_empty() {
                None
            } else {
                Some(config_opts)
            },
            tools_clone.as_deref(),
            tool_choice_clone.as_ref(),
        );
        if let Some(sid) = existing_sess {
            if let Ok(ref r) = run {
                if let Some(mut sess) = sessions.get_mut(&sid) {
                    sess.acp_session_id = r.session_id.clone();
                }
            }
        }
        run
    })
    .await;

    match result {
        Ok(Ok(run)) => {
            let ptokens = run.prompt_tokens;
            let ctokens = estimate_tokens(&run.text);

            let has_tool_calls = !run.tool_calls.is_empty();
            let tool_calls_json: Vec<Value> = run.tool_calls.iter().map(|tc| {
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })
            }).collect();

            let finish_reason = if has_tool_calls { "tool_calls" } else { "stop" };

            let mut message = json!({
                "role": "assistant",
                "content": if run.text.is_empty() { Value::Null } else { Value::String(run.text) },
            });
            if has_tool_calls {
                message["tool_calls"] = Value::Array(tool_calls_json);
            }

            if let Some(reasoning) = &run.reasoning {
                message["reasoning"] = Value::String(reasoning.clone());
                message["reasoning_content"] = Value::String(reasoning.clone());
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
                    "prompt_tokens": ptokens,
                    "completion_tokens": ctokens,
                    "total_tokens": ptokens + ctokens,
                },
            });
            tracing::info!("[chat_completions] non-stream response body: {resp}");

            Json(resp).into_response()
        }
        Ok(Err(e)) => {
            let resp = json!({ "error": { "message": e.to_string() } });
            tracing::error!("[chat_completions] non-stream response error: {resp}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(resp),
            )
                .into_response()
        }
        Err(e) => {
            let resp = json!({ "error": { "message": e.to_string() } });
            tracing::error!("[chat_completions] non-stream response error: {resp}");
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
    _body: ChatCompletionRequest,
    messages: Vec<ChatMessage>,
    agent_type: String,
    model_id: Option<String>,
    existing_sess: Option<String>,
    config_opts: Vec<serde_json::Value>,
    acp_sid: Option<String>,
) -> axum::response::Response {
    let cfg = state.config.clone();
    let sessions = state.sessions.clone();
    let sess_id_header = existing_sess.clone();

    // 两个 channel：text chunk 和 thought chunk
    let (text_tx, mut text_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let (thought_tx, mut thought_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let text_tx = Arc::new(text_tx);
    let thought_tx = Arc::new(thought_tx);

    let cmpl_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_secs();
    let display_model = match &model_id {
        Some(m) => format!("acp/{agent_type}/{m}"),
        None => format!("acp/{agent_type}"),
    };

    let model_id_clone = model_id.clone();
    let agent_type_clone = agent_type.clone();

    let prompt_text_for_estimate = messages
        .iter()
        .map(|m| {
            let content_str = m.content
                .as_ref()
                .map(|c| c.as_str().unwrap_or("").to_string())
                .unwrap_or_default();
            format!("{}: {}", m.role, content_str)
        })
        .collect::<Vec<_>>()
        .join("\n");
    let prompt_tokens = estimate_tokens(&prompt_text_for_estimate);

    // 克隆 tools 和 tool_choice 传给 blocking 线程
    let tools_for_blocking = _body.tools.clone();
    let tool_choice_for_blocking = _body.tool_choice.clone();

    // 在 blocking 线程运行 ACP
    tokio::task::spawn_blocking(move || {
        let resolved = cfg.resolve_agent(&agent_type_clone);
        let on_chunk = {
            let text_tx = text_tx.clone();
            Some(Arc::new(move |chunk: String| {
                let _ = text_tx.send(chunk);
            }) as Arc<dyn Fn(String) + Send + Sync>)
        };
        let on_thought_chunk = {
            let thought_tx = thought_tx.clone();
            Some(Arc::new(move |chunk: String| {
                let _ = thought_tx.send(chunk);
            }) as Arc<dyn Fn(String) + Send + Sync>)
        };

        let result = run_prompt(
            &resolved,
            &agent_type_clone,
            acp_sid,
            &messages,
            on_chunk,
            on_thought_chunk,
            None,
            model_id_clone.as_deref(),
            if config_opts.is_empty() {
                None
            } else {
                Some(config_opts)
            },
            tools_for_blocking.as_deref(),
            tool_choice_for_blocking.as_ref(),
        );

        if let Some(sid) = existing_sess {
            if let Ok(ref run) = &result {
                if let Some(mut sess) = sessions.get_mut(&sid) {
                    sess.acp_session_id = run.session_id.clone();
                }
            }
        }

        drop(text_tx);
        drop(thought_tx);
        result
    });

    // 构建 SSE stream
    let cmpl_id_clone = cmpl_id.clone();
    let display_model_clone = display_model.clone();
    let sess_id_for_header = sess_id_header.clone();

    let stream = async_stream::stream! {
        // 发送 role chunk
        let role_chunk = json!({
            "id": cmpl_id_clone,
            "object": "chat.completion.chunk",
            "created": created,
            "model": display_model_clone,
            "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "" }, "finish_reason": null }],
        });
        tracing::info!("[chat_completions] SSE chunk: {role_chunk}");
        yield Ok::<String, std::convert::Infallible>(format!("data: {}\n\n", role_chunk));

        let mut total_chars = 0usize;
        let mut had_thinking = false;
        let mut thinking_ended = false;
        let mut thought_closed = false;
        let mut text_closed = false;

        // keepalive：每 15 秒发一个 SSE comment，防止中间件超时断开
        let mut keepalive = tokio::time::interval(Duration::from_secs(15));
        keepalive.tick().await; // 消耗掉首次立即触发的 tick

        loop {
            tokio::select! {
                // keepalive 心跳
                _ = keepalive.tick() => {
                    // SSE comment 格式：以冒号开头，客户端会忽略
                    yield Ok(": keepalive\n\n".to_string());
                    tracing::debug!("[chat_completions] SSE keepalive sent");
                }
                thought = thought_rx.recv(), if !thought_closed => {
                    match thought {
                        Some(chunk) => {
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
                            tracing::info!("[chat_completions] SSE thinking chunk: {payload}");
                            yield Ok(format!("data: {}\n\n", payload));
                        }
                        None => {
                            thought_closed = true;
                            // thinking 结束标志
                            if had_thinking && !thinking_ended {
                                thinking_ended = true;
                                let thought_end = json!({
                                    "id": cmpl_id_clone,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": display_model_clone,
                                    "choices": [{ "index": 0, "delta": { "reasoning_content": "" }, "finish_reason": null }],
                                });
                                tracing::info!("[chat_completions] SSE thinking end: {thought_end}");
                                yield Ok(format!("data: {}\n\n", thought_end));
                            }
                        }
                    }
                }
                text = text_rx.recv(), if !text_closed => {
                    match text {
                        Some(chunk) => {
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
                                tracing::info!("[chat_completions] SSE thinking end: {thought_end}");
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
                            tracing::info!("[chat_completions] SSE chunk: {payload}");
                            yield Ok(format!("data: {}\n\n", payload));
                        }
                        None => {
                            text_closed = true;
                        }
                    }
                }
                else => break,
            }
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
        tracing::info!("[chat_completions] SSE chunk: {stop_chunk}");
        yield Ok(format!("data: {}\n\n", stop_chunk));
        tracing::info!("[chat_completions] SSE chunk: [DONE]");
        yield Ok("data: [DONE]\n\n".to_string());
    };

    let mut headers = HeaderMap::new();
    headers.insert(
        "content-type",
        "text/event-stream; charset=utf-8".parse().unwrap(),
    );
    headers.insert("cache-control", "no-cache".parse().unwrap());
    headers.insert("connection", "keep-alive".parse().unwrap());
    if let Some(sid) = sess_id_for_header {
        if let Ok(v) = sid.parse() {
            headers.insert("x-session-id", v);
        }
    }

    (headers, Body::from_stream(stream)).into_response()
}

// POST /v1/completions — 非流式（支持 tool_calls 提取）
async fn completions(
    State(state): State<AppState>,
    Json(body): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    tracing::info!("[completions] request body: {}", serde_json::to_string(&body).unwrap_or_else(|e| format!("<serialize error: {e}>")));

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

    let (agent_type, model_id, existing_sess, config_opts, acp_sid) =
        if let Some(sid) = &body.session_id {
            match state.sessions.get(sid.as_str()) {
                None => {
                    return (
                        StatusCode::NOT_FOUND,
                        Json(json!({ "error": { "message": format!("session {sid} not found") } })),
                    )
                        .into_response();
                }
                Some(sess) => (
                    sess.agent_type.clone(),
                    sess.model_id.clone(),
                    Some(sid.clone()),
                    sess.config_options.clone(),
                    Some(sess.acp_session_id.clone()),
                ),
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
            (agent, model, None, vec![], None)
        };

    let cfg = state.config.clone();
    let sessions = state.sessions.clone();
    let display_model = match &model_id {
        Some(m) => format!("acp/{agent_type}/{m}"),
        None => format!("acp/{agent_type}"),
    };

    // 克隆 tools 和 tool_choice
    let tools_clone = body.tools.clone();
    let tool_choice_clone = body.tool_choice.clone();

    let result = tokio::task::spawn_blocking(move || {
        let resolved = cfg.resolve_agent(&agent_type);
        let run = run_prompt(
            &resolved,
            &agent_type,
            acp_sid,
            &messages,
            None,
            None,
            None,
            model_id.as_deref(),
            if config_opts.is_empty() {
                None
            } else {
                Some(config_opts)
            },
            tools_clone.as_deref(),
            tool_choice_clone.as_ref(),
        );
        if let Some(sid) = existing_sess {
            if let Ok(ref r) = run {
                if let Some(mut sess) = sessions.get_mut(&sid) {
                    sess.acp_session_id = r.session_id.clone();
                }
            }
        }
        run
    })
    .await;

    match result {
        Ok(Ok(run)) => {
            let ptokens = run.prompt_tokens;
            let ctokens = estimate_tokens(&run.text);

            // 构建 tool_calls 响应
            let has_tool_calls = !run.tool_calls.is_empty();
            let tool_calls_json: Vec<Value> = run.tool_calls.iter().map(|tc| {
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })
            }).collect();

            let finish_reason = if has_tool_calls { "tool_calls" } else { "stop" };

            let mut message = json!({
                "role": "assistant",
                "content": if run.text.is_empty() { Value::Null } else { Value::String(run.text) },
            });
            if has_tool_calls {
                message["tool_calls"] = Value::Array(tool_calls_json);
            }

            // 添加 reasoning（如果存在）
            if let Some(reasoning) = &run.reasoning {
                message["reasoning"] = Value::String(reasoning.clone());
                message["reasoning_content"] = Value::String(reasoning.clone());
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
                    "prompt_tokens": ptokens,
                    "completion_tokens": ctokens,
                    "total_tokens": ptokens + ctokens,
                },
            });
            tracing::info!("[completions] response body: {resp}");

            Json(resp).into_response()
        }
        Ok(Err(e)) => {
            let resp = json!({ "error": { "message": e.to_string() } });
            tracing::error!("[completions] response error: {resp}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(resp),
            )
                .into_response()
        }
        Err(e) => {
            let resp = json!({ "error": { "message": e.to_string() } });
            tracing::error!("[completions] response error: {resp}");
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
    let sessions = new_store();

    let state = AppState {
        config: config.clone(),
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
    eprintln!("║   hermes-agent-acp-bridge  v0.1.0    ║");
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

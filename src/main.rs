mod acp;
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
use serde::Deserialize;
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

use crate::{
    acp::{estimate_tokens, query_models, run_prompt, ChatMessage},
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

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Option<Vec<ChatMessageReq>>,
    prompt: Option<String>,
    session_id: Option<String>,
    #[allow(dead_code)]
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct ChatMessageReq {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct CreateSessionRequest {
    model: String,
}

#[derive(Deserialize)]
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

fn extract_prompt(body: &ChatCompletionRequest) -> Option<Vec<ChatMessage>> {
    if let Some(messages) = &body.messages {
        if !messages.is_empty() {
            return Some(
                messages
                    .iter()
                    .map(|m| ChatMessage {
                        role: m.role.clone(),
                        content: m.content.clone(),
                    })
                    .collect(),
            );
        }
    }
    if let Some(p) = &body.prompt {
        if !p.trim().is_empty() {
            return Some(vec![ChatMessage {
                role: "user".to_string(),
                content: p.trim().to_string(),
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
                query_models(&resolved, &name, 15)
            })
        })
        .collect();

    // join_all — 所有 handle 真正并发等待
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
    let (agent_type, model_id) = parse_model(&body.model);
    let cfg = state.config.clone();
    let sessions = state.sessions.clone();

    // 在 blocking 线程里建立 ACP session
    let agent_type_clone = agent_type.clone();
    let model_id_clone = model_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        let resolved = cfg.resolve_agent(&agent_type_clone);
        // 新建 session（发 initialize + new_session）
        run_prompt(
            &resolved,
            &agent_type_clone,
            None,     // 新建 acp session
            &[],      // 空 messages，仅为获取 session_id + config_options
            None,
            None,
            model_id_clone.as_deref(),
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

            (
                StatusCode::CREATED,
                Json(json!({
                    "id": sess_id,
                    "object": "session",
                    "model": display_model,
                    "agent": agent_type,
                    "created_at": created_at,
                })),
            )
                .into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": e.to_string(), "type": "server_error" } })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": e.to_string(), "type": "server_error" } })),
        )
            .into_response(),
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

// POST /v1/chat/completions — 流式 SSE
async fn chat_completions(
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
    let sess_id_header = existing_sess.clone();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let tx = Arc::new(tx);

    let cmpl_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_secs();
    let display_model = match &model_id {
        Some(m) => format!("acp/{agent_type}/{m}"),
        None => format!("acp/{agent_type}"),
    };

    let model_id_clone = model_id.clone();
    let agent_type_clone = agent_type.clone();

    // 提前估算 prompt tokens（把所有 messages 拼成文本估算）
    let prompt_text_for_estimate = messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");
    let prompt_tokens = estimate_tokens(&prompt_text_for_estimate);

    // 在 blocking 线程运行 ACP
    tokio::task::spawn_blocking(move || {
        let resolved = cfg.resolve_agent(&agent_type_clone);
        let on_chunk = {
            let tx = tx.clone();
            Some(Arc::new(move |chunk: String| {
                let _ = tx.send(chunk);
            }) as Arc<dyn Fn(String) + Send + Sync>)
        };

        let result = run_prompt(
            &resolved,
            &agent_type_clone,
            acp_sid,
            &messages,
            on_chunk,
            None,
            model_id_clone.as_deref(),
            if config_opts.is_empty() { None } else { Some(config_opts) },
        );

        if let Some(sid) = existing_sess {
            if let Ok(ref run) = &result {
                if let Some(mut sess) = sessions.get_mut(&sid) {
                    sess.acp_session_id = run.session_id.clone();
                }
            }
        }

        // 发完后关闭 channel
        drop(tx);
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
        yield Ok::<String, std::convert::Infallible>(format!("data: {}\n\n", role_chunk));

        let mut total_chars = 0usize;

        while let Some(chunk) = rx.recv().await {
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
    headers.insert("content-type", "text/event-stream; charset=utf-8".parse().unwrap());
    headers.insert("cache-control", "no-cache".parse().unwrap());
    headers.insert("connection", "keep-alive".parse().unwrap());
    if let Some(sid) = sess_id_for_header {
        if let Ok(v) = sid.parse() {
            headers.insert("x-session-id", v);
        }
    }

    (headers, Body::from_stream(stream)).into_response()
}

// POST /v1/completions — 非流式
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

    let result = tokio::task::spawn_blocking(move || {
        let resolved = cfg.resolve_agent(&agent_type);
        let run = run_prompt(
            &resolved,
            &agent_type,
            acp_sid,
            &messages,
            None,
            None,
            model_id.as_deref(),
            if config_opts.is_empty() { None } else { Some(config_opts) },
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
            Json(json!({
                "id": format!("cmpl-{}", Uuid::new_v4()),
                "object": "text_completion",
                "created": now_secs(),
                "model": display_model,
                "choices": [{ "index": 0, "text": run.text, "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": ptokens,
                    "completion_tokens": ctokens,
                    "total_tokens": ptokens + ctokens,
                },
            }))
            .into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": e.to_string() } })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": { "message": e.to_string() } })),
        )
            .into_response(),
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "hab=info,warn".to_string()),
        )
        .init();

    let config = Arc::new(Config::load());
    let sessions = new_store();

    let state = AppState { config: config.clone(), sessions, models_cache: Arc::new(Mutex::new(None)) };

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
        .unwrap_or_else(|_| config.host.clone().unwrap_or_else(|| "127.0.0.1".to_string()));

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
    eprintln!("  POST /v1/chat/completions   (streaming SSE)");
    eprintln!("  POST /v1/completions        (non-streaming, with usage tokens)");
    eprintln!("  POST /v1/sessions           (create persistent session)");
    eprintln!("  GET  /v1/sessions           (list sessions)");
    eprintln!("  DELETE /v1/sessions/{{id}}    (close session)");
    eprintln!("  GET  /v1/sessions/{{id}}/permission");
    eprintln!("  POST /v1/sessions/{{id}}/approve");
    eprintln!("  POST /v1/sessions/{{id}}/deny");
    eprintln!();

    axum::serve(listener, app).await.unwrap();
}

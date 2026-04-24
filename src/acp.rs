/// ACP agent 通信核心
use std::{sync::Arc, time::Duration};

use agent_client_protocol::{
    self as acp,
    Agent,
    ClientSideConnection, ContentBlock, NewSessionRequest,
    PromptRequest, SessionId,
    SessionNotification, SessionUpdate, TextContent,
    InitializeRequest, ProtocolVersion,
    SetSessionConfigOptionRequest,
};
use anyhow::{Context, Result};

use agent_client_protocol::Error as AcpError;
type AcpResult<T> = std::result::Result<T, AcpError>;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::config::ResolvedAgent;

// ─── 权限回调类型 ──────────────────────────────────────────────────────────────

pub type PermissionCallback = Arc<dyn Fn(serde_json::Value) -> bool + Send + Sync>;

// ─── 内部 Client 实现 ─────────────────────────────────────────────────────────

struct OurClient {
    tx: mpsc::UnboundedSender<String>,
    #[allow(dead_code)]
    perm_cb: Option<PermissionCallback>,
}

#[async_trait::async_trait(?Send)]
impl acp::Client for OurClient {
    async fn session_notification(&self, args: SessionNotification) -> AcpResult<()> {
        let text = extract_text_chunk(&args.update);
        if let Some(t) = text {
            let _ = self.tx.send(t);
        }
        Ok(())
    }

    async fn request_permission(
        &self,
        _args: acp::RequestPermissionRequest,
    ) -> AcpResult<acp::RequestPermissionResponse> {
        // Always cancel — approve/deny is handled via the HTTP API + oneshot channel
        // (not implemented in this simplified version)
        Ok(acp::RequestPermissionResponse::new(
            acp::RequestPermissionOutcome::Cancelled,
        ))
    }
}

fn extract_text_chunk(update: &SessionUpdate) -> Option<String> {
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => {
            if let ContentBlock::Text(TextContent { text, .. }) = &chunk.content {
                if !text.is_empty() {
                    return Some(text.clone());
                }
            }
            None
        }
        _ => None,
    }
}

// ─── 公开接口 ─────────────────────────────────────────────────────────────────

pub fn run_prompt(
    resolved: &ResolvedAgent,
    agent_type: &str,
    acp_session_id: Option<String>,
    messages: &[ChatMessage],
    on_chunk: Option<Arc<dyn Fn(String) + Send + Sync>>,
    perm_cb: Option<PermissionCallback>,
    model_name: Option<&str>,
    config_options_cache: Option<Vec<serde_json::Value>>,
) -> Result<RunResult> {
    let prompt_text = build_prompt(messages);
    let command = resolved.command.clone();
    let args = resolved.args.clone();
    let env = resolved.env.clone();
    let startup_delay = resolved.startup_delay_secs;
    let agent_type = agent_type.to_string();
    let model_name = model_name.map(|s| s.to_string());
    let cached_opts = config_options_cache;

    std::thread::spawn(move || -> Result<RunResult> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let local = tokio::task::LocalSet::new();

        rt.block_on(local.run_until(async move {
            let mut child = Command::new(&command)
                .args(&args)
                .envs(&env)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .with_context(|| format!("spawn failed: {command}"))?;

            let stderr = child.stderr.take().unwrap();
            let agent_tag = agent_type.clone();
            tokio::task::spawn_local(async move {
                use tokio::io::AsyncBufReadExt;
                let mut lines = tokio::io::BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!("[{}] {}", agent_tag, line);
                }
            });

            if startup_delay > 0 {
                tokio::time::sleep(Duration::from_secs(startup_delay)).await;
            }

            let (tx, mut rx) = mpsc::unbounded_channel::<String>();

            let client = OurClient { tx, perm_cb };

            let stdin_compat = child.stdin.take().unwrap().compat_write();
            let stdout_compat = child.stdout.take().unwrap().compat();

            let (conn, io_fut) = ClientSideConnection::new(
                client,
                stdin_compat,
                stdout_compat,
                |fut| { tokio::task::spawn_local(fut); },
            );

            tokio::task::spawn_local(async move {
                let _ = io_fut.await;
            });

            // initialize
            conn.initialize(InitializeRequest::new(ProtocolVersion::LATEST))
                .await
                .context("initialize failed")?;

            // new session or reuse
            let (sess_id, config_opts): (String, Vec<serde_json::Value>) =
                if let Some(sid) = acp_session_id {
                    (sid, cached_opts.unwrap_or_default())
                } else {
                    let cwd = std::env::current_dir().unwrap_or_default();
                    let resp = conn
                        .new_session(NewSessionRequest::new(cwd))
                        .await
                        .context("new_session failed")?;
                    let opts = serde_json::to_value(&resp)
                        .ok()
                        .and_then(|v| v.get("configOptions").cloned())
                        .and_then(|v| serde_json::from_value(v).ok())
                        .unwrap_or_default();
                    (resp.session_id.to_string(), opts)
                };

            // 切换模型
            if let Some(model) = &model_name {
                switch_model(&conn, &sess_id, model, &config_opts, &agent_type).await;
            }

            // prompt
            let prompt_req = PromptRequest::new(
                SessionId::from(sess_id.clone()),
                vec![ContentBlock::Text(TextContent::new(prompt_text.clone()))],
            );
            let _resp = conn.prompt(prompt_req).await.context("prompt failed")?;

            // drain
            for _ in 0..20 {
                tokio::task::yield_now().await;
            }
            drop(conn);
            rx.close();

            let mut collected = String::new();
            while let Some(chunk) = rx.recv().await {
                if let Some(cb) = &on_chunk {
                    cb(chunk.clone());
                }
                collected.push_str(&chunk);
            }

            let _ = child.kill().await;

            Ok(RunResult {
                text: collected,
                session_id: sess_id,
                config_options: config_opts,
                prompt_tokens: estimate_tokens(&prompt_text),
            })
        }))
    })
    .join()
    .map_err(|_| anyhow::anyhow!("thread panicked"))?
}

pub fn query_models(resolved: &ResolvedAgent, agent_type: &str, timeout_secs: u64) -> Vec<(String, u64)> {
    let fallback = vec![(format!("acp/{agent_type}"), 128_000u64)];
    let command = resolved.command.clone();
    let args = resolved.args.clone();
    let env = resolved.env.clone();
    let startup_delay = resolved.startup_delay_secs;
    let agent_type = agent_type.to_string();

    let result = std::thread::spawn(move || -> Result<Vec<(String, u64)>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let local = tokio::task::LocalSet::new();

        let timeout = Duration::from_secs(timeout_secs);
        rt.block_on(local.run_until(async move {
            let inner = async {
            let mut child = Command::new(&command)
                .args(&args)
                .envs(&env)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()?;

            if startup_delay > 0 {
                tokio::time::sleep(Duration::from_secs(startup_delay)).await;
            }

            let (tx, _rx) = mpsc::unbounded_channel::<String>();
            let client = OurClient { tx, perm_cb: None };

            let stdin_compat = child.stdin.take().unwrap().compat_write();
            let stdout_compat = child.stdout.take().unwrap().compat();

            let (conn, io_fut) = ClientSideConnection::new(
                client,
                stdin_compat,
                stdout_compat,
                |fut| { tokio::task::spawn_local(fut); },
            );

            tokio::task::spawn_local(async move {
                let _ = io_fut.await;
            });

            conn.initialize(InitializeRequest::new(ProtocolVersion::LATEST)).await?;

            let cwd = std::env::current_dir().unwrap_or_default();
            let resp = conn.new_session(NewSessionRequest::new(cwd)).await?;
            let resp_val = serde_json::to_value(&resp).unwrap_or_default();

            let _ = child.kill().await;

            // ACP 标准 models.availableModels
            if let Some(models) = resp_val
                .get("models")
                .and_then(|m| m.get("availableModels"))
                .and_then(|v| v.as_array())
            {
                let ids: Vec<(String, u64)> = models
                    .iter()
                    .filter_map(|m| {
                        m.get("modelId")
                            .and_then(|v| v.as_str())
                            .map(|s| {
                                let id = format!("acp/{agent_type}/{s}");
                                let ctx = infer_context_window(s);
                                (id, ctx)
                            })
                    })
                    .collect();
                if !ids.is_empty() {
                    return Ok(ids);
                }
            }

            // fallback: configOptions[id="model"].options[].name
            if let Some(opts) = resp_val.get("configOptions").and_then(|v| v.as_array()) {
                if let Some(model_opt) = opts.iter().find(|o| {
                    o.get("id").and_then(|v| v.as_str()) == Some("model")
                        && o.get("type").and_then(|v| v.as_str()) == Some("select")
                }) {
                    if let Some(options) = model_opt.get("options").and_then(|v| v.as_array()) {
                        let ids: Vec<(String, u64)> = options
                            .iter()
                            .filter_map(|o| {
                                o.get("name")
                                    .and_then(|v| v.as_str())
                                    .map(|n| {
                                        let id = format!("acp/{agent_type}/{n}");
                                        let ctx = infer_context_window(n);
                                        (id, ctx)
                                    })
                            })
                            .collect();
                        if !ids.is_empty() {
                            return Ok(ids);
                        }
                    }
                }
            }

            Ok(vec![(format!("acp/{agent_type}"), 128_000u64)])
            }; // end inner
            tokio::time::timeout(timeout, inner)
                .await
                .unwrap_or_else(|_| {
                    tracing::warn!("[{agent_type}] query_models timed out after {timeout_secs}s");
                    Ok(vec![(format!("acp/{agent_type}"), 128_000u64)])
                })
        }))
    })
    .join();

    match result {
        Ok(Ok(models)) => models,
        Ok(Err(e)) => {
            tracing::warn!("query_models error: {e}");
            fallback
        }
        Err(_) => fallback,
    }
}

// ─── 辅助 ─────────────────────────────────────────────────────────────────────

async fn switch_model(
    conn: &ClientSideConnection,
    sess_id: &str,
    model_name: &str,
    config_opts: &[serde_json::Value],
    agent_type: &str,
) {
    let config_value = config_opts
        .iter()
        .find(|o| {
            o.get("id").and_then(|v| v.as_str()) == Some("model")
                && o.get("type").and_then(|v| v.as_str()) == Some("select")
        })
        .and_then(|opt| opt.get("options"))
        .and_then(|v| v.as_array())
        .and_then(|options| {
            options.iter().find(|o| {
                o.get("name").and_then(|v| v.as_str()) == Some(model_name)
                    || o.get("value").and_then(|v| v.as_str()) == Some(model_name)
            })
        })
        .and_then(|o| o.get("value").and_then(|v| v.as_str()))
        .map(|s| s.to_string())
        .unwrap_or_else(|| model_name.to_string());

    let req = SetSessionConfigOptionRequest::new(
        SessionId::from(sess_id.to_string()),
        "model",
        config_value.clone(),
    );

    match conn.set_session_config_option(req).await {
        Ok(_) => tracing::info!("[{agent_type}] model switched to {config_value}"),
        Err(e) => tracing::warn!("[{agent_type}] switch model failed: {e}"),
    }
}

fn build_prompt(messages: &[ChatMessage]) -> String {
    if messages.len() == 1 {
        return messages[0].content.clone();
    }
    messages
        .iter()
        .map(|m| format!("[{}]: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn estimate_tokens(text: &str) -> u32 {
    (text.len() as f32 / 4.0).ceil() as u32
}

/// 根据模型名称关键词推断 context_window（token 数）
pub fn infer_context_window(model_name: &str) -> u64 {
    let n = model_name.to_lowercase();
    if n.contains("claude-3-5") || n.contains("claude-3.5") {
        200_000
    } else if n.contains("claude") {
        200_000
    } else if n.contains("gemini-1.5-pro") {
        1_048_576
    } else if n.contains("gemini-2") || n.contains("gemini-1.5") {
        1_000_000
    } else if n.contains("gemini") {
        128_000
    } else if n.contains("gpt-4o") || n.contains("gpt-4-turbo") {
        128_000
    } else if n.contains("gpt-4") {
        8_192
    } else if n.contains("gpt-3.5") {
        16_385
    } else if n.contains("deepseek-r1") || n.contains("deepseek-v3") {
        128_000
    } else if n.contains("deepseek") {
        128_000
    } else if n.contains("qwen") || n.contains("qwq") {
        128_000
    } else if n.contains("llama-3") {
        128_000
    } else if n.contains("mistral") || n.contains("mixtral") {
        32_000
    } else {
        128_000
    }
}

// ─── 公开数据类型 ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct RunResult {
    pub text: String,
    pub session_id: String,
    pub config_options: Vec<serde_json::Value>,
    pub prompt_tokens: u32,
}

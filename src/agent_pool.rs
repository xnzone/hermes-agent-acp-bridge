/// 持久 Agent 进程池
///
/// 每个 (agent_type, model_id) 对应一个持久化的 agent 子进程。
/// 子进程运行在专用线程的 LocalSet 上（因为 ClientSideConnection 是 !Send）。
/// 外部通过 channel 发送 prompt 请求，agent 线程实时推送 StreamEvent chunk。
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::{
    self as acp, Agent, ClientSideConnection, ContentBlock, InitializeRequest, NewSessionRequest,
    PromptRequest, ProtocolVersion, SessionId, SessionNotification, SessionUpdate,
    SetSessionConfigOptionRequest, TextContent,
};
use anyhow::{Context, Result};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::config::Config;

// ─── 公开类型 ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextChunk(String),
    ThoughtChunk(String),
}

pub struct PromptResult {
    pub session_id: String,
}

// ─── 内部消息类型 ─────────────────────────────────────────────────────────────

/// 发送给 agent 线程的请求
struct PromptReq {
    prompt_text: String,
    /// 实时 chunk 推送 sender
    chunk_tx: mpsc::UnboundedSender<StreamEvent>,
    /// 完成后通知（Ok(session_id) 或 Err）
    done_tx: oneshot::Sender<Result<String>>,
}

/// agent 线程的句柄
struct AgentHandle {
    req_tx: mpsc::UnboundedSender<PromptReq>,
    session_id: String,
    config_options: Vec<serde_json::Value>,
}

// ─── Agent 池 ─────────────────────────────────────────────────────────────────

type PoolKey = String;

pub struct AgentPool {
    agents: Arc<Mutex<HashMap<PoolKey, AgentHandle>>>,
    config: Arc<Config>,
}

impl AgentPool {
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            agents: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// 发送 prompt，实时通过 chunk_tx 推送 StreamEvent，完成后返回 PromptResult
    pub async fn prompt(
        &self,
        agent_type: &str,
        model_id: Option<&str>,
        prompt_text: String,
        chunk_tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<PromptResult> {
        let key = pool_key(agent_type, model_id);

        // 获取或创建 agent 句柄
        let req_tx = {
            let mut pool = self.agents.lock().await;

            // 检查已有句柄是否还活着（channel 未关闭）
            let alive = pool
                .get(&key)
                .map(|h| !h.req_tx.is_closed())
                .unwrap_or(false);

            if !alive {
                if pool.contains_key(&key) {
                    tracing::warn!("[{key}] agent thread died, respawning");
                    pool.remove(&key);
                }

                // 启动新 agent 线程
                let handle = spawn_agent_thread(&self.config, agent_type, model_id).await?;
                pool.insert(key.clone(), handle);
            }

            pool.get(&key).unwrap().req_tx.clone()
        };

        // 发送 prompt 请求
        let (done_tx, done_rx) = oneshot::channel::<Result<String>>();
        let req = PromptReq {
            prompt_text,
            chunk_tx,
            done_tx,
        };

        req_tx
            .send(req)
            .map_err(|_| anyhow::anyhow!("agent thread channel closed"))?;

        // 等待完成
        let session_id = done_rx
            .await
            .map_err(|_| anyhow::anyhow!("agent thread dropped done_tx"))??;

        Ok(PromptResult { session_id })
    }

    /// 获取 agent 的 session 信息（session_id, config_options）
    pub async fn get_session_info(
        &self,
        agent_type: &str,
        model_id: Option<&str>,
    ) -> Option<(String, Vec<serde_json::Value>)> {
        let key = pool_key(agent_type, model_id);
        let pool = self.agents.lock().await;
        pool.get(&key)
            .map(|h| (h.session_id.clone(), h.config_options.clone()))
    }
}

// ─── 辅助 ─────────────────────────────────────────────────────────────────────

fn pool_key(agent_type: &str, model_id: Option<&str>) -> PoolKey {
    match model_id {
        Some(m) => format!("{agent_type}:{m}"),
        None => agent_type.to_string(),
    }
}

/// 启动一个专用线程运行 agent 进程和 LocalSet，返回通信句柄
async fn spawn_agent_thread(
    config: &Arc<Config>,
    agent_type: &str,
    model_id: Option<&str>,
) -> Result<AgentHandle> {
    let resolved = config.resolve_agent(agent_type);
    let agent_type = agent_type.to_string();
    let model_id = model_id.map(|s| s.to_string());

    // 用 oneshot 等待 agent 初始化完成
    let (init_tx, init_rx) = oneshot::channel::<Result<(String, Vec<serde_json::Value>, mpsc::UnboundedSender<PromptReq>)>>();

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime");
        let local = tokio::task::LocalSet::new();

        rt.block_on(local.run_until(async move {
            // 启动 agent 子进程
            let mut child = match Command::new(&resolved.command)
                .args(&resolved.args)
                .envs(&resolved.env)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    let _ = init_tx.send(Err(anyhow::anyhow!("spawn failed: {e}")));
                    return;
                }
            };

            // stderr 日志
            let stderr = child.stderr.take().unwrap();
            let agent_tag = agent_type.clone();
            tokio::task::spawn_local(async move {
                use tokio::io::AsyncBufReadExt;
                let mut lines = tokio::io::BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!("[{}] {}", agent_tag, line);
                }
            });

            if resolved.startup_delay_secs > 0 {
                tokio::time::sleep(Duration::from_secs(resolved.startup_delay_secs)).await;
            }

            // 创建 ACP 连接（使用一个占位 tx，后续每次 prompt 会替换）
            let (placeholder_tx, _) = mpsc::unbounded_channel::<StreamEvent>();
            // 用 Arc<Mutex<...>> 让 OurClient 能动态切换 chunk_tx
            let current_tx: Arc<std::sync::Mutex<mpsc::UnboundedSender<StreamEvent>>> =
                Arc::new(std::sync::Mutex::new(placeholder_tx));

            let client = DynamicClient {
                current_tx: current_tx.clone(),
            };

            let stdin_compat = child.stdin.take().unwrap().compat_write();
            let stdout_compat = child.stdout.take().unwrap().compat();

            let (conn, io_fut) =
                ClientSideConnection::new(client, stdin_compat, stdout_compat, |fut| {
                    tokio::task::spawn_local(fut);
                });

            tokio::task::spawn_local(async move {
                let _ = io_fut.await;
            });

            // initialize
            if let Err(e) = conn
                .initialize(InitializeRequest::new(ProtocolVersion::LATEST))
                .await
                .context("initialize failed")
            {
                let _ = init_tx.send(Err(e));
                return;
            }

            // new session
            let cwd = std::env::current_dir().unwrap_or_default();
            let resp = match conn.new_session(NewSessionRequest::new(cwd)).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = init_tx.send(Err(anyhow::anyhow!("new_session failed: {e}")));
                    return;
                }
            };

            let session_id = resp.session_id.to_string();
            let config_options: Vec<serde_json::Value> = serde_json::to_value(&resp)
                .ok()
                .and_then(|v| v.get("configOptions").cloned())
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();

            // 切换模型
            if let Some(model) = &model_id {
                switch_model(&conn, &session_id, model, &config_options, &agent_type).await;
            }

            // 创建 prompt 请求 channel
            let (req_tx, mut req_rx) = mpsc::unbounded_channel::<PromptReq>();

            // 通知初始化完成
            if init_tx
                .send(Ok((session_id.clone(), config_options, req_tx)))
                .is_err()
            {
                return;
            }

            // 主循环：处理 prompt 请求
            while let Some(req) = req_rx.recv().await {
                let PromptReq {
                    prompt_text,
                    chunk_tx,
                    done_tx,
                } = req;

                // 更新当前 chunk_tx，让 DynamicClient 推送到正确的 channel
                {
                    let mut tx_guard = current_tx.lock().unwrap();
                    *tx_guard = chunk_tx;
                }

                // 发送 prompt
                let prompt_req = PromptRequest::new(
                    SessionId::from(session_id.clone()),
                    vec![ContentBlock::Text(TextContent::new(prompt_text))],
                );

                let result = conn.prompt(prompt_req).await;

                // 关闭当前 chunk_tx（用占位符替换），触发 rx 端 recv() 返回 None
                let (placeholder_tx, _) = mpsc::unbounded_channel::<StreamEvent>();
                {
                    let mut tx_guard = current_tx.lock().unwrap();
                    *tx_guard = placeholder_tx;
                }

                match result {
                    Ok(_) => {
                        let _ = done_tx.send(Ok(session_id.clone()));
                    }
                    Err(e) => {
                        let _ = done_tx.send(Err(anyhow::anyhow!("prompt failed: {e}")));
                    }
                }
            }

            // req_rx 关闭，清理子进程
            let _ = child.kill().await;
        }));
    });

    // 等待初始化完成
    let (session_id, config_options, req_tx) = init_rx
        .await
        .map_err(|_| anyhow::anyhow!("agent thread panicked during init"))??;

    Ok(AgentHandle {
        req_tx,
        session_id,
        config_options,
    })
}

// ─── DynamicClient：动态切换 chunk_tx ────────────────────────────────────────

/// 每次 prompt 时动态切换推送目标 channel
struct DynamicClient {
    current_tx: Arc<std::sync::Mutex<mpsc::UnboundedSender<StreamEvent>>>,
}

#[async_trait::async_trait(?Send)]
impl acp::Client for DynamicClient {
    async fn session_notification(
        &self,
        args: SessionNotification,
    ) -> agent_client_protocol::Result<()> {
        match &args.update {
            SessionUpdate::AgentMessageChunk(chunk) => {
                if let ContentBlock::Text(TextContent { text, .. }) = &chunk.content {
                    if !text.is_empty() {
                        let tx = self.current_tx.lock().unwrap().clone();
                        let _ = tx.send(StreamEvent::TextChunk(text.clone()));
                    }
                }
            }
            SessionUpdate::AgentThoughtChunk(chunk) => {
                if let ContentBlock::Text(TextContent { text, .. }) = &chunk.content {
                    if !text.is_empty() {
                        let tx = self.current_tx.lock().unwrap().clone();
                        let _ = tx.send(StreamEvent::ThoughtChunk(text.clone()));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    async fn request_permission(
        &self,
        _args: acp::RequestPermissionRequest,
    ) -> agent_client_protocol::Result<acp::RequestPermissionResponse> {
        Ok(acp::RequestPermissionResponse::new(
            acp::RequestPermissionOutcome::Cancelled,
        ))
    }
}

// ─── 切换模型 ─────────────────────────────────────────────────────────────────

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

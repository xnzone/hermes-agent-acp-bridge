/// 持久 Agent 进程池
///
/// 每个 (agent_type, model_id) 对应一个持久化的 agent 子进程。
/// 子进程在专用线程运行，通过 channel 接收 prompt 请求，实时推送 StreamEvent chunk。
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::{
    Agent, ByteStreams, Client, ConnectionTo,
    on_receive_notification, on_receive_request,
    Responder,
    schema::{
        ContentBlock, InitializeRequest, NewSessionRequest, PermissionOptionKind,
        PromptRequest, ProtocolVersion,
        RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
        ReadTextFileRequest, ReadTextFileResponse, SelectedPermissionOutcome,
        SessionConfigId, SessionConfigValueId, SessionId, SessionNotification, SessionUpdate,
        SetSessionConfigOptionRequest, TextContent, ToolCallStatus,
        WriteTextFileRequest, WriteTextFileResponse,
    },
};
use anyhow::Result;
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

        let req_tx = {
            let mut pool = self.agents.lock().await;

            let alive = pool
                .get(&key)
                .map(|h| !h.req_tx.is_closed())
                .unwrap_or(false);

            if !alive {
                if pool.contains_key(&key) {
                    tracing::warn!("[{key}] agent thread died, respawning");
                    pool.remove(&key);
                }

                let handle = spawn_agent_thread(&self.config, agent_type, model_id).await?;
                pool.insert(key.clone(), handle);
            }

            pool.get(&key).unwrap().req_tx.clone()
        };

        let (done_tx, done_rx) = oneshot::channel::<Result<String>>();
        let req = PromptReq {
            prompt_text,
            chunk_tx,
            done_tx,
        };

        req_tx
            .send(req)
            .map_err(|_| anyhow::anyhow!("agent thread channel closed"))?;

        let session_id = done_rx
            .await
            .map_err(|_| anyhow::anyhow!("agent thread dropped done_tx"))??;

        Ok(PromptResult { session_id })
    }

    /// 获取 agent 的 session 信息
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

fn tool_status_str(status: &ToolCallStatus) -> &'static str {
    match status {
        ToolCallStatus::Pending => "pending",
        ToolCallStatus::InProgress => "in_progress",
        ToolCallStatus::Completed => "completed",
        ToolCallStatus::Failed => "failed",
        _ => "unknown",
    }
}

/// 启动一个专用线程运行 agent 进程，返回通信句柄
async fn spawn_agent_thread(
    config: &Arc<Config>,
    agent_type: &str,
    model_id: Option<&str>,
) -> Result<AgentHandle> {
    let resolved = config.resolve_agent(agent_type);
    let agent_type = agent_type.to_string();
    let model_id = model_id.map(|s| s.to_string());

    let (init_tx, init_rx) =
        oneshot::channel::<Result<(String, Vec<serde_json::Value>, mpsc::UnboundedSender<PromptReq>)>>();

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime");

        rt.block_on(async move {
            // 启动子进程
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
            tokio::spawn(async move {
                use tokio::io::AsyncBufReadExt;
                let mut lines = tokio::io::BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!("[{}] {}", agent_tag, line);
                }
            });

            if resolved.startup_delay_secs > 0 {
                tokio::time::sleep(Duration::from_secs(resolved.startup_delay_secs)).await;
            }

            let stdin = child.stdin.take().unwrap();
            let stdout = child.stdout.take().unwrap();
            let transport = ByteStreams::new(stdin.compat_write(), stdout.compat());

            // req channel
            let (req_tx, req_rx) = mpsc::unbounded_channel::<PromptReq>();
            // 用来在 notification handler 里推送 chunk
            let current_chunk_tx: Arc<tokio::sync::Mutex<Option<mpsc::UnboundedSender<StreamEvent>>>> =
                Arc::new(tokio::sync::Mutex::new(None));

            let chunk_tx_notif = current_chunk_tx.clone();
            let _chunk_tx_notif2 = current_chunk_tx.clone();

            // init_tx 用 Arc<Mutex<Option<...>>> 传进 connect_with 闭包
            let init_tx_cell = Arc::new(std::sync::Mutex::new(Some(init_tx)));
            let init_tx_cl = init_tx_cell.clone();

            let req_tx_cl = req_tx.clone();
            let model_id_cl = model_id.clone();
            let agent_type_cl = agent_type.clone();

            // req_rx 传进闭包（必须 move），但 mpsc::UnboundedReceiver 不是 Clone
            // 用 Arc<Mutex<Option<...>>> 包裹
            let req_rx_cell = Arc::new(tokio::sync::Mutex::new(Some(req_rx)));

            let result = Client
                .builder()
                .on_receive_notification(
                    move |notif: SessionNotification, _cx| {
                        let tx_cell = chunk_tx_notif.clone();
                        async move {
                            let maybe_tx = tx_cell.lock().await;
                            let tx = match &*maybe_tx {
                                Some(t) => t.clone(),
                                None => return Ok(()),
                            };
                            drop(maybe_tx); // 释放锁

                            match &notif.update {
                                SessionUpdate::AgentMessageChunk(chunk) => {
                                    if let ContentBlock::Text(t) = &chunk.content {
                                        if !t.text.is_empty() {
                                            let _ = tx.send(StreamEvent::TextChunk(t.text.clone()));
                                        }
                                    }
                                }
                                SessionUpdate::AgentThoughtChunk(chunk) => {
                                    if let ContentBlock::Text(t) = &chunk.content {
                                        if !t.text.is_empty() {
                                            let _ = tx.send(StreamEvent::ThoughtChunk(t.text.clone()));
                                        }
                                    }
                                }
                                SessionUpdate::ToolCall(tc) => {
                                    let status_str = tool_status_str(&tc.status);
                                    let thought = format!("[tool:{:?}] {} ({})", tc.kind, tc.title, status_str);
                                    tracing::debug!(
                                        "[acp] ToolCall id={} title={:?} kind={:?} status={}",
                                        tc.tool_call_id, tc.title, tc.kind, status_str
                                    );
                                    let _ = tx.send(StreamEvent::ThoughtChunk(thought));
                                }
                                SessionUpdate::ToolCallUpdate(upd) => {
                                    let status_str = upd.fields.status
                                        .as_ref()
                                        .map(|s| tool_status_str(s))
                                        .unwrap_or("update");
                                    tracing::debug!(
                                        "[acp] ToolCallUpdate id={} status={}",
                                        upd.tool_call_id, status_str
                                    );
                                    if matches!(
                                        upd.fields.status,
                                        Some(ToolCallStatus::Completed) | Some(ToolCallStatus::Failed)
                                    ) {
                                        let title = upd.fields.title.as_deref().unwrap_or("tool");
                                        let thought = format!("[tool:{status_str}] {title}");
                                        let _ = tx.send(StreamEvent::ThoughtChunk(thought));
                                    }
                                }
                                _ => {}
                            }
                            Ok(())
                        }
                    },
                    on_receive_notification!(),
                )
                .on_receive_request(
                    move |req: ReadTextFileRequest, responder: Responder<ReadTextFileResponse>, _cx| {
                        async move {
                            let path = req.path.clone();
                            let line = req.line;
                            let limit = req.limit;
                            tokio::spawn(async move {
                                let content = match tokio::fs::read_to_string(&path).await {
                                    Ok(s) => {
                                        if let Some(line_start) = line {
                                            let lines: Vec<&str> = s.lines().collect();
                                            let start = (line_start as usize).saturating_sub(1);
                                            let end = if let Some(lim) = limit {
                                                (start + lim as usize).min(lines.len())
                                            } else {
                                                lines.len()
                                            };
                                            lines[start..end].join("\n")
                                        } else {
                                            s
                                        }
                                    }
                                    Err(_) => String::new(),
                                };
                                let _ = responder.respond(ReadTextFileResponse::new(content));
                            });
                            Ok(())
                        }
                    },
                    on_receive_request!(),
                )
                .on_receive_request(
                    move |req: WriteTextFileRequest, responder: Responder<WriteTextFileResponse>, _cx| {
                        async move {
                            tokio::spawn(async move {
                                if let Some(parent) = req.path.parent() {
                                    let _ = tokio::fs::create_dir_all(parent).await;
                                }
                                let _ = tokio::fs::write(&req.path, &req.content).await;
                                let _ = responder.respond(WriteTextFileResponse::new());
                            });
                            Ok(())
                        }
                    },
                    on_receive_request!(),
                )
                .on_receive_request(
                    move |req: RequestPermissionRequest, responder: Responder<RequestPermissionResponse>, _cx| {
                        async move {
                            let allow_option = req.options.iter()
                                .find(|o| matches!(o.kind, PermissionOptionKind::AllowAlways))
                                .or_else(|| req.options.iter()
                                    .find(|o| matches!(o.kind, PermissionOptionKind::AllowOnce)));

                            let outcome = if let Some(opt) = allow_option {
                                tracing::debug!("[acp] auto-allowing permission option_id={:?}", opt.option_id);
                                RequestPermissionOutcome::Selected(
                                    SelectedPermissionOutcome::new(opt.option_id.clone()),
                                )
                            } else {
                                tracing::warn!("[acp] no allow option found, cancelling permission request");
                                RequestPermissionOutcome::Cancelled
                            };
                            let _ = responder.respond(RequestPermissionResponse::new(outcome));
                            Ok(())
                        }
                    },
                    on_receive_request!(),
                )
                .connect_with(transport, move |cx: ConnectionTo<Agent>| async move {
                    // initialize
                    cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                        .block_task()
                        .await?;

                    // new session
                    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
                    let new_sess_resp = cx
                        .send_request(NewSessionRequest::new(cwd))
                        .block_task()
                        .await?;

                    let session_id = new_sess_resp.session_id.to_string();
                    let config_opts: Vec<serde_json::Value> = serde_json::to_value(&new_sess_resp)
                        .ok()
                        .and_then(|v| v.get("configOptions").cloned())
                        .and_then(|v| serde_json::from_value(v).ok())
                        .unwrap_or_default();

                    // 切换模型
                    if let Some(model) = &model_id_cl {
                        switch_model(&cx, &session_id, model, &config_opts, &agent_type_cl).await;
                    }

                    // 通知初始化完成
                    if let Some(tx) = init_tx_cl.lock().unwrap().take() {
                        let _ = tx.send(Ok((session_id.clone(), config_opts, req_tx_cl)));
                    }

                    // 主循环：持续处理 prompt 请求
                    let sess_id = SessionId::from(session_id.clone());
                    let mut req_rx_guard = req_rx_cell.lock().await;
                    let req_rx = req_rx_guard.as_mut().unwrap();

                    loop {
                        let req = match req_rx.recv().await {
                            Some(r) => r,
                            None => break,
                        };

                        let PromptReq { prompt_text, chunk_tx, done_tx } = req;

                        // 设置当前 chunk_tx
                        *current_chunk_tx.lock().await = Some(chunk_tx);

                        let prompt_result = cx
                            .send_request(PromptRequest::new(
                                sess_id.clone(),
                                vec![ContentBlock::Text(TextContent::new(prompt_text))],
                            ))
                            .block_task()
                            .await;

                        // 清空 chunk_tx（触发 stream 完成）
                        *current_chunk_tx.lock().await = None;

                        match prompt_result {
                            Ok(_) => { let _ = done_tx.send(Ok(session_id.clone())); }
                            Err(e) => { let _ = done_tx.send(Err(anyhow::anyhow!("prompt failed: {e}"))); }
                        }
                    }

                    Ok(())
                })
                .await;

            if let Err(e) = result {
                tracing::error!("[{agent_type}] agent connection error: {e}");
                if let Some(tx) = init_tx_cell.lock().unwrap().take() {
                    let _ = tx.send(Err(anyhow::anyhow!("agent connection error: {e}")));
                }
            }

            let _ = child.kill().await;
        });
    });

    let (session_id, config_options, req_tx) = init_rx
        .await
        .map_err(|_| anyhow::anyhow!("agent thread panicked during init"))??;

    Ok(AgentHandle {
        req_tx,
        session_id,
        config_options,
    })
}

// ─── 切换模型 ─────────────────────────────────────────────────────────────────

async fn switch_model(
    cx: &ConnectionTo<Agent>,
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
        SessionConfigId::new("model"),
        SessionConfigValueId::new(config_value.as_str()),
    );

    match cx.send_request(req).block_task().await {
        Ok(_) => tracing::info!("[{agent_type}] model switched to {config_value}"),
        Err(e) => tracing::warn!("[{agent_type}] switch model failed: {e}"),
    }
}

/// 持久 Agent 进程池
///
/// 与 TS 版 daemon.ts 对齐：agent 子进程和 ACP 连接持久存在，
/// 请求复用已有 session，而非每次冷启动。
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::{
    self as acp, Agent, ClientSideConnection, ContentBlock, InitializeRequest,
    NewSessionRequest, ProtocolVersion, SessionId, SetSessionConfigOptionRequest, TextContent,
};
use anyhow::{Context, Result};
use tokio::sync::{mpsc, Mutex, OwnedMutexGuard};
use tokio::process::{Child, Command};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::config::ResolvedAgent;

// ─── Channel 消息 ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextChunk(String),
    ThoughtChunk(String),
}

// ─── 权限回调 ─────────────────────────────────────────────────────────────────

pub type PermissionCallback = Arc<dyn Fn(serde_json::Value) -> bool + Send + Sync>;

// ─── ACP Client 实现 ──────────────────────────────────────────────────────────

struct OurClient {
    tx: mpsc::UnboundedSender<StreamEvent>,
    perm_cb: Option<PermissionCallback>,
}

#[async_trait::async_trait(?Send)]
impl acp::Client for OurClient {
    async fn session_notification(
        &self,
        args: acp::SessionNotification,
    ) -> agent_client_protocol::Result<()> {
        match &args.update {
            acp::SessionUpdate::AgentMessageChunk(chunk) => {
                if let ContentBlock::Text(TextContent { text, .. }) = &chunk.content {
                    if !text.is_empty() {
                        let _ = self.tx.send(StreamEvent::TextChunk(text.clone()));
                    }
                }
            }
            acp::SessionUpdate::AgentThoughtChunk(chunk) => {
                if let ContentBlock::Text(TextContent { text, .. }) = &chunk.content {
                    if !text.is_empty() {
                        let _ = self.tx.send(StreamEvent::ThoughtChunk(text.clone()));
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
        // TODO: 对接权限审批
        Ok(acp::RequestPermissionResponse::new(
            acp::RequestPermissionOutcome::Cancelled,
        ))
    }
}

// ─── 持久 Agent 连接 ──────────────────────────────────────────────────────────

/// 一个持久运行的 agent 子进程及其 ACP 连接
pub struct AgentConnection {
    pub agent_type: String,
    pub model_id: Option<String>,
    pub conn: ClientSideConnection,
    pub acp_session_id: String,
    pub config_options: Vec<serde_json::Value>,
    pub child: Child,
    /// 上次 prompt 的时间戳（秒），用于空闲回收
    pub last_used: u64,
}

impl AgentConnection {
    /// 检查子进程是否还活着
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// 切换模型
    pub async fn switch_model(&self, model_name: &str) {
        let config_value = self
            .config_options
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
            SessionId::from(self.acp_session_id.clone()),
            "model",
            config_value.clone(),
        );

        match self.conn.set_session_config_option(req).await {
            Ok(_) => tracing::info!("[{}] model switched to {config_value}", self.agent_type),
            Err(e) => tracing::warn!("[{}] switch model failed: {e}", self.agent_type),
        }
    }
}

// ─── Agent 池 ─────────────────────────────────────────────────────────────────

type PoolKey = String; // format: "{agent_type}[:{model_id}]"

pub struct AgentPool {
    agents: Arc<Mutex<HashMap<PoolKey, AgentConnection>>>,
    config: Arc<crate::config::Config>,
}

impl AgentPool {
    pub fn new(config: Arc<crate::config::Config>) -> Self {
        Self {
            agents: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// 获取或创建一个 agent 连接
    ///
    /// 如果池中已有匹配的存活连接则复用，否则启动新进程。
    pub async fn get_or_create(
        &self,
        agent_type: &str,
        model_id: Option<&str>,
    ) -> Result<OwnedMutexGuard<AgentConnection>> {
        let key = match model_id {
            Some(m) => format!("{agent_type}:{m}"),
            None => agent_type.to_string(),
        };

        let mut pool = self.agents.lock().await;

        // 检查已有连接
        if let Some(conn) = pool.get_mut(&key) {
            if conn.is_alive() {
                conn.last_used = now_secs();
                // 释放 pool 锁，返回对单个连接的独占访问
                // 注意：这里我们不能直接 return pool 中的引用，
                // 因为 MutexGuard 被 pool 持有。
                // 使用 Arc<Mutex<AgentConnection>> 来允许独立锁
                drop(pool);
                // 需要另一种方式 — 见下方重构
                todo!()
            } else {
                tracing::warn!("[{key}] agent process died, respawning");
                pool.remove(&key);
            }
        }

        // 启动新连接
        let resolved = self.config.resolve_agent(agent_type);
        let conn = spawn_and_init(&resolved, agent_type, model_id).await?;

        pool.insert(key.clone(), conn);
        // 同样的问题 — 需要独立锁

        todo!()
    }
}

// ─── 辅助 ─────────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn spawn_and_init(
    resolved: &ResolvedAgent,
    agent_type: &str,
    model_id: Option<&str>,
) -> Result<AgentConnection> {
    let mut child = Command::new(&resolved.command)
        .args(&resolved.args)
        .envs(&resolved.env)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn failed: {}", resolved.command))?;

    // stderr 日志
    let stderr = child.stderr.take().unwrap();
    let agent_tag = agent_type.to_string();
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

    // 创建 ACP 连接
    let (tx, _rx) = mpsc::unbounded_channel::<StreamEvent>();
    let client = OurClient {
        tx,
        perm_cb: None,
    };

    let stdin_compat = child.stdin.take().unwrap().compat_write();
    let stdout_compat = child.stdout.take().unwrap().compat();

    let (conn, io_fut) =
        ClientSideConnection::new(client, stdin_compat, stdout_compat, |fut| {
            tokio::spawn(fut);
        });

    // IO task 在 tokio 多线程 runtime 上跑，不会阻塞
    tokio::spawn(io_fut);

    // initialize
    conn.initialize(InitializeRequest::new(ProtocolVersion::LATEST))
        .await
        .context("initialize failed")?;

    // new session
    let cwd = std::env::current_dir().unwrap_or_default();
    let resp = conn
        .new_session(NewSessionRequest::new(cwd))
        .await
        .context("new_session failed")?;

    let acp_session_id = resp.session_id.to_string();
    let config_options: Vec<serde_json::Value> = serde_json::to_value(&resp)
        .ok()
        .and_then(|v| v.get("configOptions").cloned())
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();

    let mut agent_conn = AgentConnection {
        agent_type: agent_type.to_string(),
        model_id: model_id.map(|s| s.to_string()),
        conn,
        acp_session_id,
        config_options,
        child,
        last_used: now_secs(),
    };

    // 切换模型
    if let Some(model) = model_id {
        agent_conn.switch_model(model).await;
    }

    Ok(agent_conn)
}

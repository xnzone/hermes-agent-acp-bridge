/// ACP agent 通信核心
use std::{sync::Arc, time::Duration};

use agent_client_protocol::{
    self as acp, Agent, ClientSideConnection, ContentBlock, InitializeRequest, NewSessionRequest,
    PromptRequest, ProtocolVersion, SessionId, SessionNotification, SessionUpdate,
    SetSessionConfigOptionRequest, TextContent,
};
use anyhow::{Context, Result};
use regex::Regex;

use agent_client_protocol::Error as AcpError;
type AcpResult<T> = std::result::Result<T, AcpError>;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::config::ResolvedAgent;

// ─── 权限回调类型 ──────────────────────────────────────────────────────────────

pub type PermissionCallback = Arc<dyn Fn(serde_json::Value) -> bool + Send + Sync>;

// ─── 内部 Channel 消息 ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum StreamEvent {
    TextChunk(String),
    ThoughtChunk(String),
}

// ─── 内部 Client 实现 ─────────────────────────────────────────────────────────

struct OurClient {
    tx: mpsc::UnboundedSender<StreamEvent>,
    #[allow(dead_code)]
    perm_cb: Option<PermissionCallback>,
}

#[async_trait::async_trait(?Send)]
impl acp::Client for OurClient {
    async fn session_notification(&self, args: SessionNotification) -> AcpResult<()> {
        match &args.update {
            SessionUpdate::AgentMessageChunk(chunk) => {
                if let ContentBlock::Text(TextContent { text, .. }) = &chunk.content {
                    if !text.is_empty() {
                        let _ = self.tx.send(StreamEvent::TextChunk(text.clone()));
                    }
                }
            }
            SessionUpdate::AgentThoughtChunk(chunk) => {
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
    ) -> AcpResult<acp::RequestPermissionResponse> {
        Ok(acp::RequestPermissionResponse::new(
            acp::RequestPermissionOutcome::Cancelled,
        ))
    }
}

// ─── 工具调用提取 ─────────────────────────────────────────────────────────────

/// 从 ACP agent 的文本输出中提取 OpenAI 格式的 tool_calls
///
/// agent 可能以两种格式输出工具调用:
/// 1. `<tool_call>{...}</tool_call>` XML 块格式（注意开始标签无 >，匹配 Python 正则）
/// 2. 裸 JSON 格式（仅当没有 XML 块时尝试）
pub fn extract_tool_calls(text: &str) -> (Vec<ExtractedToolCall>, String) {
    if text.trim().is_empty() {
        return (vec![], String::new());
    }

    let mut extracted: Vec<ExtractedToolCall> = vec![];
    let mut consumed_spans: Vec<(usize, usize)> = vec![];
    let mut call_counter = 0u32;

    // 模式1: <tool_call>{...}</tool_call> 块 (匹配 Python: r" XCTool_call\s*(\{.*?\})\s* XCTool_call")
    let block_re = Regex::new(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>").unwrap();
    for cap in block_re.captures_iter(text) {
        let m = cap.get(0).unwrap();
        let raw_json = &cap[1];
        if let Some(tc) = parse_tool_call_json(raw_json, &mut call_counter) {
            extracted.push(tc);
            consumed_spans.push((m.start(), m.end()));
        }
    }

    // 模式2: 裸 JSON（仅当没有块格式匹配时）
    if extracted.is_empty() {
        let bare_re = Regex::new(
            r#"(?s)\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{.*?\}\s*\}"#
        ).unwrap();
        for cap in bare_re.captures_iter(text) {
            let m = cap.get(0).unwrap();
            let raw_json = m.as_str();
            if let Some(tc) = parse_tool_call_json(raw_json, &mut call_counter) {
                extracted.push(tc);
                consumed_spans.push((m.start(), m.end()));
            }
        }
    }

    if consumed_spans.is_empty() {
        return (extracted, text.trim().to_string());
    }

    // 从原文中移除被消费的 tool call 块，得到干净的文本
    consumed_spans.sort();
    let merged = merge_spans(&consumed_spans);
    let mut parts: Vec<&str> = vec![];
    let mut cursor = 0usize;
    for (start, end) in &merged {
        if cursor < *start {
            parts.push(&text[cursor..*start]);
        }
        cursor = *end;
    }
    if cursor < text.len() {
        parts.push(&text[cursor..]);
    }
    let cleaned = parts
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    (extracted, cleaned)
}

fn parse_tool_call_json(raw: &str, counter: &mut u32) -> Option<ExtractedToolCall> {
    let obj: serde_json::Value = serde_json::from_str(raw).ok()?;
    let fn_obj = obj.get("function")?.as_object()?;
    let fn_name = fn_obj.get("name")?.as_str()?.trim().to_string();
    if fn_name.is_empty() {
        return None;
    }
    let fn_args = match fn_obj.get("arguments") {
        Some(v) => {
            if v.is_string() {
                v.as_str().unwrap().to_string()
            } else {
                serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string())
            }
        }
        None => "{}".to_string(),
    };
    let call_id = obj
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            *counter += 1;
            format!("acp_call_{}", counter)
        });

    Some(ExtractedToolCall {
        id: call_id.clone(),
        call_id,
        r#type: "function".to_string(),
        function: ExtractedFunction {
            name: fn_name,
            arguments: fn_args,
        },
    })
}

fn merge_spans(spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut merged: Vec<(usize, usize)> = vec![];
    for &(start, end) in spans {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

// ─── 公开接口 ─────────────────────────────────────────────────────────────────

pub fn run_prompt(
    resolved: &ResolvedAgent,
    agent_type: &str,
    acp_session_id: Option<String>,
    messages: &[ChatMessage],
    on_chunk: Option<Arc<dyn Fn(String) + Send + Sync>>,
    on_thought_chunk: Option<Arc<dyn Fn(String) + Send + Sync>>,
    perm_cb: Option<PermissionCallback>,
    model_name: Option<&str>,
    config_options_cache: Option<Vec<serde_json::Value>>,
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
) -> Result<RunResult> {
    let prompt_text = build_prompt(messages, tools, tool_choice);
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

            let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();

            let client = OurClient { tx, perm_cb };

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

            // 并发：prompt 执行期间同时 drain stream events
            // 这样 on_chunk 回调能在 prompt() await 期间实时触发，
            // 而不是等 prompt() 完成后再批量处理
            let (prompt_done_tx, mut prompt_done_rx) =
                tokio::sync::oneshot::channel::<()>();

            // 后台任务：持续从 rx 读取 stream events 并调用回调
            let on_chunk_clone = on_chunk.clone();
            let on_thought_chunk_clone = on_thought_chunk.clone();
            let drain_handle = tokio::task::spawn_local(async move {
                let mut collected_text = String::new();
                let mut collected_thought = String::new();

                loop {
                    tokio::select! {
                        event = rx.recv() => {
                            match event {
                                Some(StreamEvent::TextChunk(chunk)) => {
                                    if let Some(cb) = &on_chunk_clone {
                                        cb(chunk.clone());
                                    }
                                    collected_text.push_str(&chunk);
                                }
                                Some(StreamEvent::ThoughtChunk(chunk)) => {
                                    if let Some(cb) = &on_thought_chunk_clone {
                                        cb(chunk.clone());
                                    }
                                    collected_thought.push_str(&chunk);
                                }
                                None => break,
                            }
                        }
                        _ = &mut prompt_done_rx => {
                            // prompt 完成了，但继续 drain 剩余 events
                            // 直到 channel 关闭（conn drop 后 tx 被 drop）
                            while let Some(event) = rx.recv().await {
                                match event {
                                    StreamEvent::TextChunk(chunk) => {
                                        if let Some(cb) = &on_chunk_clone {
                                            cb(chunk.clone());
                                        }
                                        collected_text.push_str(&chunk);
                                    }
                                    StreamEvent::ThoughtChunk(chunk) => {
                                        if let Some(cb) = &on_thought_chunk_clone {
                                            cb(chunk.clone());
                                        }
                                        collected_thought.push_str(&chunk);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }

                (collected_text, collected_thought)
            });

            // prompt
            let prompt_req = PromptRequest::new(
                SessionId::from(sess_id.clone()),
                vec![ContentBlock::Text(TextContent::new(prompt_text.clone()))],
            );
            let _resp = conn.prompt(prompt_req).await.context("prompt failed")?;

            // 通知 drain 任务 prompt 已完成
            let _ = prompt_done_tx.send(());
            // drop conn 让 tx 被 drop，从而 rx.recv() 返回 None
            drop(conn);

            // 等待 drain 完成
            let (collected_text, collected_thought) = drain_handle.await?;

            let _ = child.kill().await;

            // 提取 tool calls
            let (tool_calls, cleaned_text) = extract_tool_calls(&collected_text);

            Ok(RunResult {
                text: cleaned_text,
                reasoning: if collected_thought.is_empty() {
                    None
                } else {
                    Some(collected_thought)
                },
                tool_calls,
                session_id: sess_id,
                config_options: config_opts,
                prompt_tokens: estimate_tokens(&prompt_text),
            })
        }))
    })
    .join()
    .map_err(|_| anyhow::anyhow!("thread panicked"))?
}

pub fn query_models(
    resolved: &ResolvedAgent,
    agent_type: &str,
    timeout_secs: u64,
) -> Vec<(String, u64)> {
    // 如果配置了静态模型列表，直接返回，跳过动态查询
    if !resolved.static_models.is_empty() {
        return resolved
            .static_models
            .iter()
            .map(|m| {
                let id = format!("acp/{agent_type}/{}", m.name);
                let ctx = m.context_window.unwrap_or_else(|| infer_context_window(&m.name));
                (id, ctx)
            })
            .collect();
    }

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

                let (tx, _rx) = mpsc::unbounded_channel::<StreamEvent>();
                let client = OurClient { tx, perm_cb: None };

                let stdin_compat = child.stdin.take().unwrap().compat_write();
                let stdout_compat = child.stdout.take().unwrap().compat();

                let (conn, io_fut) =
                    ClientSideConnection::new(client, stdin_compat, stdout_compat, |fut| {
                        tokio::task::spawn_local(fut);
                    });

                tokio::task::spawn_local(async move {
                    let _ = io_fut.await;
                });

                conn.initialize(InitializeRequest::new(ProtocolVersion::LATEST))
                    .await?;

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
                            m.get("modelId").and_then(|v| v.as_str()).map(|s| {
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
                                    o.get("name").and_then(|v| v.as_str()).map(|n| {
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

/// 构建 prompt 文本，包含工具定义和指令（参考 copilot_acp_client.py 的 _format_messages_as_prompt）
fn build_prompt(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
) -> String {
    let mut sections: Vec<String> = vec![];

    // 系统指令
    sections.push("You are being used as the active ACP agent backend for Hermes.".to_string());
    sections.push("Use ACP capabilities to complete tasks.".to_string());
    sections.push(
        "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.".to_string()
    );
    sections.push("If no tool is needed, answer normally.".to_string());

    // 工具定义
    if let Some(tools) = tools {
        if !tools.is_empty() {
            let mut tool_specs: Vec<serde_json::Value> = vec![];
            for t in tools {
                let fn_obj = t.get("function");
                if let Some(fn_obj) = fn_obj {
                    let name = fn_obj.get("name").and_then(|v| v.as_str());
                    if let Some(name) = name {
                        if name.trim().is_empty() {
                            continue;
                        }
                        tool_specs.push(serde_json::json!({
                            "name": name.trim(),
                            "description": fn_obj.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                            "parameters": fn_obj.get("parameters").unwrap_or(&serde_json::json!({})),
                        }));
                    }
                }
            }
            if !tool_specs.is_empty() {
                sections.push(format!(
                    "Available tools (OpenAI function schema). \
                     When using a tool, emit ONLY <tool_call>{{...}}</tool_call> with one JSON object \
                     containing id/type/function{{name,arguments}}. arguments must be a JSON string.\n\
                     {}",
                    serde_json::to_string(&tool_specs).unwrap_or_else(|_| "[]".to_string())
                ));
            }
        }
    }

    // tool_choice
    if let Some(tc) = tool_choice {
        sections.push(format!(
            "Tool choice hint: {}",
            serde_json::to_string(tc).unwrap_or_else(|_| "null".to_string())
        ));
    }

    // 对话记录
    let mut transcript: Vec<String> = vec![];
    for msg in messages {
        let label = match msg.role.as_str() {
            "system" => "System",
            "user" => "User",
            "assistant" => "Assistant",
            "tool" => "Tool",
            _ => "Context",
        };

        // 渲染 content
        let mut rendered = String::new();
        if let Some(content) = &msg.content {
            rendered = render_content(content);
        }

        // 对于 assistant 消息，如果有 tool_calls，也要包含
        if msg.role == "assistant" {
            if let Some(tool_calls) = &msg.tool_calls {
                for tc in tool_calls {
                    let tc_json = serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    });
                    rendered.push_str(&format!(
                        "\n<tool_call>{}</tool_call>",
                        serde_json::to_string(&tc_json).unwrap_or_else(|_| "{}".to_string())
                    ));
                }
            }
        }

        // 对于 tool 消息，包含 tool_call_id 和 name
        if msg.role == "tool" {
            if let Some(name) = &msg.name {
                rendered = format!(
                    "[tool_call_id: {}, function: {}]\n{}",
                    msg.tool_call_id.as_deref().unwrap_or("unknown"),
                    name,
                    rendered
                );
            }
        }

        if !rendered.trim().is_empty() {
            transcript.push(format!("{}:\n{}", label, rendered.trim()));
        }
    }

    if !transcript.is_empty() {
        sections.push(format!(
            "Conversation transcript:\n\n{}",
            transcript.join("\n\n")
        ));
    }

    sections.push("Continue the conversation from the latest user request.".to_string());

    sections
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// 渲染消息 content（支持字符串、对象、数组）
fn render_content(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.trim().to_string(),
        serde_json::Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(|v| v.as_str()) {
                text.trim().to_string()
            } else if let Some(c) = map.get("content").and_then(|v| v.as_str()) {
                c.trim().to_string()
            } else {
                serde_json::to_string(content).unwrap_or_default()
            }
        }
        serde_json::Value::Array(items) => {
            let parts: Vec<String> = items
                .iter()
                .filter_map(|item| {
                    if let serde_json::Value::String(s) = item {
                        Some(s.trim().to_string())
                    } else if let serde_json::Value::Object(map) = item {
                        map.get("text")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().to_string())
                    } else {
                        None
                    }
                })
                .filter(|s| !s.is_empty())
                .collect();
            parts.join("\n")
        }
        serde_json::Value::Null => String::new(),
        _ => content.to_string(),
    }
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
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<ExtractedToolCall>>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

/// 从 agent 输出中提取的 tool call
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractedToolCall {
    pub id: String,
    #[serde(rename = "call_id")]
    pub call_id: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: ExtractedFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractedFunction {
    pub name: String,
    pub arguments: String,
}

pub struct RunResult {
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ExtractedToolCall>,
    pub session_id: String,
    pub config_options: Vec<serde_json::Value>,
    pub prompt_tokens: u32,
}

use serde::{Deserialize, Serialize};

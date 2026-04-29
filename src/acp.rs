/// ACP agent 通信核心 (agent-client-protocol 0.11)
use std::{path::PathBuf, sync::Arc, time::Duration};

use agent_client_protocol::{
    Agent, ByteStreams, Client, ConnectionTo,
    schema::{
        ContentBlock, InitializeRequest, NewSessionRequest,
        PromptRequest, TextContent,
        ProtocolVersion, RequestPermissionOutcome, RequestPermissionRequest,
        RequestPermissionResponse, ReadTextFileRequest, ReadTextFileResponse,
        SetSessionConfigOptionRequest, SessionConfigId, SessionConfigValueId,
        WriteTextFileRequest, WriteTextFileResponse,
    },
    on_receive_request, on_receive_notification,
    Responder,
};
use agent_client_protocol::schema::{SessionNotification, SessionUpdate};
use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::config::ResolvedAgent;

// ─── 工具调用提取 ─────────────────────────────────────────────────────────────

/// 从 ACP agent 的文本输出中提取 OpenAI 格式的 tool_calls
pub fn extract_tool_calls(text: &str) -> (Vec<ExtractedToolCall>, String) {
    if text.trim().is_empty() {
        return (vec![], String::new());
    }

    let mut extracted: Vec<ExtractedToolCall> = vec![];
    let mut consumed_spans: Vec<(usize, usize)> = vec![];
    let mut call_counter = 0u32;

    // 模式1: <tool_call>{...}</tool_call>
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

#[allow(dead_code)]
pub fn run_prompt(
    resolved: &ResolvedAgent,
    agent_type: &str,
    _acp_session_id: Option<String>,
    messages: &[ChatMessage],
    on_chunk: Option<Arc<dyn Fn(String) + Send + Sync>>,
    on_thought_chunk: Option<Arc<dyn Fn(String) + Send + Sync>>,
    _perm_cb: Option<Arc<dyn Fn(serde_json::Value) -> bool + Send + Sync>>,
    model_name: Option<&str>,
    config_options_cache: Option<Vec<serde_json::Value>>,
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
) -> Result<RunResult> {
    // agent 模式（llm_mode=false）不注入 tool schema，让 agent 自主完成任务
    let (effective_tools, effective_tool_choice) = if resolved.llm_mode {
        (tools, tool_choice)
    } else {
        (None, None)
    };
    let prompt_text = build_prompt(messages, effective_tools, effective_tool_choice);
    let llm_mode = resolved.llm_mode;
    let prompt_tokens_precomputed = estimate_tokens(&prompt_text);
    let command = resolved.command.clone();
    let args = resolved.args.clone();
    let env = resolved.env.clone();
    let startup_delay = resolved.startup_delay_secs;
    let agent_type = agent_type.to_string();
    let model_name = model_name.map(|s| s.to_string());
    let _cached_opts = config_options_cache;

    std::thread::spawn(move || -> Result<RunResult> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        rt.block_on(async move {
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
            tokio::spawn(async move {
                use tokio::io::AsyncBufReadExt;
                let mut lines = tokio::io::BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!("[{}] {}", agent_tag, line);
                }
            });

            if startup_delay > 0 {
                tokio::time::sleep(Duration::from_secs(startup_delay)).await;
            }

            let stdin = child.stdin.take().unwrap();
            let stdout = child.stdout.take().unwrap();
            let transport = ByteStreams::new(stdin.compat_write(), stdout.compat());

            let collected_text = Arc::new(tokio::sync::Mutex::new(String::new()));
            let collected_thought = Arc::new(tokio::sync::Mutex::new(String::new()));
            let sess_id_out = Arc::new(tokio::sync::Mutex::new(String::new()));
            let config_opts_out = Arc::new(tokio::sync::Mutex::new(Vec::<serde_json::Value>::new()));

            let collected_text_cl = collected_text.clone();
            let collected_thought_cl = collected_thought.clone();
            let sess_id_out_cl = sess_id_out.clone();
            let config_opts_out_cl = config_opts_out.clone();
            let on_chunk_cl = on_chunk.clone();
            let on_thought_cl = on_thought_chunk.clone();

            Client
                .builder()
                // 处理 session update notifications（chunks）
                .on_receive_notification(
                    move |notif: SessionNotification, _cx| {
                        let text_cl = collected_text_cl.clone();
                        let thought_cl = collected_thought_cl.clone();
                        let on_chunk = on_chunk_cl.clone();
                        let on_thought = on_thought_cl.clone();
                        async move {
                            match &notif.update {
                                SessionUpdate::AgentMessageChunk(chunk) => {
                                    if let ContentBlock::Text(t) = &chunk.content {
                                        if !t.text.is_empty() {
                                            if let Some(cb) = &on_chunk { cb(t.text.clone()); }
                                            text_cl.lock().await.push_str(&t.text);
                                        }
                                    }
                                }
                                SessionUpdate::AgentThoughtChunk(chunk) => {
                                    if let ContentBlock::Text(t) = &chunk.content {
                                        if !t.text.is_empty() {
                                            if let Some(cb) = &on_thought { cb(t.text.clone()); }
                                            thought_cl.lock().await.push_str(&t.text);
                                        }
                                    }
                                }
                                _ => {}
                            }
                            Ok(())
                        }
                    },
                    on_receive_notification!(),
                )
                // 处理 fs/read_text_file
                .on_receive_request(
                    async move |req: ReadTextFileRequest, responder: Responder<ReadTextFileResponse>, _cx| {
                        let path = req.path.clone();
                        let line = req.line;
                        let limit = req.limit;
                        tokio::spawn(async move {
                            let result = tokio::fs::read_to_string(&path).await;
                            let content = match result {
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
                            responder.respond(ReadTextFileResponse::new(content))
                        });
                        Ok(())
                    },
                    on_receive_request!(),
                )
                // 处理 fs/write_text_file
                .on_receive_request(
                    async move |req: WriteTextFileRequest, responder: Responder<WriteTextFileResponse>, _cx| {
                        tokio::spawn(async move {
                            if let Some(parent) = req.path.parent() {
                                let _ = tokio::fs::create_dir_all(parent).await;
                            }
                            let _ = tokio::fs::write(&req.path, &req.content).await;
                            responder.respond(WriteTextFileResponse::new())
                        });
                        Ok(())
                    },
                    on_receive_request!(),
                )
                // 处理 session/request_permission — 始终 cancel
                .on_receive_request(
                    async move |_req: RequestPermissionRequest, responder, _cx| {
                        let _ = responder.respond(RequestPermissionResponse::new(
                            RequestPermissionOutcome::Cancelled,
                        ));
                        Ok(())
                    },
                    on_receive_request!(),
                )
                .connect_with(transport, async move |cx: ConnectionTo<Agent>| {
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

                    let sess_id = new_sess_resp.session_id.clone();
                    let opts: Vec<serde_json::Value> = serde_json::to_value(&new_sess_resp)
                        .ok()
                        .and_then(|v| v.get("configOptions").cloned())
                        .and_then(|v| serde_json::from_value(v).ok())
                        .unwrap_or_default();

                    *sess_id_out_cl.lock().await = sess_id.to_string();
                    *config_opts_out_cl.lock().await = opts.clone();

                    // 切换模型
                    if let Some(ref model) = model_name {
                        switch_model(&cx, &sess_id.to_string(), model, &opts, &agent_type).await;
                    }

                    // 发送 prompt，等待完成（chunks 通过 on_receive_notification 推送）
                    cx.send_request(PromptRequest::new(
                        sess_id,
                        vec![ContentBlock::Text(TextContent::new(prompt_text))],
                    ))
                    .block_task()
                    .await?;

                    Ok(())
                })
                .await?;

            let _ = child.kill().await;

            let collected_text = Arc::try_unwrap(collected_text)
                .unwrap_or_else(|a| tokio::sync::Mutex::new(a.blocking_lock().clone()))
                .into_inner();
            let collected_thought = Arc::try_unwrap(collected_thought)
                .unwrap_or_else(|a| tokio::sync::Mutex::new(a.blocking_lock().clone()))
                .into_inner();
            let sess_id = Arc::try_unwrap(sess_id_out)
                .unwrap_or_else(|a| tokio::sync::Mutex::new(a.blocking_lock().clone()))
                .into_inner();
            let config_opts = Arc::try_unwrap(config_opts_out)
                .unwrap_or_else(|a| tokio::sync::Mutex::new(a.blocking_lock().clone()))
                .into_inner();

            tracing::debug!("[acp] raw collected_text (first 2000 chars):\n{}", &collected_text[..collected_text.len().min(2000)]);
            let (tool_calls, cleaned_text) = if llm_mode {
                extract_tool_calls(&collected_text)
            } else {
                (vec![], collected_text.clone())
            };
            if tool_calls.is_empty() {
                tracing::debug!("[acp] no tool_calls extracted from agent output");
            } else {
                for tc in &tool_calls {
                    tracing::info!("[acp] extracted tool_call: id={} name={} args={}", tc.id, tc.function.name, &tc.function.arguments[..tc.function.arguments.len().min(200)]);
                }
            }

            let prompt_tokens = prompt_tokens_precomputed;

            Ok(RunResult {
                text: cleaned_text,
                reasoning: if collected_thought.is_empty() { None } else { Some(collected_thought) },
                tool_calls,
                session_id: sess_id,
                config_options: config_opts,
                prompt_tokens,
            })
        })
    })
    .join()
    .map_err(|_| anyhow::anyhow!("thread panicked"))?
}

pub fn query_models(
    resolved: &ResolvedAgent,
    agent_type: &str,
    timeout_secs: u64,
) -> Vec<(String, u64)> {
    // static_models 只作为 fallback，优先尝试 ACP 动态查询
    let static_fallback: Vec<(String, u64)> = if !resolved.static_models.is_empty() {
        resolved
            .static_models
            .iter()
            .map(|m| {
                let id = format!("{agent_type}/{}", m.name);
                let ctx = m.context_window.unwrap_or_else(|| infer_context_window(&m.name));
                (id, ctx)
            })
            .collect()
    } else {
        vec![(format!("{agent_type}"), 200_000u64)]
    };

    let command = resolved.command.clone();
    let args = resolved.args.clone();
    let env = resolved.env.clone();
    let startup_delay = resolved.startup_delay_secs;
    let agent_type_str = agent_type.to_string();
    let fallback_clone = static_fallback.clone();
    let fallback_timeout = static_fallback.clone();

    let result = std::thread::spawn(move || -> Result<Vec<(String, u64)>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        let timeout = Duration::from_secs(timeout_secs);
        rt.block_on(async move {
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

                let stdin = child.stdin.take().unwrap();
                let stdout = child.stdout.take().unwrap();
                let transport = ByteStreams::new(stdin.compat_write(), stdout.compat());

                let config_opts_out: Arc<tokio::sync::Mutex<Vec<serde_json::Value>>> =
                    Arc::new(tokio::sync::Mutex::new(vec![]));
                let config_opts_cl = config_opts_out.clone();

                Client
                    .builder()
                    .on_receive_request(
                        async move |_req: RequestPermissionRequest, responder, _cx| {
                            let _ = responder.respond(RequestPermissionResponse::new(
                                RequestPermissionOutcome::Cancelled,
                            ));
                            Ok(())
                        },
                        on_receive_request!(),
                    )
                    .connect_with(transport, async move |cx: ConnectionTo<Agent>| {
                        cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                            .block_task()
                            .await?;

                        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
                        let new_sess_resp = cx
                            .send_request(NewSessionRequest::new(cwd))
                            .block_task()
                            .await?;

                        let opts: Vec<serde_json::Value> = serde_json::to_value(&new_sess_resp)
                            .ok()
                            .and_then(|v| v.get("configOptions").cloned())
                            .and_then(|v| serde_json::from_value(v).ok())
                            .unwrap_or_default();

                        *config_opts_cl.lock().await = opts;
                        Ok(())
                    })
                    .await?;

                let _ = child.kill().await;

                let opts = Arc::try_unwrap(config_opts_out)
                    .unwrap_or_else(|a| tokio::sync::Mutex::new(a.blocking_lock().clone()))
                    .into_inner();

                // 从 configOptions[id="model"].options[].name 提取模型列表
                if let Some(model_opt) = opts.iter().find(|o| {
                    o.get("id").and_then(|v| v.as_str()) == Some("model")
                        && o.get("type").and_then(|v| v.as_str()) == Some("select")
                }) {
                    if let Some(options) = model_opt.get("options").and_then(|v| v.as_array()) {
                        let ids: Vec<(String, u64)> = options
                            .iter()
                            .filter_map(|o| {
                                o.get("name").and_then(|v| v.as_str()).map(|n| {
                                    let id = format!("{agent_type_str}/{n}");
                                    let ctx = infer_context_window(n);
                                    (id, ctx)
                                })
                            })
                            .collect();
                        if !ids.is_empty() {
                            tracing::info!(
                                "[{agent_type_str}] got {} models from ACP",
                                ids.len()
                            );
                            return Ok(ids);
                        }
                    }
                }

                // ACP 没返回模型列表，用 static_models fallback
                tracing::debug!("[{agent_type_str}] ACP returned no model list, using static fallback");
                Ok(fallback_clone)
            };
            tokio::time::timeout(timeout, inner)
                .await
                .unwrap_or_else(|_| {
                    tracing::warn!("[{agent_type_str}] query_models timed out after {timeout_secs}s");
                    Ok(fallback_timeout)
                })
        })
    })
    .join();

    match result {
        Ok(Ok(models)) => models,
        Ok(Err(e)) => {
            tracing::warn!("query_models error for {agent_type}: {e}");
            static_fallback
        }
        Err(_) => static_fallback,
    }
}

// ─── 辅助 ─────────────────────────────────────────────────────────────────────

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
                let opt_name = o.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let opt_value = o.get("value").and_then(|v| v.as_str()).unwrap_or("");
                opt_name == model_name
                    || normalize_model_name(opt_name) == normalize_model_name(model_name)
                    || opt_value == model_name
            })
        })
        .and_then(|o| o.get("value").and_then(|v| v.as_str()))
        .map(|s| s.to_string())
        .unwrap_or_else(|| model_name.to_string());

    let req = SetSessionConfigOptionRequest::new(
        agent_client_protocol::schema::SessionId::from(sess_id.to_string()),
        SessionConfigId::from(Arc::from("model")),
        SessionConfigValueId::from(Arc::from(config_value.as_str())),
    );

    match cx.send_request(req).block_task().await {
        Ok(_) => tracing::info!("[{agent_type}] model switched to {config_value}"),
        Err(e) => tracing::warn!("[{agent_type}] switch model failed: {e}"),
    }
}

/// 判断是否是工具调用 roundtrip（messages 末尾有 role:tool 消息）
pub fn build_incremental_prompt(messages: &[ChatMessage]) -> Option<String> {
    let tool_start = messages
        .iter()
        .rposition(|m| m.role != "tool")
        .map(|i| i + 1)
        .unwrap_or(0);

    if tool_start >= messages.len() || messages[tool_start].role != "tool" {
        return None;
    }

    let tool_msgs = &messages[tool_start..];
    let mut parts: Vec<String> = vec![
        "The following tool results have been returned. Read them and continue your response.".to_string(),
    ];

    for msg in tool_msgs {
        let content = msg.content.as_ref().map(|c| render_content(c)).unwrap_or_default();
        let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
        let fn_name = msg.name.as_deref().unwrap_or("unknown");
        parts.push(format!(
            "<tool_result>\n<tool_call_id>{}</tool_call_id>\n<function>{}</function>\n<result>{}</result>\n</tool_result>",
            tool_call_id, fn_name, content
        ));
    }

    Some(parts.join("\n\n"))
}

/// 构建 prompt 文本
pub fn build_prompt(
    messages: &[ChatMessage],
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
) -> String {
    let mut sections: Vec<String> = vec![];

    sections.push("You are a language model assistant. Respond to the user's request directly.".to_string());
    sections.push(
        "CRITICAL RULES — read carefully:\n\
         1. DO NOT use bash, shell, terminal, or any execution tools.\n\
         2. DO NOT autonomously execute any commands or scripts.\n\
         3. Your ONLY job is to think and respond in plain text.\n\
         4. If the conversation includes an 'Available tools' section, you must declare tool usage \
            by outputting a <tool_call> JSON block in your response text — do NOT execute anything yourself.\n\
         5. If no tools are listed, just answer the question normally.\n\
         6. When you see <tool_result> blocks, those are results already executed by the caller — \
            read them and continue your response.".to_string()
    );

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

    if let Some(tc) = tool_choice {
        sections.push(format!(
            "Tool choice hint: {}",
            serde_json::to_string(tc).unwrap_or_else(|_| "null".to_string())
        ));
    }

    let mut transcript: Vec<String> = vec![];
    for msg in messages {
        let label = match msg.role.as_str() {
            "system" => "System",
            "user" => "User",
            "assistant" => "Assistant",
            "tool" => "Tool",
            _ => "Context",
        };

        let mut rendered = String::new();
        if let Some(content) = &msg.content {
            rendered = render_content(content);
        }

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

        if msg.role == "tool" {
            let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
            let fn_name = msg.name.as_deref().unwrap_or("unknown");
            rendered = format!(
                "<tool_result>\n<tool_call_id>{}</tool_call_id>\n<function>{}</function>\n<result>{}</result>\n</tool_result>",
                tool_call_id,
                fn_name,
                rendered
            );
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

pub fn infer_context_window(_model_name: &str) -> u64 {
    200_000
}

/// 把模型名规范化为小写 + 连字符格式
/// 例如: "TME GLM-5.1" -> "tme-glm-5.1", "Claude Sonnet 4.6" -> "claude-sonnet-4.6"
pub fn normalize_model_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c.to_ascii_lowercase() } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
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

#[allow(dead_code)]
pub struct RunResult {
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ExtractedToolCall>,
    pub session_id: String,
    pub config_options: Vec<serde_json::Value>,
    pub prompt_tokens: u32,
}

// Re-export SessionId for use in session reuse

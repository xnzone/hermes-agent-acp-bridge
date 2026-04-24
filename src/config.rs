use std::{collections::HashMap, path::PathBuf};
use serde::Deserialize;

#[derive(Deserialize, Clone, Debug, Default)]
pub struct AgentConfig {
    pub command: Option<String>,
    pub args: Option<Vec<String>>,
    pub env: Option<HashMap<String, String>>,
    pub startup_delay_secs: Option<u64>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Config {
    pub port: Option<u16>,
    pub host: Option<String>,
    pub agents: Option<HashMap<String, AgentConfig>>,
}

impl Config {
    pub fn load() -> Self {
        let candidates: Vec<PathBuf> = vec![
            std::env::var("ACP_BRIDGE_CONFIG").ok().map(PathBuf::from),
            dirs_home().map(|h| h.join(".config/hermes-agent-acp-bridge/config.json")),
            dirs_home().map(|h| h.join(".config/acp-bridge/config.json")),
            Some(PathBuf::from("./hab.config.json")),
            Some(PathBuf::from("./acp-bridge.config.json")),
        ]
        .into_iter()
        .flatten()
        .collect();

        for path in candidates {
            if path.exists() {
                if let Ok(data) = std::fs::read_to_string(&path) {
                    if let Ok(cfg) = serde_json::from_str::<Config>(&data) {
                        tracing::info!("config loaded from {}", path.display());
                        return cfg;
                    }
                }
            }
        }
        tracing::warn!("no config file found, using defaults");
        Config::default()
    }

    pub fn agent_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .agents
            .as_ref()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        // always include "continue" as a default fallback
        if !names.contains(&"continue".to_string()) {
            names.push("continue".to_string());
        }
        names
    }

    pub fn resolve_agent(&self, agent_type: &str) -> ResolvedAgent {
        let agent_cfg = self.agents.as_ref().and_then(|m| m.get(agent_type));

        // default commands
        let (def_cmd, def_args) = match agent_type {
            "continue" => ("cn".to_string(), vec!["acp".to_string()]),
            other => (other.to_string(), vec![]),
        };

        let command = agent_cfg
            .and_then(|a| a.command.clone())
            .map(expand_home)
            .unwrap_or(def_cmd);

        let args = agent_cfg
            .and_then(|a| a.args.clone())
            .unwrap_or(def_args);

        let mut env: HashMap<String, String> = std::env::vars().collect();
        if let Some(extra) = agent_cfg.and_then(|a| a.env.as_ref()) {
            env.extend(extra.clone());
        }

        let startup_delay = agent_cfg
            .and_then(|a| a.startup_delay_secs)
            .unwrap_or(0);

        ResolvedAgent { command, args, env, startup_delay_secs: startup_delay }
    }
}

pub struct ResolvedAgent {
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub startup_delay_secs: u64,
}

fn expand_home(s: String) -> String {
    if s.starts_with("~/") {
        if let Some(home) = dirs_home() {
            return format!("{}/{}", home.display(), &s[2..]);
        }
    }
    s
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

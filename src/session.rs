use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use uuid::Uuid;

pub struct Session {
    pub id: String,
    pub agent_type: String,
    pub model_id: Option<String>,
    pub acp_session_id: String,
    pub config_options: Vec<serde_json::Value>,
    pub created_at: u64,
    // pending permission: params + oneshot tx to approve(true)/deny(false)
    pub pending_permission: Option<(serde_json::Value, tokio::sync::oneshot::Sender<bool>)>,
}

pub type SessionStore = Arc<DashMap<String, Session>>;

pub fn new_store() -> SessionStore {
    Arc::new(DashMap::new())
}

impl Session {
    pub fn new(agent_type: &str, model_id: Option<String>) -> Self {
        Session {
            id: Uuid::new_v4().to_string(),
            agent_type: agent_type.to_string(),
            model_id,
            acp_session_id: String::new(),
            config_options: vec![],
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            pending_permission: None,
        }
    }

    pub fn display_model(&self) -> String {
        match &self.model_id {
            Some(m) => format!("acp/{}/{m}", self.agent_type),
            None => format!("acp/{}", self.agent_type),
        }
    }
}

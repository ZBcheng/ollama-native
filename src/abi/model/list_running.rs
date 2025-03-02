use serde::{Deserialize, Serialize};

use crate::action::OllamaRequest;

use super::ModelInfoDetail;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Default, Serialize)]
pub struct ListRunningModelsRequest {}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct ListRunningModelsResponse {
    pub models: Vec<ListRunningModelsInfo>,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct ListRunningModelsInfo {
    pub name: String,
    pub model: String,
    pub size: i64,
    pub digest: String,
    pub details: ModelInfoDetail,
    pub expires_at: String,
    pub size_vram: i64,
}

impl OllamaRequest for ListRunningModelsRequest {
    fn path(&self) -> String {
        "/api/ps".to_string()
    }
}

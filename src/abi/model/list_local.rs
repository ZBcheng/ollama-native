use serde::{Deserialize, Serialize};

use crate::client::OllamaRequest;

use super::ModelInfoDetail;

#[cfg(feature = "model")]
#[derive(Serialize, Default)]
pub struct ListLocalModelsRequest {}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct ListLocalModelsResponse {
    pub models: Vec<ModelInfo>,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: i64,
    pub digest: String,
    pub details: ModelInfoDetail,
}

impl OllamaRequest for ListLocalModelsRequest {
    fn path(&self) -> String {
        "/api/tags".to_string()
    }
}

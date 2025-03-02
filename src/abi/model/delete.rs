use serde::{Deserialize, Serialize};

use crate::action::OllamaRequest;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize)]
pub struct DeleteModelRequest {
    /// Model name to delete.
    pub model: String,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DeleteModelResponse {}

impl OllamaRequest for DeleteModelRequest {
    fn path(&self) -> String {
        "/api/delete".to_string()
    }
}

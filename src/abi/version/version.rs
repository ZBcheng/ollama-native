use serde::{Deserialize, Serialize};

use crate::client::OllamaRequest;

#[derive(Debug, Clone, Default, Serialize)]
pub struct VersionRequest {}

#[derive(Debug, Clone, Deserialize)]
pub struct VersionResponse {
    pub version: String,
}

impl OllamaRequest for VersionRequest {
    fn path(&self) -> String {
        "/api/version".to_string()
    }
}

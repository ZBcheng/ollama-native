use serde::{Deserialize, Serialize};

use crate::action::OllamaRequest;

#[derive(Debug, Clone, Default, Serialize)]
pub struct VersionRequest {}

#[derive(Debug, Clone, Deserialize)]
pub struct VersionResponse {
    pub version: String,
}

impl<'a> OllamaRequest for VersionRequest<'a> {
    fn path(&self) -> String {
        "/api/version".to_string()
    }
}

use serde::{Deserialize, Serialize};

use crate::action::OllamaRequest;

#[derive(Debug, Clone, Serialize)]
pub struct PushBlobRequest {
    pub file: String,
    pub digest: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PushBlobResponse {}

impl OllamaRequest for PushBlobRequest {
    fn path(&self) -> String {
        format!("/api/blobs/{}", self.digest)
    }
}

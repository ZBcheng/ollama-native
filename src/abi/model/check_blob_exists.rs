use serde::{Deserialize, Serialize};

use crate::action::OllamaRequest;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize)]
pub struct CheckBlobExistsRequest {
    pub digest: String,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CheckBlobExistsResponse {}

impl OllamaRequest for CheckBlobExistsRequest {
    fn path(&self) -> String {
        format!("/api/blobs/{}", self.digest.to_string())
    }
}

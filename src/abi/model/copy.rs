use serde::{Deserialize, Serialize};

use crate::client::OllamaRequest;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize)]
pub struct CopyModelRequest {
    pub source: String,
    pub destination: String,
}

#[cfg(feature = "model")]
#[derive(Debug, Deserialize, Default)]
pub struct CopyModelResponse {}

impl OllamaRequest for CopyModelRequest {
    fn path(&self) -> String {
        "/api/copy".to_string()
    }
}

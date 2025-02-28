use serde::{Deserialize, Serialize};

use crate::client::OllamaRequest;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Default, Serialize)]
pub struct PushModelRequest {
    /// Name of the model to push in the form of `<namespace>/<model>:<tag>`.
    pub model: String,

    /// Allow insecure connections to the library.
    /// Only use this if you are pulling from your own library during development.
    pub insecure: Option<bool>,

    /// If false the response will be returned as a single response object,
    /// rather than a stream of objects
    pub stream: bool,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushModelResponse {
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<i64>,
}

impl OllamaRequest for PushModelRequest {
    fn path(&self) -> String {
        "/api/push".to_string()
    }
}

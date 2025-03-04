use serde::{Deserialize, Serialize};

use crate::action::OllamaRequest;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Default, Serialize)]
pub struct PullModelRequest<'a> {
    /// Name of the model to pull.
    pub model: &'a str,

    /// Allow insecure connections to the library.
    /// Only use this if you are pulling from your own library during development.
    pub insecure: Option<bool>,

    /// If false the response will be returned as a single response object,
    /// rather than a stream of objects
    pub stream: bool,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullModelResponse {
    pub status: String,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullModelStreamingResponse {
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<i64>,
    pub completed: Option<i64>,
}

impl<'a> OllamaRequest for PullModelRequest<'a> {
    fn path(&self) -> String {
        "/api/pull".to_string()
    }
}

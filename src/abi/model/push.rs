use serde::{Deserialize, Serialize};

use crate::action::OllamaRequest;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Default, Serialize)]
pub struct PushModelRequest<'a> {
    /// Name of the model to push in the form of `<namespace>/<model>:<tag>`.
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
pub struct PushModelResponse {
    pub status: String,
}

#[cfg(feature = "model")]
#[cfg(feature = "stream")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushModelStreamingResponse {
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<i64>,
}

impl<'a> OllamaRequest for PushModelRequest<'a> {
    fn path(&self) -> String {
        "/api/push".to_string()
    }
}

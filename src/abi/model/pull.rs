use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    client::{OllamaRequest, OllamaResponse, RequestMethod},
    error::OllamaError,
};

#[derive(Debug, Clone, Default, Serialize)]
pub struct PullModelRequest {
    /// Name of the model to pull.
    pub model: String,

    /// Allow insecure connections to the library.
    /// Only use this if you are pulling from your own library during development.
    pub insecure: Option<bool>,

    /// If false the response will be returned as a single response object,
    /// rather than a stream of objects
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullModelResponse {
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<i64>,
    pub completed: Option<i64>,
}

impl OllamaRequest for PullModelRequest {
    fn path(&self) -> &str {
        "/api/pull"
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::POST
    }

    fn set_stream(&mut self) -> Result<(), crate::error::OllamaError> {
        self.stream = true;
        Ok(())
    }
}

#[async_trait]
impl OllamaResponse for PullModelResponse {
    async fn parse_response(response: reqwest::Response) -> Result<Self, OllamaError> {
        let content = response
            .json()
            .await
            .map_err(|e| OllamaError::DecodingError(e))?;
        Ok(content)
    }

    #[cfg(feature = "stream")]
    async fn parse_chunk(chunk: bytes::Bytes) -> Result<Self, OllamaError> {
        match serde_json::from_slice(&chunk) {
            Ok(r) => Ok(r),
            Err(e) => Err(OllamaError::StreamDecodingError(e)),
        }
    }
}

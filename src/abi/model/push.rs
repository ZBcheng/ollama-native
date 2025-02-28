use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    client::{OllamaRequest, OllamaResponse, RequestMethod},
    error::OllamaError,
};

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

    fn method(&self) -> RequestMethod {
        RequestMethod::POST
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), crate::error::OllamaError> {
        self.stream = true;
        Ok(())
    }
}

#[async_trait]
impl OllamaResponse for PushModelResponse {
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

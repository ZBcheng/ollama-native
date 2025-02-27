use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use crate::{
    client::{OllamaRequest, OllamaResponse, RequestMethod},
    error::OllamaError,
};

#[derive(Debug, Clone, Serialize)]
pub struct DeleteModelRequest {
    /// Model name to delete.
    pub model: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct DeleteModelResponse {}

impl OllamaRequest for DeleteModelRequest {
    fn path(&self) -> &str {
        "/api/delete"
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::DELETE
    }

    fn set_stream(&mut self) -> Result<(), crate::error::OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

#[async_trait]
impl OllamaResponse for DeleteModelResponse {
    async fn parse_response(response: reqwest::Response) -> Result<Self, OllamaError> {
        // Returns a 200 OK if successful, 404 Not Found if the model to be deleted doesn't exist.
        match response.status() {
            StatusCode::OK => Ok(Self::default()),
            StatusCode::NOT_FOUND => Err(OllamaError::ModelDoesNotExist),
            other => Err(OllamaError::UnknownError(format!(
                "/api/copy got unknown status code: {other}"
            ))),
        }
    }

    #[cfg(feature = "stream")]
    async fn parse_chunk(_: bytes::Bytes) -> Result<Self, OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use crate::{
    client::{OllamaRequest, OllamaResponse, RequestMethod},
    error::OllamaError,
};

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize)]
pub struct DeleteModelRequest {
    /// Model name to delete.
    pub model: String,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DeleteModelResponse {}

impl OllamaRequest for DeleteModelRequest {
    fn path(&self) -> String {
        "/api/delete".to_string()
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::Delete
    }

    #[cfg(feature = "stream")]
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
                "/api/delete got unknown status code: {other}"
            ))),
        }
    }

    #[cfg(feature = "stream")]
    async fn parse_chunk(_: bytes::Bytes) -> Result<Self, OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

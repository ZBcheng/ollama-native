use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use crate::client::{OllamaRequest, OllamaResponse, RequestMethod};
use crate::error::OllamaError;

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

    fn method(&self) -> RequestMethod {
        RequestMethod::HEAD
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

#[async_trait]
impl OllamaResponse for CheckBlobExistsResponse {
    async fn parse_response(response: reqwest::Response) -> Result<Self, OllamaError> {
        match response.status() {
            StatusCode::OK => Ok(Self::default()),
            StatusCode::NOT_FOUND => Err(OllamaError::BlobDoesNotExist),
            other => Err(OllamaError::UnknownError(format!(
                "/api/blobs/ got unknown status code: {other}",
            ))),
        }
    }

    #[cfg(feature = "stream")]
    async fn parse_chunk(_: bytes::Bytes) -> Result<Self, OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

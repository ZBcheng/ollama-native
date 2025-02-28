use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use crate::client::{OllamaRequest, OllamaResponse, RequestMethod};
use crate::error::OllamaError;

#[derive(Debug, Clone, Serialize)]
pub struct PushBlobRequest {
    pub file: String,
    pub digest: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PushBlobResponse {}

impl OllamaRequest for PushBlobRequest {
    fn path(&self) -> String {
        format!("/api/blobs/{}", self.digest)
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::PostFile(self.file.clone())
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), crate::error::OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

#[async_trait]
impl OllamaResponse for PushBlobResponse {
    async fn parse_response(response: reqwest::Response) -> Result<Self, OllamaError> {
        match response.status() {
            StatusCode::CREATED => Ok(Self::default()),
            StatusCode::BAD_REQUEST => Err(OllamaError::UnexpectedDigest),
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

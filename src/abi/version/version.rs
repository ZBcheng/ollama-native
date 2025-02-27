use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    client::{OllamaRequest, OllamaResponse, RequestMethod},
    error::OllamaError,
};

#[derive(Debug, Clone, Default, Serialize)]
pub struct VersionRequest {}

#[derive(Debug, Clone, Deserialize)]
pub struct VersionResponse {
    pub version: String,
}

impl OllamaRequest for VersionRequest {
    fn path(&self) -> &str {
        "/api/version"
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::GET
    }

    fn set_stream(&mut self) -> Result<(), OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

#[async_trait]
impl OllamaResponse for VersionResponse {
    async fn parse_response(response: reqwest::Response) -> Result<Self, OllamaError> {
        let content = response
            .json()
            .await
            .map_err(|e| OllamaError::DecodingError(e))?;
        Ok(content)
    }

    #[cfg(feature = "stream")]
    async fn parse_chunk(_: bytes::Bytes) -> Result<Self, OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

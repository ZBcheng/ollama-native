use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::client::{OllamaRequest, OllamaResponse, RequestMethod};
use crate::error::OllamaError;

use super::ModelInfoDetail;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Default, Serialize)]
pub struct ListRunningModelsRequest {}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct ListRunningModelsResponse {
    pub models: Vec<ListRunningModelsInfo>,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct ListRunningModelsInfo {
    pub name: String,
    pub model: String,
    pub size: i64,
    pub digest: String,
    pub details: ModelInfoDetail,
    pub expires_at: String,
    pub size_vram: i64,
}

impl OllamaRequest for ListRunningModelsRequest {
    fn path(&self) -> &str {
        "/api/ps"
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::GET
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

#[async_trait]
impl OllamaResponse for ListRunningModelsResponse {
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

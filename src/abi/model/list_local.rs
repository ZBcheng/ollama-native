use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::client::{OllamaRequest, OllamaResponse, RequestMethod};
use crate::error::OllamaError;

use super::ModelInfoDetail;

#[cfg(feature = "model")]
#[derive(Serialize, Default)]
pub struct ListLocalModelsRequest {}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct ListLocalModelsResponse {
    pub models: Vec<ModelInfo>,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: i64,
    pub digest: String,
    pub details: ModelInfoDetail,
}

impl OllamaRequest for ListLocalModelsRequest {
    fn path(&self) -> String {
        "/api/tags".to_string()
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::Get
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), crate::error::OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

#[async_trait]
impl OllamaResponse for ListLocalModelsResponse {
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

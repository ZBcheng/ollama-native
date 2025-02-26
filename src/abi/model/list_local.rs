use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::client::{OllamaRequest, OllamaResponse, RequestMethod};
use crate::error::OllamaError;

#[derive(Serialize, Default)]
pub struct ListLocalModelsRequest {}

#[derive(Debug, Clone, Deserialize)]
pub struct ListLocalModelResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: usize,
    pub digest: String,
    pub details: ModelInfoDetail,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfoDetail {
    pub format: String,
    pub family: String,
    pub families: Option<String>,
    pub parameter_size: String,
    pub quantization_level: String,
}

impl OllamaRequest for ListLocalModelsRequest {
    fn path(&self) -> &str {
        "/api/tags"
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::GET
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), crate::error::OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".into()))
    }
}

#[async_trait]
impl OllamaResponse for ListLocalModelResponse {
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

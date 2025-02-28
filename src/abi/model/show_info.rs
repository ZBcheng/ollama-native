use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    client::{OllamaRequest, OllamaResponse, RequestMethod},
    error::OllamaError,
};

use super::ModelInfoDetail;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize, Default)]
pub struct ShowModelInformationRequest {
    /// Name of the model to show.
    pub model: String,

    /// If set to `true`, returns full data for verbose response fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbose: Option<bool>,
}

#[cfg(feature = "model")]
#[derive(Deserialize, Debug)]
pub struct ShowModelInformationResponse {
    pub license: String,
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: ModelInfoDetail,
    pub model_info: HashMap<String, serde_json::Value>,
}

impl OllamaRequest for ShowModelInformationRequest {
    fn path(&self) -> String {
        "/api/show".to_string()
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::Post
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), crate::error::OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

#[async_trait]
impl OllamaResponse for ShowModelInformationResponse {
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

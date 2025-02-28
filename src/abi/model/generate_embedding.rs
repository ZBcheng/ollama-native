use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    abi::Parameter,
    client::{OllamaRequest, OllamaResponse, RequestMethod},
    error::OllamaError,
};

#[cfg(feature = "model")]
#[derive(Debug, Clone, Default, Serialize)]
pub struct GenerateEmbeddingRequest {
    /// Name of model to generate embeddings from.
    pub model: String,

    /// Text to generate embeddings for.
    pub prompt: String,

    /// Additional model parameters listed in the documentation for the
    /// [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
    /// such as `temperature`.
    #[serde(skip_serializing_if = "Parameter::is_default")]
    pub options: Parameter,

    /// Controls how long the model will stay loaded into memory following the request
    /// (default: 5m).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<i64>,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct GenerateEmbeddingResponse {
    pub embedding: Vec<f64>,
}

impl OllamaRequest for GenerateEmbeddingRequest {
    fn path(&self) -> &str {
        "/api/embeddings"
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::POST
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

#[async_trait]
impl OllamaResponse for GenerateEmbeddingResponse {
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

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    abi::{Message, Parameter},
    client::{OllamaRequest, OllamaResponse, RequestMethod},
    error::OllamaError,
};

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize, Default)]
pub struct CreateModelRequest {
    /// Name of the model to create.
    pub model: String,

    /// Name of an existing model to create the new model from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,

    /// A dictionary of file names to SHA256 digests of blobs to create the model from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<HashMap<String, String>>,

    /// A dictionary of file names to SHA256 digests of blobs for LORA adapters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapters: Option<HashMap<String, String>>,

    /// The prompt template for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,

    /// A list of strings containing the license or licenses for the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub license: Vec<String>,

    /// A string containing the system prompt for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// A dictionary of parameters for the model (see
    /// [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
    /// for a list of parameters.
    #[serde(skip_serializing_if = "Parameter::is_default")]
    pub parameters: Parameter,

    /// A list of message objects used to create a conversation.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,

    /// if false the response will be returned as a single response object,
    /// rather than a stream of objects.
    pub stream: bool,

    /// Quantize a non-quantized (e.g. float16) model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantize: Option<String>,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct CreateModelResponse {
    pub status: String,
}

impl OllamaRequest for CreateModelRequest {
    fn path(&self) -> String {
        "/api/create".to_string()
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::Post
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), crate::error::OllamaError> {
        self.stream = true;
        Ok(())
    }
}

#[async_trait]
impl OllamaResponse for CreateModelResponse {
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

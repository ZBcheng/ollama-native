use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::abi::{Message, Options};
use crate::action::OllamaRequest;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize, Default)]
pub struct CreateModelRequest<'a> {
    /// Name of the model to create.
    pub model: &'a str,

    /// Name of an existing model to create the new model from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<&'a str>,

    /// A dictionary of file names to SHA256 digests of blobs to create the model from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<HashMap<&'a str, &'a str>>,

    /// A dictionary of file names to SHA256 digests of blobs for LORA adapters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapters: Option<HashMap<&'a str, &'a str>>,

    /// The prompt template for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<&'a str>,

    /// A list of strings containing the license or licenses for the model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub license: Vec<&'a str>,

    /// A string containing the system prompt for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<&'a str>,

    /// A dictionary of parameters for the model (see
    /// [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
    /// for a list of parameters.
    #[serde(skip_serializing_if = "Options::is_default")]
    pub parameters: Options,

    /// A list of message objects used to create a conversation.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,

    /// if false the response will be returned as a single response object,
    /// rather than a stream of objects.
    pub stream: bool,

    /// Quantize a non-quantized (e.g. float16) model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantize: Option<&'a str>,
}

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateModelResponse {
    pub status: String,
}

impl<'a> OllamaRequest for CreateModelRequest<'a> {
    fn path(&self) -> String {
        "/api/create".to_string()
    }
}

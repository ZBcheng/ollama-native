use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::{abi::Options, action::OllamaRequest};

use super::chat::Format;

#[derive(Debug, Clone, Default, Serialize)]
pub struct GenerateCompletionRequest<'a> {
    /// The model name.
    pub model: &'a str,

    /// The prompt to generate a response for.
    pub prompt: Option<&'a str>,

    /// The text after the model response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<&'a str>,

    /// A list of base64-encoded images (for multimodal models such as `llava`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<&'a str>,

    /// The foramt to return a response in. Format can be `json` or a JSON schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Format<'a>>,

    /// Additional model parameters listed in the documentation for the
    /// [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
    /// such as `temperature`.
    #[serde(skip_serializing_if = "Options::is_default")]
    pub options: Options,

    /// System message to (overrides what is defined in the `Modelfile`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<&'a str>,

    /// The prompt template to use (overrides what is defined in the `Modelfile`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<&'a str>,

    /// If `false` the response will be returned as a single response object, rather than a stream of objects
    pub stream: bool,

    /// If `true` no formatting will be applied to the prompt. You may choose to use the `raw`
    /// parameter if you are specifying a full templated prompt in your request to the API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,

    /// Controls how long the model will stay loaded into memory following the request (default: 5m).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerateCompletionResponse {
    /// The model name.
    pub model: String,

    pub created_at: String,

    /// Empty if the response was streamed, if not streamed,
    /// this will contain the full response.
    pub response: String,

    pub done: bool,

    pub done_reason: Option<String>,

    /// Time spent generating the response.
    pub total_duration: Option<i64>,

    /// Time spent in nanoseconds loading the model.
    pub load_duration: Option<i64>,

    /// Number of tokens in the prompt.
    pub prompt_eval_count: Option<i64>,

    /// Time in nanoseconds spent generating the response
    pub prompt_eval_duration: Option<i64>,

    /// An encoding of the conversation used in this response,
    /// this can be sent in the next request to keep a conversational memory.
    pub context: Option<Vec<i64>>,

    /// Number of tokens in the response.
    pub eval_count: Option<i64>,

    /// Time in nanoseconds spent generating the response.
    pub eval_duration: Option<i64>,
}

impl<'a> GenerateCompletionRequest<'a> {
    #[inline]
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    #[inline]
    pub fn to_load_model(mut self) -> Self {
        self = Self {
            model: self.model,
            ..Default::default()
        };
        self
    }

    #[inline]
    pub fn to_unload_model(mut self) -> Self {
        self = Self {
            model: self.model,
            keep_alive: Some(0),
            ..Default::default()
        };
        self
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenerateCompletionModelResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub done_reason: String,
}

impl<'a> OllamaRequest for GenerateCompletionRequest<'a> {
    fn path(&self) -> String {
        "/api/generate".to_string()
    }
}

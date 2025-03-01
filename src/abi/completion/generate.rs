use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::{abi::Parameter, client::OllamaRequest};

#[derive(Debug, Clone, Default, Serialize)]
pub struct GenerateRequest {
    /// The model name.
    pub model: String,

    /// The prompt to generate a response for.
    pub prompt: String,

    /// The text after the model response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    /// A list of base64-encoded images (for multimodal models such as `llava`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,

    /// The foramt to return a response in. Format can be `json` or a JSON schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,

    /// Additional model parameters listed in the documentation for the
    /// [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
    /// such as `temperature`.
    #[serde(skip_serializing_if = "Parameter::is_default")]
    pub options: Parameter,

    /// System message to (overrides what is defined in the `Modelfile`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// The prompt template to use (overrides what is defined in the `Modelfile`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,

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
pub struct GenerateResponse {
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

impl OllamaRequest for GenerateRequest {
    fn path(&self) -> String {
        "/api/generate".to_string()
    }
}

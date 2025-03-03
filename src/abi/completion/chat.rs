use serde::{Deserialize, Serialize, Serializer};

use crate::{
    abi::{Message, Parameter},
    action::OllamaRequest,
};

#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatRequest<'a> {
    /// The model name.
    pub model: &'a str,

    /// The messages of the chat, this can be used to keep a chat memory.
    pub messages: Vec<Message>,

    /// List of tools in JSON for the model to use if supported.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool<'a>>,

    /// The foramt to return a response in. Format can be `json` or a JSON schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Format<'a>>,

    /// Additional model parameters listed in the documentation for the
    /// [Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values)
    /// such as `temperature`.
    #[serde(skip_serializing_if = "Parameter::is_default")]
    pub options: Parameter,

    /// If `false` the response will be returned as a single response object,
    /// rather than a stream of objects.
    pub stream: bool,

    /// Controls how long the model will stay loaded into memory following the request
    /// (default: 5m).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The model name.
    pub model: String,

    pub created_at: String,

    /// Eempty if the response was streamed, if not streamed,
    /// this will contain the full response.
    pub message: Option<Message>,

    pub done_reason: Option<String>,

    pub done: bool,

    /// Time spent generating the response.
    pub total_duration: Option<i64>,

    /// Time spent in nanoseconds loading the model.
    pub load_duration: Option<i64>,

    /// Number of tokens in the prompt.
    pub prompt_eval_count: Option<i64>,

    /// Time in nanoseconds spent generating the response
    pub prompt_eval_duration: Option<i64>,

    /// Number of tokens in the response.
    pub eval_count: Option<i64>,

    /// Time in nanoseconds spent generating the response.
    pub eval_duration: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct Tool<'a>(&'a str);

impl<'a> Tool<'a> {
    pub fn new(inner: &'a str) -> Self {
        Self(inner)
    }
}

impl<'a> Serialize for Tool<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let schema_value: serde_json::Value = serde_json::from_str(&self.0)
            .map_err(|e| serde::ser::Error::custom(format!("invalid tool format: {e}")))?;
        schema_value.serialize(serializer)
    }
}

#[derive(Debug, Clone)]
pub enum Format<'a> {
    Json,
    Schema(&'a str),
}

impl<'a> Serialize for Format<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match *self {
            Format::Json => serializer.serialize_str("json"),
            Format::Schema(ref s) => {
                let schema_value: serde_json::Value = serde_json::from_str(s).map_err(|e| {
                    serde::ser::Error::custom(format!("invalid format schema: {e}"))
                })?;

                schema_value.serialize(serializer)
            }
        }
    }
}

impl<'a> OllamaRequest for ChatRequest<'a> {
    fn path(&self) -> String {
        "/api/chat".to_string()
    }
}

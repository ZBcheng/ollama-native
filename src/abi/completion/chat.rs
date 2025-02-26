use async_trait::async_trait;
use serde::{Deserialize, Serialize, Serializer};

use crate::{
    abi::{Message, Parameter, Role},
    client::{OllamaRequest, OllamaResponse, RequestMethod},
    error::OllamaError,
};

#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatRequest {
    /// The model name.
    pub model: String,

    /// The messages of the chat, this can be used to keep a chat memory.
    pub messages: Vec<Message>,

    /// List of tools in JSON for the model to use if supported.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,

    /// The foramt to return a response in. Format can be `json` or a JSON schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Format>,

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

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseMessage {
    /// The role of the message, either `system`, `user`, `assistant`, or `tool`.
    pub role: Role,

    /// Response content.
    pub content: String,
}

#[derive(Debug, Clone)]
pub enum Tool {
    Tool(String),
}

impl Serialize for Tool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match *self {
            Tool::Tool(ref s) => {
                let schema_value: serde_json::Value = serde_json::from_str(s)
                    .map_err(|e| serde::ser::Error::custom(format!("invalid tool format: {e}")))?;
                schema_value.serialize(serializer)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Format {
    Json,
    Schema(String),
}

impl Serialize for Format {
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

impl OllamaRequest for ChatRequest {
    fn path(&self) -> &str {
        "/api/chat"
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::POST
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), OllamaError> {
        if self.tools.len() > 0 {
            return Err(OllamaError::FeatureNotAvailable(String::from("stream")));
        }
        self.stream = true;
        Ok(())
    }
}

#[async_trait]
impl OllamaResponse for ChatResponse {
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

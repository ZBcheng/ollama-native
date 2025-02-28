use serde::{Deserialize, Serialize};

use crate::{abi::Parameter, client::OllamaRequest};

#[cfg(feature = "model")]
#[derive(Debug, Clone, Default, Serialize)]
pub struct GenerateEmbeddingsRequest {
    /// Name of model to generate embeddings from.
    pub model: String,

    /// List of text to generate embeddings for.
    pub input: Vec<String>,

    /// Truncates the end of each input to fit within context length.
    /// Returns error if `false` and context length is exceeded. Defaults to `true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,

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
pub struct GenerateEmbeddingsResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f64>>,
    pub total_duration: Option<i64>,
    pub load_duration: Option<i64>,
    pub prompt_eval_count: Option<i64>,
}

impl OllamaRequest for GenerateEmbeddingsRequest {
    fn path(&self) -> String {
        "/api/embed".to_string()
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OllamaError {
    #[error("request error: {0}")]
    RequestError(reqwest::Error),

    #[error("decoding error: {0}")]
    DecodingError(reqwest::Error),

    #[error("stream decoding error: {0}")]
    StreamDecodingError(serde_json::Error),

    #[error("feature not available: {0}")]
    FeatureNotAvailable(String),

    #[error("invalid format: {0}")]
    InvalidFormat(String),

    #[error("model doesn't exist")]
    ModelDoesNotExist,

    #[error("unknown error: {0}")]
    UnknownError(String),
}

unsafe impl Send for OllamaError {}

unsafe impl Sync for OllamaError {}

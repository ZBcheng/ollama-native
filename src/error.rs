use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OllamaError {
    #[error("request error: {0}")]
    RequestError(reqwest::Error),

    #[error("decoding error: {0}")]
    DecodingError(reqwest::Error),

    #[cfg(feature = "stream")]
    #[error("stream decoding error: {0}")]
    StreamDecodingError(String),

    #[error("invalid format: {0}")]
    InvalidFormat(String),

    #[error("ollama error: {0}")]
    ServerError(String),

    #[cfg(feature = "model")]
    #[error("model does not exist")]
    ModelDoesNotExist,

    #[cfg(feature = "model")]
    #[error("blob does not exist")]
    BlobDoesNotExist,

    #[cfg(feature = "model")]
    #[error("unexpected digest")]
    UnexpectedDigest,

    #[cfg(feature = "model")]
    #[error("file error: {0}")]
    FileError(std::io::Error),
}

#[derive(Debug, Deserialize)]
pub struct ServerError {
    pub error: String,
}

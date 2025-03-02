use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OllamaError {
    /// Error occurred during HTTP request execution.
    #[error("request error: {0}")]
    RequestError(reqwest::Error),

    /// Error occurred while decoding the response body.
    #[error("decoding error: {0}")]
    DecodingError(reqwest::Error),

    /// Error occurred while decoding a streaming response.
    #[cfg(feature = "stream")]
    #[error("stream decoding error: {0}")]
    StreamDecodingError(String),

    /// The response format is invalid or unexpected.
    #[error("invalid format: {0}")]
    InvalidFormat(String),

    /// Error returned by the Ollama server.
    #[error("ollama error: {0}")]
    OllamaServerError(String),

    /// The requested model does not exist on the server.
    #[cfg(feature = "model")]
    #[error("model does not exist")]
    ModelDoesNotExist,

    /// The requested blob does not exist on the server.
    #[cfg(feature = "model")]
    #[error("blob does not exist")]
    BlobDoesNotExist,

    /// The digest of the blob does not match the expected value.
    #[cfg(feature = "model")]
    #[error("unexpected digest")]
    UnexpectedDigest,

    /// Error occurred while performing file operations.
    #[cfg(feature = "model")]
    #[error("file error: {0}")]
    FileError(std::io::Error),
}

#[derive(Debug, Deserialize)]
pub struct OllamaServerError {
    pub error: String,
}

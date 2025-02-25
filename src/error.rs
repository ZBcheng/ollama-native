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
}

unsafe impl Send for OllamaError {}

unsafe impl Sync for OllamaError {}

// impl From<reqwest::Error> for OllamaError {
//     fn from(e: reqwest::Error) -> Self {
//         Self::RequestError(e)
//     }
// }

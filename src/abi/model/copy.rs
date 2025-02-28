use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use crate::{
    client::{OllamaRequest, OllamaResponse, RequestMethod},
    error::OllamaError,
};

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize)]
pub struct CopyModelRequest {
    pub source: String,
    pub destination: String,
}

#[cfg(feature = "model")]
#[derive(Debug, Deserialize, Default)]
pub struct CopyModelResponse {}

impl OllamaRequest for CopyModelRequest {
    fn path(&self) -> String {
        "/api/copy".to_string()
    }

    fn method(&self) -> RequestMethod {
        RequestMethod::POST
    }

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

#[async_trait]
impl OllamaResponse for CopyModelResponse {
    async fn parse_response(response: reqwest::Response) -> Result<Self, OllamaError> {
        match response.status() {
            StatusCode::OK => Ok(Self::default()),
            StatusCode::NOT_FOUND => Err(OllamaError::ModelDoesNotExist),
            other => Err(OllamaError::UnknownError(format!(
                "/api/copy got unknown status code: {other}"
            ))),
        }
    }

    #[cfg(feature = "stream")]
    async fn parse_chunk(_: bytes::Bytes) -> Result<Self, OllamaError> {
        Err(OllamaError::FeatureNotAvailable("stream".to_string()))
    }
}

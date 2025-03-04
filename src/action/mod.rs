pub mod completion;
pub mod version;

#[cfg(feature = "model")]
pub mod model;

use reqwest::header::HeaderMap;
use serde::{Serialize, de::DeserializeOwned};

use crate::config::OllamaConfig;
use crate::error::OllamaError;

#[cfg(feature = "stream")]
use {async_trait::async_trait, futures::Stream, std::pin::Pin};

#[derive(Clone)]
pub struct OllamaClient {
    pub cli: reqwest::Client,
    pub config: OllamaConfig,
}

impl OllamaClient {
    pub fn new(config: OllamaConfig) -> Self {
        let cli = reqwest::Client::new();
        Self { cli, config }
    }

    pub fn url(&self) -> String {
        self.config.url.to_string()
    }

    pub async fn post(
        &self,
        request: &impl OllamaRequest,
        headers: Option<HeaderMap>,
    ) -> Result<reqwest::Response, OllamaError> {
        let serialized =
            serde_json::to_vec(&request).map_err(|e| OllamaError::InvalidFormat(e.to_string()))?;

        let url = format!("{}{}", self.config.url, request.path());
        let response = self
            .cli
            .post(url)
            .headers(headers.unwrap_or_default())
            .body(serialized)
            .send()
            .await
            .map_err(|e| OllamaError::RequestError(e))?;
        Ok(response)
    }

    pub async fn get(
        &self,
        request: &impl OllamaRequest,
    ) -> Result<reqwest::Response, OllamaError> {
        let url = format!("{}{}", self.config.url, request.path());
        let response = self
            .cli
            .get(url)
            .send()
            .await
            .map_err(|e| OllamaError::RequestError(e))?;
        Ok(response)
    }
}

#[cfg(feature = "stream")]
pub type OllamaStream<T> = Pin<Box<dyn Stream<Item = Result<T, OllamaError>>>>;

#[cfg(feature = "stream")]
#[async_trait]
pub trait IntoStream<Response> {
    async fn stream(mut self) -> Result<OllamaStream<Response>, OllamaError>;
}

pub trait OllamaRequest: Serialize + Send + Sync {
    fn path(&self) -> String;
}

pub(crate) async fn parse_response<T: DeserializeOwned>(
    response: reqwest::Response,
) -> Result<T, OllamaError> {
    response
        .json()
        .await
        .map_err(|e| OllamaError::DecodingError(e))
}

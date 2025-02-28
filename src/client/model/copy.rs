use std::{marker::PhantomData, sync::Arc};

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::copy::{CopyModelRequest, CopyModelResponse},
    client::{Action, OllamaRequest, ollama::OllamaClient},
    error::OllamaError,
};

impl Action<CopyModelRequest, CopyModelResponse> {
    pub fn new(ollama: Arc<OllamaClient>, source: &str, destination: &str) -> Self {
        let request = CopyModelRequest {
            source: source.to_string(),
            destination: destination.to_string(),
        };

        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }
}

impl IntoFuture for Action<CopyModelRequest, CopyModelResponse> {
    type Output = Result<CopyModelResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let url = format!("{}{}", self.ollama.url(), self.request.path());
            let reqwest_resp = self.ollama.post(&url, &self.request).await?;
            match reqwest_resp.status() {
                StatusCode::OK => Ok(CopyModelResponse::default()),
                StatusCode::NOT_FOUND => Err(OllamaError::ModelDoesNotExist),
                other => Err(OllamaError::UnknownError(format!(
                    "/api/copy got unknown status code: {other}"
                ))),
            }
        })
    }
}

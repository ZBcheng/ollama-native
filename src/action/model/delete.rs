use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::delete::{DeleteModelRequest, DeleteModelResponse},
    action::{Action, OllamaClient, OllamaRequest, parse_response},
    error::{OllamaError, OllamaServerError},
};

impl Action<DeleteModelRequest, DeleteModelResponse> {
    pub fn new(ollama: OllamaClient, model: &str) -> Self {
        let request = DeleteModelRequest {
            model: model.to_string(),
        };
        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }
}

#[cfg(feature = "model")]
impl IntoFuture for Action<DeleteModelRequest, DeleteModelResponse> {
    type Output = Result<DeleteModelResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let url = format!("{}{}", self.ollama.url(), self.request.path());

            let serialized = serde_json::to_vec(&self.request)
                .map_err(|e| OllamaError::InvalidFormat(e.to_string()))?;

            let reqwest_resp = reqwest::Client::new()
                .delete(url)
                .body(serialized)
                .send()
                .await
                .map_err(|e| OllamaError::RequestError(e))?;

            match reqwest_resp.status() {
                StatusCode::OK => Ok(DeleteModelResponse::default()),
                StatusCode::NOT_FOUND => Err(OllamaError::ModelDoesNotExist),
                _code => {
                    let error: OllamaServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::OllamaServerError(error.error))
                }
            }
        })
    }
}

use std::{marker::PhantomData, sync::Arc};

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::delete::{DeleteModelRequest, DeleteModelResponse},
    client::{Action, OllamaRequest, ollama::OllamaClient},
    error::OllamaError,
};

impl Action<DeleteModelRequest, DeleteModelResponse> {
    pub fn new(ollama: Arc<OllamaClient>, model: &str) -> Self {
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

            // Returns a 200 OK if successful, 404 Not Found if the model to be deleted doesn't exist.
            match reqwest_resp.status() {
                StatusCode::OK => Ok(DeleteModelResponse::default()),
                StatusCode::NOT_FOUND => Err(OllamaError::ModelDoesNotExist),
                other => Err(OllamaError::UnknownError(format!(
                    "/api/delete got unknown status code: {other}"
                ))),
            }
        })
    }
}

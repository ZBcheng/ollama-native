use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::list_running::{ListRunningModelsRequest, ListRunningModelsResponse},
    action::{Action, OllamaClient, parse_response},
    error::{OllamaError, ServerError},
};

impl Action<ListRunningModelsRequest, ListRunningModelsResponse> {
    pub fn new(ollama: OllamaClient) -> Self {
        Self {
            ollama,
            request: ListRunningModelsRequest::default(),
            _resp: PhantomData,
        }
    }
}

impl IntoFuture for Action<ListRunningModelsRequest, ListRunningModelsResponse> {
    type Output = Result<ListRunningModelsResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let reqwest_resp = self.ollama.get(&self.request).await?;
            match reqwest_resp.status() {
                StatusCode::OK => parse_response(reqwest_resp).await,
                _ => {
                    let error: ServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::ServerError(error.error))
                }
            }
        })
    }
}

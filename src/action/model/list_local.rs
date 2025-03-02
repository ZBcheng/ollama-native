use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::list_local::{ListLocalModelsRequest, ListLocalModelsResponse},
    action::{Action, OllamaClient, parse_response},
    error::{OllamaError, OllamaServerError},
};

impl Action<ListLocalModelsRequest, ListLocalModelsResponse> {
    pub fn new(ollama: OllamaClient) -> Self {
        Self {
            ollama,
            request: ListLocalModelsRequest::default(),
            _resp: PhantomData,
        }
    }
}

impl IntoFuture for Action<ListLocalModelsRequest, ListLocalModelsResponse> {
    type Output = Result<ListLocalModelsResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let reqwest_resp = self.ollama.get(&self.request).await?;
            match reqwest_resp.status() {
                StatusCode::OK => parse_response(reqwest_resp).await,
                _code => {
                    let error: OllamaServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::OllamaServerError(error.error))
                }
            }
        })
    }
}

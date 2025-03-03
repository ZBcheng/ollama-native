use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::list_running::{ListRunningModelsRequest, ListRunningModelsResponse},
    action::{OllamaClient, parse_response},
    error::{OllamaError, OllamaServerError},
};

pub struct ListRunningModelsAction<'a> {
    ollama: OllamaClient,
    request: ListRunningModelsRequest,
    _marker: &'a PhantomData<()>,
}

impl<'a> ListRunningModelsAction<'a> {
    pub fn new(ollama: OllamaClient) -> Self {
        Self {
            ollama,
            request: ListRunningModelsRequest::default(),
            _marker: &PhantomData::<()>,
        }
    }
}

impl<'a> IntoFuture for ListRunningModelsAction<'a> {
    type Output = Result<ListRunningModelsResponse, OllamaError>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

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

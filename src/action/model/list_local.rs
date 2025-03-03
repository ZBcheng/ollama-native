use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::list_local::{ListLocalModelsRequest, ListLocalModelsResponse},
    action::{OllamaClient, parse_response},
    error::{OllamaError, OllamaServerError},
};

pub struct ListLocalModelAction<'a> {
    ollama: OllamaClient,
    request: ListLocalModelsRequest,
    _marker: &'a PhantomData<()>,
}

impl<'a> ListLocalModelAction<'a> {
    pub fn new(ollama: OllamaClient) -> Self {
        Self {
            ollama,
            request: ListLocalModelsRequest::default(),
            _marker: &PhantomData::<()>,
        }
    }
}

impl<'a> IntoFuture for ListLocalModelAction<'a> {
    type Output = Result<ListLocalModelsResponse, OllamaError>;
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

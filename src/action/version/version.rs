use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::version::version::{VersionRequest, VersionResponse},
    action::{OllamaClient, parse_response},
    error::{OllamaError, OllamaServerError},
};

pub struct VersionAction<'a> {
    ollama: OllamaClient,
    request: VersionRequest<'a>,
}

impl<'a> VersionAction<'a> {
    pub fn new(ollama: OllamaClient) -> Self {
        Self {
            ollama,
            request: VersionRequest::default(),
            _marker: &PhantomData::<()>,
        }
    }
}

impl<'a> IntoFuture for VersionAction<'a> {
    type Output = Result<VersionResponse, OllamaError>;
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

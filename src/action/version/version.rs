use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::version::version::{VersionRequest, VersionResponse},
    action::{Action, OllamaClient, parse_response},
    error::{OllamaError, ServerError},
};

impl Action<VersionRequest, VersionResponse> {
    pub fn new(ollama: OllamaClient) -> Self {
        Self {
            ollama,
            request: VersionRequest::default(),
            _resp: PhantomData,
        }
    }
}

impl IntoFuture for Action<VersionRequest, VersionResponse> {
    type Output = Result<VersionResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let reqwest_resp = self.ollama.get(&self.request).await?;
            match reqwest_resp.status() {
                StatusCode::OK => parse_response(reqwest_resp).await,
                _code => {
                    let error: ServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::ServerError(error.error))
                }
            }
        })
    }
}

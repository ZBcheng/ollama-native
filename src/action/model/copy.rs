use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::copy::CopyModelRequest,
    action::{OllamaClient, parse_response},
    error::{OllamaError, OllamaServerError},
};

pub struct CopyModelAction<'a> {
    ollama: OllamaClient,
    request: CopyModelRequest<'a>,
}

impl<'a> CopyModelAction<'a> {
    pub fn new(ollama: OllamaClient, source: &'a str, destination: &'a str) -> Self {
        let request = CopyModelRequest {
            source,
            destination,
        };

        Self { ollama, request }
    }
}

impl<'a> IntoFuture for CopyModelAction<'a> {
    type Output = Result<(), OllamaError>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let reqwest_resp = self.ollama.post(&self.request, None).await?;
            match reqwest_resp.status() {
                StatusCode::OK => Ok(()),
                StatusCode::NOT_FOUND => Err(OllamaError::ModelDoesNotExist),
                _code => {
                    let error: OllamaServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::OllamaServerError(error.error))
                }
            }
        })
    }
}

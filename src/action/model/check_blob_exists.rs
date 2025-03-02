use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::check_blob_exists::{CheckBlobExistsRequest, CheckBlobExistsResponse},
    action::{Action, OllamaClient, OllamaRequest, parse_response},
    error::{OllamaError, ServerError},
};

impl Action<CheckBlobExistsRequest, CheckBlobExistsResponse> {
    pub fn new(ollama: OllamaClient, digest: &str) -> Self {
        let request = CheckBlobExistsRequest {
            digest: digest.to_string(),
        };
        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }
}

#[cfg(feature = "model")]
impl IntoFuture for Action<CheckBlobExistsRequest, CheckBlobExistsResponse> {
    type Output = Result<CheckBlobExistsResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let url = format!("{}{}", self.ollama.url(), self.request.path());
            let reqwest_resp = reqwest::Client::new()
                .head(url)
                .send()
                .await
                .map_err(|e| OllamaError::RequestError(e))?;

            match reqwest_resp.status() {
                StatusCode::OK => Ok(CheckBlobExistsResponse::default()),
                StatusCode::NOT_FOUND => Err(OllamaError::BlobDoesNotExist),
                _code => {
                    let error: ServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::ServerError(error.error))
                }
            }
        })
    }
}

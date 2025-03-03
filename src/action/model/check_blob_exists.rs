use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::check_blob_exists::CheckBlobExistsRequest,
    action::{OllamaClient, OllamaRequest, parse_response},
    error::{OllamaError, OllamaServerError},
};

pub struct CheckBlobExistsAction<'a> {
    ollama: OllamaClient,
    request: CheckBlobExistsRequest<'a>,
}

impl<'a> CheckBlobExistsAction<'a> {
    pub fn new(ollama: OllamaClient, digest: &'a str) -> Self {
        let request = CheckBlobExistsRequest { digest };
        Self { ollama, request }
    }
}

#[cfg(feature = "model")]
impl<'a> IntoFuture for CheckBlobExistsAction<'a> {
    type Output = Result<(), OllamaError>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let url = format!("{}{}", self.ollama.url(), self.request.path());
            let reqwest_resp = reqwest::Client::new()
                .head(url)
                .send()
                .await
                .map_err(|e| OllamaError::RequestError(e))?;

            match reqwest_resp.status() {
                StatusCode::OK => Ok(()),
                StatusCode::NOT_FOUND => Err(OllamaError::BlobDoesNotExist),
                _code => {
                    let error: OllamaServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::OllamaServerError(error.error))
                }
            }
        })
    }
}

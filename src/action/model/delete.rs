use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::delete::DeleteModelRequest,
    action::{OllamaClient, OllamaRequest, parse_response},
    error::{OllamaError, OllamaServerError},
};

pub struct DeleteModelAction<'a> {
    ollama: OllamaClient,
    request: DeleteModelRequest<'a>,
}

impl<'a> DeleteModelAction<'a> {
    pub fn new(ollama: OllamaClient, model: &'a str) -> Self {
        let request = DeleteModelRequest { model };
        Self { ollama, request }
    }
}

#[cfg(feature = "model")]
impl<'a> IntoFuture for DeleteModelAction<'a> {
    type Output = Result<(), OllamaError>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

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

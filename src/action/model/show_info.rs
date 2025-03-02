use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::show_info::{ShowModelInformationRequest, ShowModelInformationResponse},
    action::{Action, OllamaClient, parse_response},
    error::{OllamaError, ServerError},
};

impl Action<ShowModelInformationRequest, ShowModelInformationResponse> {
    pub fn new(ollama: OllamaClient, model: &str) -> Self {
        let request = ShowModelInformationRequest {
            model: model.to_string(),
            ..Default::default()
        };

        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }

    /// If set to true, returns full data for verbose response fields.
    pub fn verbose(mut self) -> Self {
        self.request.verbose = Some(true);
        self
    }
}

impl IntoFuture for Action<ShowModelInformationRequest, ShowModelInformationResponse> {
    type Output = Result<ShowModelInformationResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let reqwest_resp = self.ollama.post(&self.request, None).await?;
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

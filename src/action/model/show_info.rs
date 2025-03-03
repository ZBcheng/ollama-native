use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::show_info::{ShowModelInformationRequest, ShowModelInformationResponse},
    action::{OllamaClient, parse_response},
    error::{OllamaError, OllamaServerError},
};

pub struct ShowModelInformationAction<'a> {
    ollama: OllamaClient,
    request: ShowModelInformationRequest<'a>,
}

impl<'a> ShowModelInformationAction<'a> {
    pub fn new(ollama: OllamaClient, model: &'a str) -> Self {
        let request = ShowModelInformationRequest {
            model,
            ..Default::default()
        };

        Self { ollama, request }
    }

    /// If set to true, returns full data for verbose response fields.
    #[inline]
    pub fn verbose(mut self) -> Self {
        self.request.verbose = Some(true);
        self
    }
}

impl<'a> IntoFuture for ShowModelInformationAction<'a> {
    type Output = Result<ShowModelInformationResponse, OllamaError>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let reqwest_resp = self.ollama.post(&self.request, None).await?;
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

use std::{marker::PhantomData, sync::Arc};

use futures::future::BoxFuture;

use crate::{
    abi::model::list_running::{ListRunningModelsRequest, ListRunningModelsResponse},
    client::{Action, OllamaRequest, ollama::OllamaClient},
    error::OllamaError,
};

impl Action<ListRunningModelsRequest, ListRunningModelsResponse> {
    pub fn new(ollama: Arc<OllamaClient>) -> Self {
        Self {
            ollama,
            request: ListRunningModelsRequest::default(),
            _resp: PhantomData,
        }
    }
}

impl IntoFuture for Action<ListRunningModelsRequest, ListRunningModelsResponse> {
    type Output = Result<ListRunningModelsResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let url = format!("{}{}", self.ollama.url(), self.request.path());
            let reqwest_resp = self.ollama.get(&url).await?;
            let response = reqwest_resp
                .json()
                .await
                .map_err(|e| OllamaError::DecodingError(e))?;
            Ok(response)
        })
    }
}

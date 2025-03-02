use std::marker::PhantomData;

use futures::future::BoxFuture;

use crate::{
    abi::model::list_running::{ListRunningModelsRequest, ListRunningModelsResponse},
    action::{Action, OllamaClient},
    error::OllamaError,
};

impl Action<ListRunningModelsRequest, ListRunningModelsResponse> {
    pub fn new(ollama: OllamaClient) -> Self {
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
            let reqwest_resp = self.ollama.get(&self.request).await?;
            let response = reqwest_resp
                .json()
                .await
                .map_err(|e| OllamaError::DecodingError(e))?;
            Ok(response)
        })
    }
}

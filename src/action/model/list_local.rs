use std::marker::PhantomData;

use futures::future::BoxFuture;

use crate::{
    abi::model::list_local::{ListLocalModelsRequest, ListLocalModelsResponse},
    action::{Action, OllamaClient},
    error::OllamaError,
};

impl Action<ListLocalModelsRequest, ListLocalModelsResponse> {
    pub fn new(ollama: OllamaClient) -> Self {
        Self {
            ollama,
            request: ListLocalModelsRequest::default(),
            _resp: PhantomData,
        }
    }
}

impl IntoFuture for Action<ListLocalModelsRequest, ListLocalModelsResponse> {
    type Output = Result<ListLocalModelsResponse, OllamaError>;
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

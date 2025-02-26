use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::list_local::{ListLocalModelsRequest, ListLocalModelsResponse},
    client::{Action, ollama::OllamaClient},
};

impl Action<ListLocalModelsRequest, ListLocalModelsResponse> {
    pub fn new(ollama: Arc<OllamaClient>) -> Self {
        Self {
            ollama,
            request: ListLocalModelsRequest::default(),
            _resp: PhantomData,
        }
    }
}

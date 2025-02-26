use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::list_local::{ListLocalModelResponse, ListLocalModelsRequest},
    client::{Action, ollama::OllamaClient},
};

impl Action<ListLocalModelsRequest, ListLocalModelResponse> {
    pub fn new(ollama: Arc<OllamaClient>) -> Self {
        Self {
            ollama,
            request: ListLocalModelsRequest::default(),
            _resp: PhantomData,
        }
    }
}

use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::list_running::{ListRunningModelsRequest, ListRunningModelsResponse},
    client::{Action, ollama::OllamaClient},
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

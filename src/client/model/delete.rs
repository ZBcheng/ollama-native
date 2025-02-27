use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::delete::{DeleteModelRequest, DeleteModelResponse},
    client::{Action, ollama::OllamaClient},
};

impl Action<DeleteModelRequest, DeleteModelResponse> {
    pub fn new(ollama: Arc<OllamaClient>, model: &str) -> Self {
        let request = DeleteModelRequest {
            model: model.to_string(),
        };
        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }
}

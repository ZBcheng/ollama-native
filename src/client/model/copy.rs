use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::copy::{CopyModelRequest, CopyModelResponse},
    client::{Action, ollama::OllamaClient},
};

impl Action<CopyModelRequest, CopyModelResponse> {
    pub fn new(ollama: Arc<OllamaClient>, source: &str, destination: &str) -> Self {
        let request = CopyModelRequest {
            source: source.to_string(),
            destination: destination.to_string(),
        };

        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }
}

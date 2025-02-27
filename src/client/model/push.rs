use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::push::{PushModelRequest, PushModelResponse},
    client::{Action, ollama::OllamaClient},
};

impl Action<PushModelRequest, PushModelResponse> {
    pub fn new(ollama: Arc<OllamaClient>, model: &str) -> Self {
        let request = PushModelRequest {
            model: model.to_string(),
            ..Default::default()
        };

        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }

    /// Allow insecure connections to the library.
    /// Only use this if you are pushing to your library during development.
    pub fn insecure(mut self) -> Self {
        self.request.insecure = Some(true);
        self
    }
}

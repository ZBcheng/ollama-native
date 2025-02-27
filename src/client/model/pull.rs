use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::pull::{PullModelRequest, PullModelResponse},
    client::{Action, ollama::OllamaClient},
};

impl Action<PullModelRequest, PullModelResponse> {
    pub fn new(ollama: Arc<OllamaClient>, model: &str) -> Self {
        let request = PullModelRequest {
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
    /// Only use this if you are pulling from your own library during development.
    pub fn insecure(mut self) -> Self {
        self.request.insecure = Some(true);
        self
    }
}

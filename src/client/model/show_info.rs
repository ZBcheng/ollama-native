use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::show_info::{ShowModelInformationRequest, ShowModelInformationResponse},
    client::{Action, ollama::OllamaClient},
};

impl Action<ShowModelInformationRequest, ShowModelInformationResponse> {
    pub fn new(ollama: Arc<OllamaClient>, model: &str) -> Self {
        let request = ShowModelInformationRequest {
            model: model.to_string(),
            ..Default::default()
        };

        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }

    /// If set to true, returns full data for verbose response fields.
    pub fn verbose(mut self) -> Self {
        self.request.verbose = Some(true);
        self
    }
}

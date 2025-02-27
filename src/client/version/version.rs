use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::version::version::{VersionRequest, VersionResponse},
    client::{Action, ollama::OllamaClient},
};

impl Action<VersionRequest, VersionResponse> {
    pub fn new(ollama: Arc<OllamaClient>) -> Self {
        Self {
            ollama,
            request: VersionRequest::default(),
            _resp: PhantomData,
        }
    }
}

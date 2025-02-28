use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::push_blob::{PushBlobRequest, PushBlobResponse},
    client::{Action, ollama::OllamaClient},
};

impl Action<PushBlobRequest, PushBlobResponse> {
    pub fn new(ollama: Arc<OllamaClient>, file: &str, digest: &str) -> Self {
        let request = PushBlobRequest {
            file: file.to_string(),
            digest: digest.to_string(),
        };

        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }
}

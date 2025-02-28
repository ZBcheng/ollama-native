use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::check_blob_exists::{CheckBlobExistsRequest, CheckBlobExistsResponse},
    client::{Action, ollama::OllamaClient},
};

impl Action<CheckBlobExistsRequest, CheckBlobExistsResponse> {
    pub fn new(ollama: Arc<OllamaClient>, digest: &str) -> Self {
        let request = CheckBlobExistsRequest {
            digest: digest.to_string(),
        };
        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }
}

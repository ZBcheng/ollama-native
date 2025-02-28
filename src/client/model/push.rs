use std::{marker::PhantomData, sync::Arc};

use crate::{
    abi::model::push::{PushModelRequest, PushModelResponse},
    client::{Action, ollama::OllamaClient},
};

#[cfg(feature = "stream")]
use {
    crate::client::{IntoStream, OllamaRequest, OllamaStream},
    crate::error::OllamaError,
    async_stream::stream,
    async_trait::async_trait,
    tokio_stream::StreamExt,
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

#[cfg(feature = "stream")]
#[async_trait]
impl IntoStream<PushModelResponse> for Action<PushModelRequest, PushModelResponse> {
    async fn stream(mut self) -> Result<OllamaStream<PushModelResponse>, OllamaError> {
        self.request.stream = true;

        let url = format!("{}{}", self.ollama.url(), self.request.path());
        let mut reqwest_stream = self.ollama.post(&url, &self.request).await?.bytes_stream();

        let s = stream! {
            while let Some(stream_item) = reqwest_stream.next().await {
                match stream_item {
                    Ok(chunks) => match parse_chunks(&chunks) {
                        Ok(chunk) => for c in chunk {
                            yield Ok(c)
                        },
                        Err(e) => yield Err(e.into()),
                    }
                    Err(e) => yield Err(OllamaError::DecodingError(e.into()))
                }

            };
        };

        Ok(Box::pin(s))
    }
}

#[cfg(feature = "stream")]
fn parse_chunks(chunks: &[u8]) -> Result<Vec<PushModelResponse>, OllamaError> {
    let chunks = str::from_utf8(&chunks).map_err(|e| {
        OllamaError::StreamDecodingError(format!("failed to parse chunk to utf8: {e}"))
    })?;

    let splitted: Vec<&str> = chunks.trim().split('\n').collect();

    let mut resp = vec![];

    for sp in splitted {
        let deserialized: PushModelResponse = serde_json::from_str(sp).map_err(|e| {
            OllamaError::StreamDecodingError(format!(
                "failed to deserialize PushModelResponse from {sp}: {e}",
            ))
        })?;
        resp.push(deserialized);
    }

    Ok(resp)
}

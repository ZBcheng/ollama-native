use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::push::{PushModelRequest, PushModelResponse},
    action::{OllamaClient, parse_response},
    error::{OllamaError, OllamaServerError},
};

#[cfg(feature = "stream")]
use {
    crate::action::{IntoStream, OllamaStream},
    async_stream::stream,
    async_trait::async_trait,
    tokio_stream::StreamExt,
};

pub struct PushModelAction<'a> {
    ollama: OllamaClient,
    request: PushModelRequest<'a>,
}

impl<'a> PushModelAction<'a> {
    pub fn new(ollama: OllamaClient, model: &'a str) -> Self {
        let request = PushModelRequest {
            model,
            ..Default::default()
        };

        Self { ollama, request }
    }

    /// Allow insecure connections to the library.
    /// Only use this if you are pushing to your library during development.
    #[inline]
    pub fn insecure(mut self) -> Self {
        self.request.insecure = Some(true);
        self
    }
}

impl<'a> IntoFuture for PushModelAction<'a> {
    type Output = Result<PushModelResponse, OllamaError>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let reqwest_resp = self.ollama.post(&self.request, None).await?;
            match reqwest_resp.status() {
                StatusCode::OK => parse_response(reqwest_resp).await,
                _code => {
                    let error: OllamaServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::OllamaServerError(error.error))
                }
            }
        })
    }
}

#[cfg(feature = "stream")]
#[async_trait]
impl<'a> IntoStream<PushModelResponse> for PushModelAction<'a> {
    async fn stream(mut self) -> Result<OllamaStream<PushModelResponse>, OllamaError> {
        self.request.stream = true;
        let mut reqwest_stream = self.ollama.post(&self.request, None).await?.bytes_stream();

        let s = stream! {
            while let Some(stream_item) = reqwest_stream.next().await {
                match stream_item {
                    Ok(chunks) => match parse_chunks(&chunks) {
                        Ok(chunk) => for c in chunk {
                            yield Ok(c)
                        },
                        Err(e) => yield Err(e.into()),
                    }
                    Err(e) => yield Err(OllamaError::DecodingError(e))
                }

            };
        };

        Ok(Box::pin(s))
    }
}

#[cfg(feature = "stream")]
fn parse_chunks(chunks: &[u8]) -> Result<Vec<PushModelResponse>, OllamaError> {
    let chunks = std::str::from_utf8(&chunks).map_err(|e| {
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

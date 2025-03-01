use std::marker::PhantomData;

#[cfg(feature = "stream")]
use {
    crate::client::{IntoStream, OllamaStream},
    crate::error::OllamaError,
    async_stream::stream,
    async_trait::async_trait,
    tokio_stream::StreamExt,
};

use crate::{
    abi::model::pull::{PullModelRequest, PullModelResponse},
    client::{Action, ollama::OllamaClient},
};

impl Action<PullModelRequest, PullModelResponse> {
    pub fn new(ollama: OllamaClient, model: &str) -> Self {
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

#[cfg(feature = "stream")]
#[async_trait]
impl IntoStream<PullModelResponse> for Action<PullModelRequest, PullModelResponse> {
    async fn stream(mut self) -> Result<OllamaStream<PullModelResponse>, OllamaError> {
        self.request.stream = true;
        let mut reqwest_stream = self.ollama.post(&self.request, None).await?.bytes_stream();

        let s = stream! {
            while let Some(stream_item) = reqwest_stream.next().await {
                match stream_item {
                    Ok(chunks) => match parse_chunks(&chunks) {
                        Ok(r) => for c in r {
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
fn parse_chunks(chunks: &[u8]) -> Result<Vec<PullModelResponse>, OllamaError> {
    let chunks = std::str::from_utf8(&chunks).map_err(|e| {
        OllamaError::StreamDecodingError(format!("failed to parse chunk to utf8: {e}"))
    })?;

    let splitted: Vec<&str> = chunks.trim().split('\n').collect();

    let mut resp = vec![];

    for sp in splitted {
        let deserialized: PullModelResponse = serde_json::from_str(sp).map_err(|e| {
            OllamaError::StreamDecodingError(format!(
                "failed to deserialize PullModelResponse from {sp}: {e}",
            ))
        })?;
        resp.push(deserialized);
    }

    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::parse_chunks;

    #[test]
    fn parse_chunk_should_work() {
        let chunks = vec![
            r#"{"status":"pulling manifest"}"#,
            r#"{"status":"pulling 74701a8c35f6","digest":"sha256:74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45","total":1321082688,"completed":1321082688}
            {"status":"pulling 966de95ca8a6","digest":"sha256:966de95ca8a62200913e3f8bfbf84c8494536f1b94b49166851e76644e966396","total":1429,"completed":1429}"#,
            r#"{"status":"pulling fcc5a6bec9da","digest":"sha256:fcc5a6bec9daf9b561a68827b67ab6088e1dba9d1fa2a50d7bbcc8384e0a265d","total":7711,"completed":7711}
            {"status":"pulling a70ff7e570d9","digest":"sha256:a70ff7e570d97baaf4e62ac6e6ad9975e04caa6d900d3742d37698494479e0cd","total":6016,"completed":6016}
            {"status":"pulling 4f659a1e86d7","digest":"sha256:4f659a1e86d7f5a33c389f7991e7224b7ee6ad0358b53437d54c02d2e1b1118d","total":485,"completed":485}
            {"status":"verifying sha256 digest"}"#,
            r#"{"status":"writing manifest"}
            {"status":"success"}"#,
        ];
        for chunk in chunks {
            let _ = parse_chunks(chunk.as_bytes()).unwrap();
        }
    }
}

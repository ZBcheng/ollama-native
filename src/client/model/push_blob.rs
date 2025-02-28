use std::{marker::PhantomData, sync::Arc};

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::{
    abi::model::push_blob::{PushBlobRequest, PushBlobResponse},
    client::{Action, OllamaRequest, ollama::OllamaClient},
    error::OllamaError,
};

impl Action<PushBlobRequest, PushBlobResponse> {
    pub fn new(ollama: OllamaClient, file: &str, digest: &str) -> Self {
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

impl IntoFuture for Action<PushBlobRequest, PushBlobResponse> {
    type Output = Result<PushBlobResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let url = format!("{}{}", self.ollama.url(), self.request.path());
            let file_path = &self.request.file;
            let reqwest_resp = upload_file(&url, file_path).await?;
            match reqwest_resp.status() {
                StatusCode::CREATED => Ok(PushBlobResponse::default()),
                StatusCode::BAD_REQUEST => Err(OllamaError::UnexpectedDigest),
                other => Err(OllamaError::UnknownError(format!(
                    "/api/blobs/ got unknown status code: {other}",
                ))),
            }
        })
    }
}

async fn upload_file(url: &str, file_path: &str) -> Result<reqwest::Response, OllamaError> {
    let file = tokio::fs::File::open(file_path)
        .await
        .map_err(|e| OllamaError::FileError(e))?;

    use tokio_util::codec::{BytesCodec, FramedRead};
    let stream = FramedRead::new(file, BytesCodec::new());
    let body = reqwest::Body::wrap_stream(stream);

    let response = reqwest::Client::new()
        .post(url)
        .body(body)
        .send()
        .await
        .map_err(|e| OllamaError::RequestError(e))?;
    Ok(response)
}

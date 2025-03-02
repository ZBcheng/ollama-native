use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::StatusCode;
use tokio_util::codec::{BytesCodec, FramedRead};

use crate::{
    abi::model::push_blob::{PushBlobRequest, PushBlobResponse},
    action::{Action, OllamaClient, OllamaRequest, parse_response},
    error::{OllamaError, ServerError},
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
            let reqwest_resp = upload_file(self.ollama, &url, file_path).await?;
            match reqwest_resp.status() {
                StatusCode::CREATED => Ok(PushBlobResponse::default()),
                StatusCode::BAD_REQUEST => Err(OllamaError::UnexpectedDigest),
                _code => {
                    let error: ServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::ServerError(error.error))
                }
            }
        })
    }
}

async fn upload_file(
    ollama: OllamaClient,
    url: &str,
    file_path: &str,
) -> Result<reqwest::Response, OllamaError> {
    let file = tokio::fs::File::open(file_path)
        .await
        .map_err(|e| OllamaError::FileError(e))?;

    let stream = FramedRead::new(file, BytesCodec::new());
    let body = reqwest::Body::wrap_stream(stream);

    let response = ollama
        .cli
        .post(url)
        .body(body)
        .send()
        .await
        .map_err(|e| OllamaError::RequestError(e))?;
    Ok(response)
}

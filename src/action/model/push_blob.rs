use futures::future::BoxFuture;
use reqwest::StatusCode;
use tokio_util::codec::{BytesCodec, FramedRead};

use crate::{
    abi::model::push_blob::PushBlobRequest,
    action::{OllamaClient, OllamaRequest, parse_response},
    error::{OllamaError, OllamaServerError},
};

pub struct PushBlobAction<'a> {
    ollama: OllamaClient,
    request: PushBlobRequest<'a>,
}

impl<'a> PushBlobAction<'a> {
    pub fn new(ollama: OllamaClient, file: &'a str, digest: &'a str) -> Self {
        let request = PushBlobRequest { file, digest };
        Self { ollama, request }
    }
}

impl<'a> IntoFuture for PushBlobAction<'a> {
    type Output = Result<(), OllamaError>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let url = format!("{}{}", self.ollama.url(), self.request.path());
            let file_path = &self.request.file;
            let reqwest_resp = upload_file(self.ollama, &url, file_path).await?;
            match reqwest_resp.status() {
                StatusCode::CREATED => Ok(()),
                StatusCode::BAD_REQUEST => Err(OllamaError::UnexpectedDigest),
                _code => {
                    let error: OllamaServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::OllamaServerError(error.error))
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

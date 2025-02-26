pub mod completion;
pub mod ollama;
pub mod version;

#[cfg(feature = "model")]
pub mod model;

#[cfg(feature = "embedding")]
pub mod embedding;

use std::{marker::PhantomData, sync::Arc};

use async_trait::async_trait;
use futures::future::BoxFuture;
use ollama::OllamaClient;
use serde::{Serialize, de::DeserializeOwned};

#[cfg(feature = "stream")]
use async_stream::stream;
#[cfg(feature = "stream")]
use futures::Stream;
#[cfg(feature = "stream")]
use std::pin::Pin;
#[cfg(feature = "stream")]
use tokio_stream::StreamExt;

use crate::error::OllamaError;

pub struct Action<Request: OllamaRequest, Response: OllamaResponse> {
    pub ollama: Arc<OllamaClient>,
    pub request: Request,
    pub _resp: PhantomData<Response>,
}

impl<Request, Response> IntoFuture for Action<Request, Response>
where
    Request: OllamaRequest,
    Response: OllamaResponse,
{
    type Output = Result<Response, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let resp = self.ollama.request(&self.request).await?;
            let output = Response::parse_response(resp).await?;
            Ok(output)
        })
    }
}

#[cfg(feature = "stream")]
pub type OllamaStream<T> = Pin<Box<dyn Stream<Item = Result<T, OllamaError>>>>;

#[cfg(feature = "stream")]
#[allow(dead_code)]
impl<Request: OllamaRequest, Response: OllamaResponse> Action<Request, Response> {
    async fn stream(mut self) -> Result<OllamaStream<Response>, OllamaError> {
        let _ = self.request.set_stream()?;
        let mut reqwest_stream = self.ollama.request(&self.request).await?.bytes_stream();

        let s = stream! {
            while let Some(item) = reqwest_stream.next().await {
                match item {
                    Ok(chunk) => yield Response::parse_chunk(chunk).await,
                    Err(e) => yield Err(OllamaError::DecodingError(e))
                }

            };
        };

        Ok(Box::pin(s))
    }
}

#[async_trait]
pub trait OllamaRequest: Serialize + Send + Sync + 'static {
    fn path(&self) -> &str;

    fn method(&self) -> RequestMethod;

    #[cfg(feature = "stream")]
    fn set_stream(&mut self) -> Result<(), OllamaError>;
}

pub enum RequestMethod {
    POST,
    GET,
}

#[async_trait]
pub trait OllamaResponse: DeserializeOwned + Send + Sync + 'static {
    async fn parse_response(response: reqwest::Response) -> Result<Self, OllamaError>;

    #[cfg(feature = "stream")]
    async fn parse_chunk(chunk: bytes::Bytes) -> Result<Self, OllamaError>;
}

pub mod completion;
pub mod ollama;
pub mod version;

#[cfg(feature = "model")]
pub mod model;

use std::{marker::PhantomData, sync::Arc};

use ollama::OllamaClient;
use serde::Serialize;

#[cfg(feature = "stream")]
use {crate::error::OllamaError, async_trait::async_trait, futures::Stream, std::pin::Pin};

pub struct Action<Request: OllamaRequest, Response> {
    pub ollama: Arc<OllamaClient>,
    pub request: Request,
    pub _resp: PhantomData<Response>,
}

// }

#[cfg(feature = "stream")]
pub type OllamaStream<T> = Pin<Box<dyn Stream<Item = Result<T, OllamaError>>>>;

#[cfg(feature = "stream")]
#[async_trait]
pub trait IntoStream<Response> {
    async fn stream(mut self) -> Result<OllamaStream<Response>, OllamaError>;
}

pub trait OllamaRequest: Serialize + Send + Sync + 'static {
    fn path(&self) -> String;
}

// #[async_trait]
// pub trait OllamaResponse: DeserializeOwned + Send + Sync + 'static {
//     async fn parse_response(response: reqwest::Response) -> Result<Self, OllamaError>;
// }

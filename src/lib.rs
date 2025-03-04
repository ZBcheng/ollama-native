pub mod abi;
pub mod action;
pub mod config;
pub mod error;
pub mod ollama;

pub use error::OllamaError;
pub use ollama::Ollama;

pub use abi::Message;
pub use abi::completion::{
    chat::{ChatCompletionModelResponse, ChatCompletionResponse},
    generate::{GenerateCompletionModelResponse, GenerateCompletionResponse},
};
pub use abi::version::VersionResponse;

#[cfg(feature = "model")]
pub use abi::model::{
    create::CreateModelResponse, generate_embeddings::GenerateEmbeddingsResponse,
    list_local::ListLocalModelsResponse, list_running::ListRunningModelsResponse,
    pull::PullModelResponse, push::PushModelResponse, show_info::ShowModelInformationResponse,
};

#[cfg(feature = "model")]
#[cfg(feature = "stream")]
pub use abi::model::{pull::PullModelStreamingResponse, push::PushModelStreamingResponse};

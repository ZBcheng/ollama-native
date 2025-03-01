pub mod abi;
pub mod client;
pub mod config;
pub mod error;

pub use abi::Message;
pub use client::ollama::Ollama;
pub use error::OllamaError;

use serde::Deserialize;

pub mod check_blob_exists;
pub mod copy;
pub mod create;
pub mod delete;
pub mod generate_embedding;
pub mod generate_embeddings;
pub mod list_local;
pub mod list_running;
pub mod pull;
pub mod push;
pub mod show_info;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfoDetail {
    pub format: String,
    pub family: String,
    pub families: Option<Vec<String>>,
    pub parameter_size: String,
    pub quantization_level: String,
}

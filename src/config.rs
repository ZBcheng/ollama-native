use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub url: String,
}

impl OllamaConfig {
    pub fn from_url(url: &str) -> Self {
        let url = url.to_string();
        Self { url }
    }
}

use serde::Serialize;

use crate::action::OllamaRequest;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize)]
pub struct DeleteModelRequest<'a> {
    /// Model name to delete.
    pub model: &'a str,
}

impl<'a> OllamaRequest for DeleteModelRequest<'a> {
    fn path(&self) -> String {
        "/api/delete".to_string()
    }
}

use serde::Serialize;

use crate::action::OllamaRequest;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize)]
pub struct CopyModelRequest<'a> {
    pub source: &'a str,
    pub destination: &'a str,
}

impl<'a> OllamaRequest for CopyModelRequest<'a> {
    fn path(&self) -> String {
        "/api/copy".to_string()
    }
}

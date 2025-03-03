use serde::Serialize;

use crate::action::OllamaRequest;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize)]
pub struct CheckBlobExistsRequest<'a> {
    pub digest: &'a str,
}

impl<'a> OllamaRequest for CheckBlobExistsRequest<'a> {
    fn path(&self) -> String {
        format!("/api/blobs/{}", self.digest.to_string())
    }
}

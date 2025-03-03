use serde::Serialize;

use crate::action::OllamaRequest;

#[derive(Debug, Clone, Serialize)]
pub struct PushBlobRequest<'a> {
    pub file: &'a str,
    pub digest: &'a str,
}

impl<'a> OllamaRequest for PushBlobRequest<'a> {
    fn path(&self) -> String {
        format!("/api/blobs/{}", self.digest)
    }
}

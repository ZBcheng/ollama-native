use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::action::OllamaRequest;

#[derive(Debug, Clone, Serialize)]
pub struct VersionRequest<'a> {
    _marker: &'a PhantomData<()>,
}

impl<'a> Default for VersionRequest<'a> {
    fn default() -> Self {
        let _marker = &PhantomData::<()>;
        Self { _marker }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VersionResponse {
    pub version: String,
}

impl<'a> OllamaRequest for VersionRequest<'a> {
    fn path(&self) -> String {
        "/api/version".to_string()
    }
}

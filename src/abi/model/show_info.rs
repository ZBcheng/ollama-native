use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::action::OllamaRequest;

use super::ModelInfoDetail;

#[cfg(feature = "model")]
#[derive(Debug, Clone, Serialize, Default)]
pub struct ShowModelInformationRequest<'a> {
    /// Name of the model to show.
    pub model: &'a str,

    /// If set to `true`, returns full data for verbose response fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbose: Option<bool>,
}

#[cfg(feature = "model")]
#[derive(Deserialize, Debug)]
pub struct ShowModelInformationResponse {
    pub license: String,
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: ModelInfoDetail,
    pub model_info: HashMap<String, serde_json::Value>,
}

impl<'a> OllamaRequest for ShowModelInformationRequest<'a> {
    fn path(&self) -> String {
        "/api/show".to_string()
    }
}

use serde::{Deserialize, Serialize};

pub mod chat;
pub mod generate;

#[cfg(feature = "model")]
pub mod create_model;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message, either `system`, `user`, `assistant`, or `tool`.
    pub role: Role,

    /// The content of the message.
    pub content: String,

    /// A list of images to include in the message (for multimodal models such as `llava`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,

    /// A list of tools in JSON that the model wants to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Message {
    pub fn new(role: Role, content: &str) -> Self {
        Self {
            role,
            content: content.to_string(),
            images: None,
            tool_calls: None,
        }
    }

    pub fn new_system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
            images: None,
            tool_calls: None,
        }
    }

    pub fn new_user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
            images: None,
            tool_calls: None,
        }
    }

    pub fn new_assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
            images: None,
            tool_calls: None,
        }
    }

    pub fn images(mut self, images: Vec<String>) -> Self {
        let mut cur_images = self.images.unwrap_or_default();
        images.into_iter().for_each(|img| cur_images.push(img));
        self.images = Some(cur_images);
        self
    }

    pub fn image(mut self, image: &str) -> Self {
        let mut cur_images = self.images.unwrap_or_default();
        cur_images.push(image.to_string());
        self.images = Some(cur_images);
        self
    }

    pub fn tool_calls(mut self, tool_calls: Vec<serde_json::Value>) -> Self {
        let mut cur_tool_calls = self.tool_calls.unwrap_or_default();
        tool_calls
            .into_iter()
            .for_each(|tc| cur_tool_calls.push(tc));
        self.tool_calls = Some(cur_tool_calls);
        self
    }

    pub fn tool_call(mut self, tool_call: serde_json::Value) -> Self {
        let mut tool_calls = self.tool_calls.unwrap_or_default();
        tool_calls.push(tool_call);
        self.tool_calls = Some(tool_calls);
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Parameter {
    /// Enable Mirostat sampling for controlling perplexity.
    /// (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<u8>,

    /// Influences how quickly the algorithm responds to feedback from the generated text.
    /// A lower learning rate will result in slower adjustments, while a higher learning
    /// rate will make the algorithm more responsive.
    /// (Default: 0.1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f32>,

    /// Controls the balance between coherence and diversity of the output. A lower value
    /// will result in more focused and coherent text.
    /// (Default: 5.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f32>,

    /// Sets the size of the context window used to generate the next token.
    /// (Default: 2048).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,

    /// Sets how far back for the model to look back to prevent repetition.
    /// (Default: 64, 0 = disabled, -1 = num_ctx).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,

    /// Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize
    /// repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.
    /// (Default: 1.1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,

    /// The temperature of the model. Increasing the temperature will make the model answer more creatively.
    /// (Default: 0.8).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Sets the random number seed to use for generation. Setting this to a specific number
    /// will make the model generate the same text for the same prompt.
    /// (Default: 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,

    /// Sets the stop sequences to use. When this pattern is encountered the LLM will stop
    /// generating text and return. Multiple stop patterns may be set by specifying multiple
    /// separate `stop` parameters in a modelfile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,

    /// Maximum number of tokens to predict when generating text.
    /// (Default: -1, infinite generation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,

    /// Reduces the probability of generating nonsense. A higher value (e.g. 100) will give
    /// more diverse answers, while a lower value (e.g. 10) will be more conservative.
    /// (Default: 40).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text,
    /// while a lower value (e.g., 0.5) will generate more focused and conservative text.
    /// (Default: 0.9).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter
    /// p represents the minimum probability for a token to be considered, relative to the probability
    /// of the most likely token. For example, with p=0.05 and the most likely token having a probability
    /// of 0.9, logits with a value less than 0.045 are filtered out.
    /// (Default: 0.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
}

impl Parameter {
    pub fn mirostat(&mut self, mirostat: u8) {
        self.mirostat = Some(mirostat);
    }

    pub fn mirostat_eta(&mut self, mirostat_eta: f32) {
        self.mirostat_eta = Some(mirostat_eta);
    }

    pub fn mirostat_tau(&mut self, mirostat_tau: f32) {
        self.mirostat_tau = Some(mirostat_tau);
    }

    pub fn num_ctx(&mut self, num_ctx: i32) {
        self.num_ctx = Some(num_ctx);
    }

    pub fn repeat_last_n(&mut self, repeat_last_n: i32) {
        self.repeat_last_n = Some(repeat_last_n);
    }

    pub fn repeat_penalty(&mut self, repeat_penalty: f32) {
        self.repeat_penalty = Some(repeat_penalty);
    }

    pub fn temperature(&mut self, temperature: f32) {
        self.temperature = Some(temperature);
    }

    pub fn seed(&mut self, seed: i32) {
        self.seed = Some(seed);
    }

    pub fn stop(&mut self, stop: &str) {
        self.stop = Some(stop.to_string());
    }

    pub fn num_predict(&mut self, num_predict: i32) {
        self.num_predict = Some(num_predict);
    }

    pub fn top_k(&mut self, top_k: i32) {
        self.top_k = Some(top_k);
    }

    pub fn top_p(&mut self, top_p: f32) {
        self.top_p = Some(top_p);
    }

    pub fn min_p(&mut self, min_p: f32) {
        self.min_p = Some(min_p);
    }

    pub fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::Parameter;

    #[test]
    fn is_default_should_work() {
        let mut p = Parameter::default();
        assert!(p.is_default());
        p.mirostat = Some(1);
        assert!(!p.is_default());
    }
}

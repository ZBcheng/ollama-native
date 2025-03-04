use serde::{Deserialize, Serialize};

pub mod completion;
pub mod version;

#[cfg(feature = "model")]
pub mod model;

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
    #[inline]
    fn new(role: Role, content: &str) -> Self {
        Self {
            role,
            content: content.to_string(),
            images: None,
            tool_calls: None,
        }
    }

    #[inline]
    pub fn system(content: &str) -> Self {
        Self::new(Role::System, content)
    }

    #[inline]
    pub fn user(content: &str) -> Self {
        Self::new(Role::User, content)
    }

    #[inline]
    pub fn assistant(content: &str) -> Self {
        Self::new(Role::Assistant, content)
    }

    #[inline]
    pub fn images(mut self, images: Vec<impl ToString>) -> Self {
        let mut cur_images = self.images.unwrap_or_default();
        images
            .into_iter()
            .for_each(|img| cur_images.push(img.to_string()));
        self.images = Some(cur_images);
        self
    }

    #[inline]
    pub fn image(mut self, image: &str) -> Self {
        let mut cur_images = self.images.unwrap_or_default();
        cur_images.push(image.to_string());
        self.images = Some(cur_images);
        self
    }

    #[inline]
    pub fn tool_calls(mut self, tool_calls: Vec<serde_json::Value>) -> Self {
        let mut cur_tool_calls = self.tool_calls.unwrap_or_default();
        tool_calls
            .into_iter()
            .for_each(|tc| cur_tool_calls.push(tc));
        self.tool_calls = Some(cur_tool_calls);
        self
    }

    #[inline]
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
    pub mirostat_eta: Option<f64>,

    /// Controls the balance between coherence and diversity of the output. A lower value
    /// will result in more focused and coherent text.
    /// (Default: 5.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f64>,

    /// Sets the size of the context window used to generate the next token.
    /// (Default: 2048).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i64>,

    /// Sets how far back for the model to look back to prevent repetition.
    /// (Default: 64, 0 = disabled, -1 = num_ctx).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i64>,

    /// Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize
    /// repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.
    /// (Default: 1.1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,

    /// The temperature of the model. Increasing the temperature will make the model answer more creatively.
    /// (Default: 0.8).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Sets the random number seed to use for generation. Setting this to a specific number
    /// will make the model generate the same text for the same prompt.
    /// (Default: 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Sets the stop sequences to use. When this pattern is encountered the LLM will stop
    /// generating text and return. Multiple stop patterns may be set by specifying multiple
    /// separate `stop` parameters in a modelfile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,

    /// Maximum number of tokens to predict when generating text.
    /// (Default: -1, infinite generation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i64>,

    /// Reduces the probability of generating nonsense. A higher value (e.g. 100) will give
    /// more diverse answers, while a lower value (e.g. 10) will be more conservative.
    /// (Default: 40).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,

    /// Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text,
    /// while a lower value (e.g., 0.5) will generate more focused and conservative text.
    /// (Default: 0.9).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter
    /// p represents the minimum probability for a token to be considered, relative to the probability
    /// of the most likely token. For example, with p=0.05 and the most likely token having a probability
    /// of 0.9, logits with a value less than 0.045 are filtered out.
    /// (Default: 0.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
}

impl Parameter {
    #[inline]
    pub fn mirostat(&mut self, mirostat: u8) {
        self.mirostat = Some(mirostat);
    }

    #[inline]
    pub fn mirostat_eta(&mut self, mirostat_eta: f64) {
        self.mirostat_eta = Some(mirostat_eta);
    }

    #[inline]
    pub fn mirostat_tau(&mut self, mirostat_tau: f64) {
        self.mirostat_tau = Some(mirostat_tau);
    }

    #[inline]
    pub fn num_ctx(&mut self, num_ctx: i64) {
        self.num_ctx = Some(num_ctx);
    }

    #[inline]
    pub fn repeat_last_n(&mut self, repeat_last_n: i64) {
        self.repeat_last_n = Some(repeat_last_n);
    }

    #[inline]
    pub fn repeat_penalty(&mut self, repeat_penalty: f64) {
        self.repeat_penalty = Some(repeat_penalty);
    }

    #[inline]
    pub fn temperature(&mut self, temperature: f64) {
        self.temperature = Some(temperature);
    }

    #[inline]
    pub fn seed(&mut self, seed: i64) {
        self.seed = Some(seed);
    }

    #[inline]
    pub fn stop(&mut self, stop: &str) {
        self.stop = Some(stop.to_string());
    }

    #[inline]
    pub fn num_predict(&mut self, num_predict: i64) {
        self.num_predict = Some(num_predict);
    }

    #[inline]
    pub fn top_k(&mut self, top_k: i64) {
        self.top_k = Some(top_k);
    }

    #[inline]
    pub fn top_p(&mut self, top_p: f64) {
        self.top_p = Some(top_p);
    }

    #[inline]
    pub fn min_p(&mut self, min_p: f64) {
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

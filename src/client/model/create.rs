use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use crate::abi::{
    Message,
    model::create::{CreateModelRequest, CreateModelResponse},
};
use crate::client::{Action, ollama::OllamaClient};

impl Action<CreateModelRequest, CreateModelResponse> {
    pub fn new(ollama: Arc<OllamaClient>, model: &str) -> Self {
        let request = CreateModelRequest {
            model: model.to_string(),
            ..Default::default()
        };

        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }

    /// Name of the model to create.
    pub fn model(mut self, model: &str) -> Self {
        self.request.model = model.to_string();
        self
    }

    /// Name of an existing model to create the new model from.
    pub fn from(mut self, from: &str) -> Self {
        self.request.from = Some(from.to_string());
        self
    }

    /// A dictionary of file names to SHA256 digests of blobs to create the model from.
    pub fn files(mut self, files: HashMap<String, String>) -> Self {
        let mut cur_files = self.request.files.unwrap_or_default();
        files.iter().for_each(|(k, v)| {
            cur_files.insert(k.to_string(), v.to_string());
        });
        self.request.files = Some(cur_files);
        self
    }

    /// A dictionary of file names to SHA256 digests of blobs to create the model from.
    pub fn file(mut self, name: &str, sha: &str) -> Self {
        let mut cur_files = self.request.files.unwrap_or_default();
        cur_files.insert(name.to_string(), sha.to_string());
        self.request.files = Some(cur_files);
        self
    }

    /// A dictionary of file names to SHA256 digests of blobs for LORA adapters.
    pub fn adapters(mut self, adapters: HashMap<String, String>) -> Self {
        let mut cur_adapters = self.request.adapters.unwrap_or_default();
        adapters.iter().for_each(|(k, v)| {
            cur_adapters.insert(k.to_string(), v.to_string());
        });
        self.request.adapters = Some(cur_adapters);
        self
    }

    /// A dictionary of file names to SHA256 digests of blobs for LORA adapters.
    pub fn adapter(mut self, name: &str, sha: &str) -> Self {
        let mut cur_adapters = self.request.adapters.unwrap_or_default();
        cur_adapters.insert(name.to_string(), sha.to_string());
        self.request.adapters = Some(cur_adapters);
        self
    }

    /// The prompt template for the model.
    pub fn template(mut self, template: &str) -> Self {
        self.request.template = Some(template.to_string());
        self
    }

    /// A list of strings containing the license or licenses for the model.
    pub fn license(mut self, license: Vec<String>) -> Self {
        self.request.license = license;
        self
    }

    /// A string containing the system prompt for the model.
    pub fn system(mut self, system: &str) -> Self {
        self.request.system = Some(system.to_string());
        self
    }

    /// A list of message objects used to create a conversation.
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.request.messages = messages;
        self
    }

    /// A message objects used to create a conversation.
    pub fn message(mut self, message: &Message) -> Self {
        self.request.messages.push(message.clone());
        self
    }

    /// A system message objects used to create a conversation.
    pub fn system_message(mut self, content: &str) -> Self {
        self.request.messages.push(Message::new_system(content));
        self
    }

    /// A user message objects used to create a conversation.
    pub fn user_message(mut self, content: &str) -> Self {
        self.request.messages.push(Message::new_user(content));
        self
    }

    /// Quantize a non-quantized (e.g. float16) model.
    pub fn quantize(mut self, quantize: &str) -> Self {
        self.request.quantize = Some(quantize.to_string());
        self
    }

    /// Enable Mirostat sampling for controlling perplexity.
    /// (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
    pub fn mirostat(mut self, mirostat: u8) -> Self {
        self.request.parameters.mirostat(mirostat);
        self
    }

    /// Influences how quickly the algorithm responds to feedback from the generated text.
    /// A lower learning rate will result in slower adjustments, while a higher learning
    /// rate will make the algorithm more responsive.
    /// (Default: 0.1).
    pub fn mirostat_eta(mut self, mirostat_eta: f64) -> Self {
        self.request.parameters.mirostat_eta(mirostat_eta);
        self
    }

    /// Controls the balance between coherence and diversity of the output. A lower value
    /// will result in more focused and coherent text.
    /// (Default: 5.0).
    pub fn mirostat_tau(mut self, mirostat_tau: f64) -> Self {
        self.request.parameters.mirostat_tau(mirostat_tau);
        self
    }

    /// Sets the size of the context window used to generate the next token.
    /// (Default: 2048).
    pub fn num_ctx(mut self, num_ctx: i64) -> Self {
        self.request.parameters.num_ctx(num_ctx);
        self
    }

    /// Sets how far back for the model to look back to prevent repetition.
    /// (Default: 64, 0 = disabled, -1 = num_ctx).
    pub fn repeat_last_n(mut self, repeat_last_n: i64) -> Self {
        self.request.parameters.repeat_last_n(repeat_last_n);
        self
    }

    /// Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize
    /// repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.
    /// (Default: 1.1).
    pub fn repeat_penalty(mut self, repeat_penalty: f64) -> Self {
        self.request.parameters.repeat_penalty(repeat_penalty);
        self
    }

    /// The temperature of the model. Increasing the temperature will make the model answer more creatively.
    /// (Default: 0.8).
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.request.parameters.temperature(temperature);
        self
    }

    /// Sets the random number seed to use for generation. Setting this to a specific number
    /// will make the model generate the same text for the same prompt.
    /// (Default: 0).
    pub fn seed(mut self, seed: i64) -> Self {
        self.request.parameters.seed(seed);
        self
    }

    /// Sets the stop sequences to use. When this pattern is encountered the LLM will stop
    /// generating text and return. Multiple stop patterns may be set by specifying multiple
    /// separate `stop` parameters in a modelfile.
    pub fn stop(mut self, stop: &str) -> Self {
        self.request.parameters.stop(stop);
        self
    }

    /// Maximum number of tokens to predict when generating text.
    /// (Default: -1, infinite generation)
    pub fn num_predict(mut self, num_predict: i64) -> Self {
        self.request.parameters.num_predict(num_predict);
        self
    }

    /// Reduces the probability of generating nonsense. A higher value (e.g. 100) will give
    /// more diverse answers, while a lower value (e.g. 10) will be more conservative.
    /// (Default: 40)
    pub fn top_k(mut self, top_k: i64) -> Self {
        self.request.parameters.top_k(top_k);
        self
    }

    /// Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text,
    /// while a lower value (e.g., 0.5) will generate more focused and conservative text.
    /// (Default: 0.9)
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.request.parameters.top_p(top_p);
        self
    }

    /// Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter
    /// p represents the minimum probability for a token to be considered, relative to the probability
    /// of the most likely token. For example, with p=0.05 and the most likely token having a probability
    /// of 0.9, logits with a value less than 0.045 are filtered out.
    /// (Default: 0.0)
    pub fn min_p(mut self, min_p: f64) -> Self {
        self.request.parameters.min_p(min_p);
        self
    }
}

use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};

#[cfg(feature = "stream")]
use {
    crate::client::{IntoStream, OllamaStream},
    async_stream::stream,
    async_trait::async_trait,
    tokio_stream::StreamExt,
};

use crate::{
    abi::completion::{
        chat::Format,
        generate::{GenerateRequest, GenerateResponse},
    },
    client::{Action, ollama::OllamaClient},
    error::OllamaError,
};

impl Action<GenerateRequest, GenerateResponse> {
    pub fn new(ollama: OllamaClient, model: &str, prompt: &str) -> Self {
        let request = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            ..Default::default()
        };

        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }
}

impl Action<GenerateRequest, GenerateResponse> {
    /// The text after the model response.
    pub fn suffix(mut self, suffix: &str) -> Self {
        self.request.suffix = Some(suffix.to_string());
        self
    }

    /// A list of base64-encoded images (for multimodal models such as `llava`).
    pub fn images(mut self, images: Vec<String>) -> Self {
        self.request.images = images;
        self
    }

    /// Return a response in JSON format.
    pub fn format(mut self, format: &str) -> Self {
        let fmt = match format.to_lowercase().as_str() {
            "json" => Format::Json,
            _ => Format::Schema(format.to_string()),
        };
        self.request.format = Some(fmt);
        self
    }

    /// System message to (overrides what is defined in the `Modelfile`).
    pub fn system(mut self, system: &str) -> Self {
        self.request.system = Some(system.to_string());
        self
    }

    /// The prompt template to use (overrides what is defined in the `Modelfile`).
    pub fn template(mut self, template: &str) -> Self {
        self.request.template = Some(template.to_string());
        self
    }

    /// If `true` no formatting will be applied to the prompt. You may choose to use the `raw`
    /// parameter if you are specifying a full templated prompt in your request to the API.
    pub fn raw(mut self) -> Self {
        self.request.raw = Some(true);
        self
    }

    /// Controls how long the model will stay loaded into memory following the request (default: 5m).
    pub fn keep_alive(mut self, keep_alive: i64) -> Self {
        self.request.keep_alive = Some(keep_alive);
        self
    }

    /// Enable Mirostat sampling for controlling perplexity.
    /// (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
    pub fn mirostat(mut self, mirostat: u8) -> Self {
        self.request.options.mirostat(mirostat);
        self
    }

    /// Influences how quickly the algorithm responds to feedback from the generated text.
    /// A lower learning rate will result in slower adjustments, while a higher learning
    /// rate will make the algorithm more responsive.
    /// (Default: 0.1).
    pub fn mirostat_eta(mut self, mirostat_eta: f64) -> Self {
        self.request.options.mirostat_eta(mirostat_eta);
        self
    }

    /// Controls the balance between coherence and diversity of the output. A lower value
    /// will result in more focused and coherent text.
    /// (Default: 5.0).
    pub fn mirostat_tau(mut self, mirostat_tau: f64) -> Self {
        self.request.options.mirostat_tau(mirostat_tau);
        self
    }

    /// Sets the size of the context window used to generate the next token.
    /// (Default: 2048).
    pub fn num_ctx(mut self, num_ctx: i64) -> Self {
        self.request.options.num_ctx(num_ctx);
        self
    }

    /// Sets how far back for the model to look back to prevent repetition.
    /// (Default: 64, 0 = disabled, -1 = num_ctx).
    pub fn repeat_last_n(mut self, repeat_last_n: i64) -> Self {
        self.request.options.repeat_last_n(repeat_last_n);
        self
    }

    /// Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize
    /// repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.
    /// (Default: 1.1).
    pub fn repeat_penalty(mut self, repeat_penalty: f64) -> Self {
        self.request.options.repeat_penalty(repeat_penalty);
        self
    }

    /// The temperature of the model. Increasing the temperature will make the model answer more creatively.
    /// (Default: 0.8).
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.request.options.temperature(temperature);
        self
    }

    /// Sets the random number seed to use for generation. Setting this to a specific number
    /// will make the model generate the same text for the same prompt.
    /// (Default: 0).
    pub fn seed(mut self, seed: i64) -> Self {
        self.request.options.seed(seed);
        self
    }

    /// Sets the stop sequences to use. When this pattern is encountered the LLM will stop
    /// generating text and return. Multiple stop patterns may be set by specifying multiple
    /// separate `stop` parameters in a modelfile.
    pub fn stop(mut self, stop: &str) -> Self {
        self.request.options.stop(stop);
        self
    }

    /// Maximum number of tokens to predict when generating text.
    /// (Default: -1, infinite generation)
    pub fn num_predict(mut self, num_predict: i64) -> Self {
        self.request.options.num_predict(num_predict);
        self
    }

    /// Reduces the probability of generating nonsense. A higher value (e.g. 100) will give
    /// more diverse answers, while a lower value (e.g. 10) will be more conservative.
    /// (Default: 40)
    pub fn top_k(mut self, top_k: i64) -> Self {
        self.request.options.top_k(top_k);
        self
    }

    /// Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text,
    /// while a lower value (e.g., 0.5) will generate more focused and conservative text.
    /// (Default: 0.9)
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.request.options.top_p(top_p);
        self
    }

    /// Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter
    /// p represents the minimum probability for a token to be considered, relative to the probability
    /// of the most likely token. For example, with p=0.05 and the most likely token having a probability
    /// of 0.9, logits with a value less than 0.045 are filtered out.
    /// (Default: 0.0)
    pub fn min_p(mut self, min_p: f64) -> Self {
        self.request.options.min_p(min_p);
        self
    }
}

impl IntoFuture for Action<GenerateRequest, GenerateResponse> {
    type Output = Result<GenerateResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let headers = match self.request.format {
                Some(_) => {
                    let mut headers = HeaderMap::new();
                    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                    Some(headers)
                }
                None => None,
            };
            let reqwest_resp = self.ollama.post(&self.request, headers).await?;
            let response = reqwest_resp
                .json()
                .await
                .map_err(|e| OllamaError::DecodingError(e))?;
            Ok(response)
        })
    }
}

#[cfg(feature = "stream")]
#[async_trait]
impl IntoStream<GenerateResponse> for Action<GenerateRequest, GenerateResponse> {
    async fn stream(mut self) -> Result<OllamaStream<GenerateResponse>, OllamaError> {
        self.request.stream = true;
        let mut reqwest_stream = self.ollama.post(&self.request, None).await?.bytes_stream();

        let s = stream! {
            while let Some(item) = reqwest_stream.next().await {
                match item {
                    Ok(chunk) => match serde_json::from_slice(&chunk) {
                        Ok(r) => yield Ok(r),
                        Err(e) => yield Err(OllamaError::StreamDecodingError(e.to_string())),
                    }
                    Err(e) => yield Err(OllamaError::DecodingError(e))
                }
            };
        };
        Ok(Box::pin(s))
    }
}

use std::marker::PhantomData;

use futures::future::BoxFuture;

use crate::{
    abi::model::generate_embeddings::{GenerateEmbeddingsRequest, GenerateEmbeddingsResponse},
    client::{Action, ollama::OllamaClient},
    error::OllamaError,
};

impl Action<GenerateEmbeddingsRequest, GenerateEmbeddingsResponse> {
    pub fn new(ollama: OllamaClient, model: &str) -> Self {
        let request = GenerateEmbeddingsRequest {
            model: model.to_string(),
            ..Default::default()
        };

        Self {
            ollama,
            request,
            _resp: PhantomData,
        }
    }

    /// Text to generate embeddings for.
    pub fn input(mut self, input: &str) -> Self {
        self.request.input.push(input.to_string());
        self
    }

    /// List of text to generate embeddings for.
    pub fn inputs(mut self, inputs: &Vec<impl ToString>) -> Self {
        inputs
            .iter()
            .for_each(|input| self.request.input.push(input.to_string()));
        self
    }

    /// Truncates the end of each input to fit within context length.
    /// Returns error if `false` and context length is exceeded. Defaults to `true`.
    pub fn truncate(mut self, truncate: bool) -> Self {
        if truncate == false {
            self.request.truncate = Some(false);
        }
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

impl IntoFuture for Action<GenerateEmbeddingsRequest, GenerateEmbeddingsResponse> {
    type Output = Result<GenerateEmbeddingsResponse, OllamaError>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let reqwest_resp = self.ollama.post(&self.request, None).await?;
            let response = reqwest_resp
                .json()
                .await
                .map_err(|e| OllamaError::DecodingError(e))?;
            Ok(response)
        })
    }
}

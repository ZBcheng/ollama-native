use std::marker::PhantomData;

use futures::future::BoxFuture;
use reqwest::{
    StatusCode,
    header::{CONTENT_TYPE, HeaderMap, HeaderValue},
};

use crate::abi::{
    Message,
    completion::chat::{
        ChatCompletionModelResponse, ChatCompletionRequest, ChatCompletionResponse, Format, Tool,
    },
};
use crate::action::parse_response;
use crate::error::OllamaError;
use crate::{action::OllamaClient, error::OllamaServerError};

#[cfg(feature = "stream")]
use {
    crate::action::{IntoStream, OllamaStream},
    async_stream::stream,
    async_trait::async_trait,
    tokio_stream::StreamExt,
};

pub struct ChatAction<'a, R> {
    request: ChatCompletionRequest<'a>,
    ollama: OllamaClient,
    _resp: PhantomData<R>,
}

impl<'a> ChatAction<'a, ChatCompletionResponse> {
    pub fn new(ollama: OllamaClient, model: &'a str) -> ChatAction<'a, ChatCompletionResponse> {
        Self {
            ollama,
            request: ChatCompletionRequest::new(model),
            _resp: PhantomData::<ChatCompletionResponse>,
        }
    }
}

impl<'a> ChatAction<'a, ChatCompletionResponse> {
    /// Load the model into memory.
    #[inline]
    pub fn load(self) -> ChatAction<'a, ChatCompletionModelResponse> {
        ChatAction {
            ollama: self.ollama,
            request: self.request.to_load_model(),
            _resp: PhantomData::<ChatCompletionModelResponse>,
        }
    }

    /// Unload the model from memory.
    #[inline]
    pub fn unload(self) -> ChatAction<'a, ChatCompletionModelResponse> {
        ChatAction {
            ollama: self.ollama,
            request: self.request.to_unload_model(),
            _resp: PhantomData::<ChatCompletionModelResponse>,
        }
    }

    #[inline]
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        messages
            .into_iter()
            .for_each(|m| self.request.messages.push(m.to_owned()));
        self
    }

    #[inline]
    pub fn message(mut self, message: Message) -> Self {
        self.request.messages.push(message);
        self
    }

    #[inline]
    pub fn system_message(mut self, content: &'a str) -> Self {
        self.request.messages.push(Message::system(content));
        self
    }

    #[inline]
    pub fn user_message(mut self, content: &'a str) -> Self {
        self.request.messages.push(Message::user(content));
        self
    }

    #[inline]
    pub fn assistant_message(mut self, content: &'a str) -> Self {
        self.request.messages.push(Message::assistant(content));
        self
    }

    /// Tool in JSON for the model to use if supported.
    #[inline]
    pub fn tool(mut self, tool: &'a str) -> Self {
        self.request.tools.push(Tool::new(tool));
        self
    }

    /// List of tools in JSON for the model to use if supported.
    #[inline]
    pub fn tools(mut self, tools: Vec<&'a str>) -> Self {
        self.request.tools = tools.into_iter().map(|t| Tool::new(t)).collect();
        self
    }

    /// Return a response in JSON format.
    #[inline]
    pub fn format(mut self, format: &'a str) -> Self {
        let fmt = match format.to_lowercase().as_str() {
            "json" => Format::Json,
            _ => Format::Schema(format),
        };
        self.request.format = Some(fmt);
        self
    }

    /// Controls how long the model will stay loaded into memory following the request (default: 5m).
    #[inline]
    pub fn keep_alive(mut self, keep_alive: i64) -> Self {
        self.request.keep_alive = Some(keep_alive);
        self
    }

    /// Enable Mirostat sampling for controlling perplexity.
    /// (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
    #[inline]
    pub fn mirostat(mut self, mirostat: u8) -> Self {
        self.request.options.mirostat(mirostat);
        self
    }

    /// Influences how quickly the algorithm responds to feedback from the generated text.
    /// A lower learning rate will result in slower adjustments, while a higher learning
    /// rate will make the algorithm more responsive.
    /// (Default: 0.1).
    #[inline]
    pub fn mirostat_eta(mut self, mirostat_eta: f64) -> Self {
        self.request.options.mirostat_eta(mirostat_eta);
        self
    }

    /// Controls the balance between coherence and diversity of the output. A lower value
    /// will result in more focused and coherent text.
    /// (Default: 5.0).
    #[inline]
    pub fn mirostat_tau(mut self, mirostat_tau: f64) -> Self {
        self.request.options.mirostat_tau(mirostat_tau);
        self
    }

    /// Sets the size of the context window used to generate the next token.
    /// (Default: 2048).
    #[inline]
    pub fn num_ctx(mut self, num_ctx: i64) -> Self {
        self.request.options.num_ctx(num_ctx);
        self
    }

    /// Sets how far back for the model to look back to prevent repetition.
    /// (Default: 64, 0 = disabled, -1 = num_ctx).
    #[inline]
    pub fn repeat_last_n(mut self, repeat_last_n: i64) -> Self {
        self.request.options.repeat_last_n(repeat_last_n);
        self
    }

    /// Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize
    /// repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.
    /// (Default: 1.1).
    #[inline]
    pub fn repeat_penalty(mut self, repeat_penalty: f64) -> Self {
        self.request.options.repeat_penalty(repeat_penalty);
        self
    }

    /// The temperature of the model. Increasing the temperature will make the model answer more creatively.
    /// (Default: 0.8).
    #[inline]
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.request.options.temperature(temperature);
        self
    }

    /// Sets the random number seed to use for generation. Setting this to a specific number
    /// will make the model generate the same text for the same prompt.
    /// (Default: 0).
    #[inline]
    pub fn seed(mut self, seed: i64) -> Self {
        self.request.options.seed(seed);
        self
    }

    /// Sets the stop sequences to use. When this pattern is encountered the LLM will stop
    /// generating text and return. Multiple stop patterns may be set by specifying multiple
    /// separate `stop` parameters in a modelfile.
    #[inline]
    pub fn stop(mut self, stop: &str) -> Self {
        self.request.options.stop(stop);
        self
    }

    /// Maximum number of tokens to predict when generating text.
    /// (Default: -1, infinite generation)
    #[inline]
    pub fn num_predict(mut self, num_predict: i64) -> Self {
        self.request.options.num_predict(num_predict);
        self
    }

    /// Reduces the probability of generating nonsense. A higher value (e.g. 100) will give
    /// more diverse answers, while a lower value (e.g. 10) will be more conservative.
    /// (Default: 40)
    #[inline]
    pub fn top_k(mut self, top_k: i64) -> Self {
        self.request.options.top_k(top_k);
        self
    }

    /// Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text,
    /// while a lower value (e.g., 0.5) will generate more focused and conservative text.
    /// (Default: 0.9)
    #[inline]
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.request.options.top_p(top_p);
        self
    }

    /// Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter
    /// p represents the minimum probability for a token to be considered, relative to the probability
    /// of the most likely token. For example, with p=0.05 and the most likely token having a probability
    /// of 0.9, logits with a value less than 0.045 are filtered out.
    /// (Default: 0.0)
    #[inline]
    pub fn min_p(mut self, min_p: f64) -> Self {
        self.request.options.min_p(min_p);
        self
    }
}

impl<'a> IntoFuture for ChatAction<'a, ChatCompletionResponse> {
    type Output = Result<ChatCompletionResponse, OllamaError>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let headers = if let Some(_) = self.request.format {
                let mut headers = HeaderMap::new();
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                Some(headers)
            } else {
                None
            };

            let reqwest_resp = self.ollama.post(&self.request, headers).await?;
            match reqwest_resp.status() {
                StatusCode::OK => parse_response(reqwest_resp).await,
                _code => {
                    let error: OllamaServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::OllamaServerError(error.error))
                }
            }
        })
    }
}

impl<'a> IntoFuture for ChatAction<'a, ChatCompletionModelResponse> {
    type Output = Result<ChatCompletionModelResponse, OllamaError>;
    type IntoFuture = BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let reqwest_resp = self.ollama.post(&self.request, None).await?;
            match reqwest_resp.status() {
                StatusCode::OK => parse_response(reqwest_resp).await,
                _code => {
                    let error: OllamaServerError = parse_response(reqwest_resp).await?;
                    Err(OllamaError::OllamaServerError(error.error))
                }
            }
        })
    }
}

#[cfg(feature = "stream")]
#[async_trait]
impl<'a> IntoStream<ChatCompletionResponse> for ChatAction<'a, ChatCompletionResponse> {
    async fn stream(mut self) -> Result<OllamaStream<ChatCompletionResponse>, OllamaError> {
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

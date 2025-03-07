use std::collections::HashMap;

use futures::future::BoxFuture;
use reqwest::StatusCode;

use crate::action::OllamaClient;
use crate::error::OllamaServerError;
use crate::{
    abi::{
        Message,
        model::create::{CreateModelRequest, CreateModelResponse},
    },
    action::parse_response,
};

#[cfg(feature = "stream")]
use {
    crate::action::{IntoStream, OllamaStream},
    async_stream::stream,
    async_trait::async_trait,
    tokio_stream::StreamExt,
};

#[cfg(feature = "model")]
use crate::error::OllamaError;

pub struct CreateModelAction<'a> {
    ollama: OllamaClient,
    request: CreateModelRequest<'a>,
}

impl<'a> CreateModelAction<'a> {
    pub fn new(ollama: OllamaClient, model: &'a str) -> Self {
        let request = CreateModelRequest {
            model,
            ..Default::default()
        };

        Self { ollama, request }
    }

    /// Name of the model to create.
    #[inline]
    pub fn model(mut self, model: &'a str) -> Self {
        self.request.model = model;
        self
    }

    /// Name of an existing model to create the new model from.
    #[inline]
    pub fn from(mut self, from: &'a str) -> Self {
        self.request.from = Some(from);
        self
    }

    /// A dictionary of file names to SHA256 digests of blobs to create the model from.
    #[inline]
    pub fn files(mut self, files: HashMap<&'a str, &'a str>) -> Self {
        let mut cur_files = self.request.files.unwrap_or_default();
        files.iter().for_each(|(k, v)| {
            cur_files.insert(k, v);
        });
        self.request.files = Some(cur_files);
        self
    }

    /// A dictionary of file names to SHA256 digests of blobs to create the model from.
    #[inline]
    pub fn file(mut self, name: &'a str, sha: &'a str) -> Self {
        let mut cur_files = self.request.files.unwrap_or_default();
        cur_files.insert(name, sha);
        self.request.files = Some(cur_files);
        self
    }

    /// A dictionary of file names to SHA256 digests of blobs for LORA adapters.
    #[inline]
    pub fn adapters(mut self, adapters: HashMap<&'a str, &'a str>) -> Self {
        let mut cur_adapters = self.request.adapters.unwrap_or_default();
        adapters.iter().for_each(|(k, v)| {
            cur_adapters.insert(k, v);
        });
        self.request.adapters = Some(cur_adapters);
        self
    }

    /// A dictionary of file names to SHA256 digests of blobs for LORA adapters.
    #[inline]
    pub fn adapter(mut self, name: &'a str, sha: &'a str) -> Self {
        let mut cur_adapters = self.request.adapters.unwrap_or_default();
        cur_adapters.insert(name, sha);
        self.request.adapters = Some(cur_adapters);
        self
    }

    /// The prompt template for the model.
    #[inline]
    pub fn template(mut self, template: &'a str) -> Self {
        self.request.template = Some(template);
        self
    }

    /// A list of strings containing the license or licenses for the model.
    #[inline]
    pub fn license(mut self, license: Vec<&'a str>) -> Self {
        license
            .into_iter()
            .for_each(|l| self.request.license.push(l));
        self
    }

    /// A string containing the system prompt for the model.
    #[inline]
    pub fn system(mut self, system: &'a str) -> Self {
        self.request.system = Some(system);
        self
    }

    /// A list of message objects used to create a conversation.
    #[inline]
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        messages
            .into_iter()
            .for_each(|m| self.request.messages.push(m));
        self
    }

    /// A message objects used to create a conversation.
    #[inline]
    pub fn message(mut self, message: Message) -> Self {
        self.request.messages.push(message);
        self
    }

    /// A system message objects used to create a conversation.
    #[inline]
    pub fn system_message(mut self, content: &'a str) -> Self {
        self.request.messages.push(Message::system(content));
        self
    }

    /// A user message objects used to create a conversation.
    #[inline]
    pub fn user_message(mut self, content: &'a str) -> Self {
        self.request.messages.push(Message::user(content));
        self
    }

    /// Quantize a non-quantized (e.g. float16) model.
    #[inline]
    pub fn quantize(mut self, quantize: &'a str) -> Self {
        self.request.quantize = Some(quantize);
        self
    }

    /// Enable Mirostat sampling for controlling perplexity.
    /// (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
    #[inline]
    pub fn mirostat(mut self, mirostat: u8) -> Self {
        self.request.parameters.mirostat(mirostat);
        self
    }

    /// Influences how quickly the algorithm responds to feedback from the generated text.
    /// A lower learning rate will result in slower adjustments, while a higher learning
    /// rate will make the algorithm more responsive.
    /// (Default: 0.1).
    #[inline]
    pub fn mirostat_eta(mut self, mirostat_eta: f64) -> Self {
        self.request.parameters.mirostat_eta(mirostat_eta);
        self
    }

    /// Controls the balance between coherence and diversity of the output. A lower value
    /// will result in more focused and coherent text.
    /// (Default: 5.0).
    #[inline]
    pub fn mirostat_tau(mut self, mirostat_tau: f64) -> Self {
        self.request.parameters.mirostat_tau(mirostat_tau);
        self
    }

    /// Sets the size of the context window used to generate the next token.
    /// (Default: 2048).
    #[inline]
    pub fn num_ctx(mut self, num_ctx: i64) -> Self {
        self.request.parameters.num_ctx(num_ctx);
        self
    }

    /// Sets how far back for the model to look back to prevent repetition.
    /// (Default: 64, 0 = disabled, -1 = num_ctx).
    #[inline]
    pub fn repeat_last_n(mut self, repeat_last_n: i64) -> Self {
        self.request.parameters.repeat_last_n(repeat_last_n);
        self
    }

    /// Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize
    /// repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.
    /// (Default: 1.1).
    #[inline]
    pub fn repeat_penalty(mut self, repeat_penalty: f64) -> Self {
        self.request.parameters.repeat_penalty(repeat_penalty);
        self
    }

    /// The temperature of the model. Increasing the temperature will make the model answer more creatively.
    /// (Default: 0.8).
    #[inline]
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.request.parameters.temperature(temperature);
        self
    }

    /// Sets the random number seed to use for generation. Setting this to a specific number
    /// will make the model generate the same text for the same prompt.
    /// (Default: 0).
    #[inline]
    pub fn seed(mut self, seed: i64) -> Self {
        self.request.parameters.seed(seed);
        self
    }

    /// Sets the stop sequences to use. When this pattern is encountered the LLM will stop
    /// generating text and return. Multiple stop patterns may be set by specifying multiple
    /// separate `stop` parameters in a modelfile.
    #[inline]
    pub fn stop(mut self, stop: &str) -> Self {
        self.request.parameters.stop(stop);
        self
    }

    /// Maximum number of tokens to predict when generating text.
    /// (Default: -1, infinite generation)
    #[inline]
    pub fn num_predict(mut self, num_predict: i64) -> Self {
        self.request.parameters.num_predict(num_predict);
        self
    }

    /// Reduces the probability of generating nonsense. A higher value (e.g. 100) will give
    /// more diverse answers, while a lower value (e.g. 10) will be more conservative.
    /// (Default: 40)
    #[inline]
    pub fn top_k(mut self, top_k: i64) -> Self {
        self.request.parameters.top_k(top_k);
        self
    }

    /// Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text,
    /// while a lower value (e.g., 0.5) will generate more focused and conservative text.
    /// (Default: 0.9)
    #[inline]
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.request.parameters.top_p(top_p);
        self
    }

    /// Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter
    /// p represents the minimum probability for a token to be considered, relative to the probability
    /// of the most likely token. For example, with p=0.05 and the most likely token having a probability
    /// of 0.9, logits with a value less than 0.045 are filtered out.
    /// (Default: 0.0)
    #[inline]
    pub fn min_p(mut self, min_p: f64) -> Self {
        self.request.parameters.min_p(min_p);
        self
    }
}

#[cfg(feature = "model")]
impl<'a> IntoFuture for CreateModelAction<'a> {
    type Output = Result<CreateModelResponse, OllamaError>;
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
impl<'a> IntoStream<CreateModelResponse> for CreateModelAction<'a> {
    async fn stream(mut self) -> Result<OllamaStream<CreateModelResponse>, OllamaError> {
        self.request.stream = true;

        let mut reqwest_stream = self.ollama.post(&self.request, None).await?.bytes_stream();

        let s = stream! {
            while let Some(stream_item) = reqwest_stream.next().await {
                match stream_item {
                    Ok(chunks) => match parse_chunks(&chunks) {
                        Ok(r) => for c in r {
                            yield Ok(c)
                        },
                        Err(e) => yield Err(e.into()),
                    }
                    Err(e) => yield Err(OllamaError::DecodingError(e))
                }

            };
        };

        Ok(Box::pin(s))
    }
}

#[cfg(feature = "stream")]
fn parse_chunks(chunks: &[u8]) -> Result<Vec<CreateModelResponse>, OllamaError> {
    let chunks = std::str::from_utf8(&chunks).map_err(|e| {
        OllamaError::StreamDecodingError(format!("failed to parse chunk to utf8: {e}"))
    })?;

    let splitted: Vec<&str> = chunks.trim().split('\n').collect();

    let mut resp = vec![];

    for sp in splitted {
        let deserialized: CreateModelResponse = serde_json::from_str(sp).map_err(|e| {
            OllamaError::StreamDecodingError(format!(
                "failed to deserialize PullModelResponse from {sp}: {e}",
            ))
        })?;
        resp.push(deserialized);
    }

    Ok(resp)
}

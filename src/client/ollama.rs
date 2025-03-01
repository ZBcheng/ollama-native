use std::sync::Arc;

use crate::abi::completion::{
    chat::{ChatRequest, ChatResponse},
    generate::{GenerateRequest, GenerateResponse},
};
use crate::abi::version::{VersionRequest, VersionResponse};
use crate::config::OllamaConfig;
use crate::error::OllamaError;

use super::{Action, OllamaRequest};

#[cfg(feature = "model")]
use crate::abi::model::{
    check_blob_exists::{CheckBlobExistsRequest, CheckBlobExistsResponse},
    copy::{CopyModelRequest, CopyModelResponse},
    create::{CreateModelRequest, CreateModelResponse},
    delete::{DeleteModelRequest, DeleteModelResponse},
    generate_embedding::{GenerateEmbeddingRequest, GenerateEmbeddingResponse},
    generate_embeddings::{GenerateEmbeddingsRequest, GenerateEmbeddingsResponse},
    list_local::{ListLocalModelsRequest, ListLocalModelsResponse},
    list_running::{ListRunningModelsRequest, ListRunningModelsResponse},
    pull::{PullModelRequest, PullModelResponse},
    push::{PushModelRequest, PushModelResponse},
    push_blob::{PushBlobRequest, PushBlobResponse},
    show_info::{ShowModelInformationRequest, ShowModelInformationResponse},
};

#[derive(Clone)]
pub struct OllamaClient {
    pub cli: reqwest::Client,
    pub config: Arc<OllamaConfig>,
}

impl OllamaClient {
    pub fn new(config: OllamaConfig) -> Self {
        let cli = reqwest::Client::new();
        let config = Arc::new(config);
        Self { cli, config }
    }

    pub fn url(&self) -> String {
        self.config.url.to_string()
    }

    pub async fn post(
        &self,
        request: &impl OllamaRequest,
    ) -> Result<reqwest::Response, OllamaError> {
        let serialized =
            serde_json::to_vec(&request).map_err(|e| OllamaError::InvalidFormat(e.to_string()))?;

        let url = format!("{}{}", self.config.url, request.path());
        let response = self
            .cli
            .post(url)
            .body(serialized)
            .send()
            .await
            .map_err(|e| OllamaError::RequestError(e))?;
        Ok(response)
    }

    pub async fn get(
        &self,
        request: &impl OllamaRequest,
    ) -> Result<reqwest::Response, OllamaError> {
        let url = format!("{}{}", self.config.url, request.path());
        let response = self
            .cli
            .get(url)
            .send()
            .await
            .map_err(|e| OllamaError::RequestError(e))?;
        Ok(response)
    }
}

pub struct Ollama {
    client: OllamaClient,
}

// Default feature
impl Ollama {
    pub fn new(url: &str) -> Self {
        let config = OllamaConfig::from_url(url);
        let client = OllamaClient::new(config);
        Self { client }
    }

    /// Generate a response for a given prompt with a provided model.
    /// The final response object will include statistics and additional data from the request.
    ///
    /// # Parameters
    /// - `model`: The model to use for generating the response.
    /// - `prompt`: The prompt to generate a response for.
    /// - `suffix`: (optional) The text after the model response.
    /// - `images`: (optional) A list of base64-encoded images (for multimodal models such as `llava`).
    /// - `format`: (optional) The format to return a response in. Format can be `json` or a JSON schema.
    /// - `options`: (optional) Additional model parameters listed in the documentation for the Modelfile such as temperature.
    /// - `system`: (optional) System message to (overrides what is defined in the `Modelfile`).
    /// - `template`: (optional) The prompt template to use (overrides what is defined in the `Modelfile`).
    /// - `stream`: (optional) If `false` the response will be returned as a single response object, rather than a stream of objects.
    /// - `raw`: (optional) If `true` no formatting will be applied to the prompt. You may choose to use the `raw` parameter if you are specifying a full templated prompt in your request to the API.
    /// - `keep_alive`: (optional) Controls how long the model will stay loaded into memory following the request (default: 5m).
    ///
    /// # Errors
    /// - `OllamaError::RequestError`: There is an error with the request.
    /// - `OllamaError::DecodeError`: There is an error decoding the response.
    /// - `OllamaError::StreamDecodingError`: There is an error decoding the stream.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use ollama_native::Ollama;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let ollama = Ollama::new("http://localhost:11434");
    ///
    ///     let response = ollama.generate("llama3.1:8b", "Tell me a joke about sharks").await?;
    ///     println!("{}", response.response);
    ///     Ok(())
    /// }
    pub fn generate(&self, model: &str, prompt: &str) -> Action<GenerateRequest, GenerateResponse> {
        Action::<GenerateRequest, GenerateResponse>::new(self.client.clone(), model, prompt)
    }

    /// Generate the next message in a chat with a provided model. This is a streaming endpoint,
    /// so there will be a series of responses. The final response object will include statistics and
    /// additional data from the request.
    ///
    /// # Parameters
    /// - `model`: The model to use for generating the response.
    /// - `messages`: The messages of the chat.
    /// - `tools`: (optional) List of tools in JSON for the model to use if supported
    /// - `format`: (optional) The format to return a response in. Format can be `json` or a JSON schema.
    /// - `options`: (optional) Additional model parameters listed in the documentation for the Modelfile such as `temperature`.
    /// - `keep_alive`: (optional) Controls how long the model will stay loaded into memory following the request (default: 5m).
    /// - `stream`: (optional) If `false` the response will be returned as a single response object, rather than a stream of objects.
    ///
    /// # Errors
    /// - `OllamaError::RequestError`: There is an error with the request.
    /// - `OllamaError::DecodeError`: There is an error decoding the response.
    /// - `OllamaError::StreamDecodingError`: There is an error decoding the stream.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use ollama_native::{Message, Ollama};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let ollama = Ollama::new("http://localhost:11434");
    ///     
    ///     // Generate a completion in json format.
    ///     let response = ollama
    ///         .chat("llama3.1:8b")
    ///         .system_message("You are a robot who likes to tell jokes")
    ///         .user_message("Who are you?")
    ///         .format("json")
    ///         .await?;
    ///
    ///     println!("{}", response.message.unwrap().content);
    ///
    ///     // Or use a Vec of messages.
    ///     let messages = vec![
    ///         Message::new_system("You are a robot who likes to tell jokes"),
    ///         Message::new_user("Who are you?"),
    ///     ];
    ///
    ///     let response = ollama
    ///         .chat("llama3.1:8b")
    ///         .messages(&messages)
    ///         .format("json")
    ///         .await?;
    ///
    ///     println!("{}", response.message.unwrap().content);
    ///
    ///     Ok(())
    /// }
    pub fn chat(&self, model: &str) -> Action<ChatRequest, ChatResponse> {
        Action::<ChatRequest, ChatResponse>::new(self.client.clone(), model)
    }

    /// Retrieve the Ollama version.
    ///
    /// # Errors
    /// - `OllamaError::RequestError`: There is an error with the request.
    /// - `OllamaError::DecodeError`: There is an error decoding the response.
    pub fn version(&self) -> Action<VersionRequest, VersionResponse> {
        Action::<VersionRequest, VersionResponse>::new(self.client.clone())
    }
}

#[cfg(feature = "model")]
impl Ollama {
    /// Create a model from:
    /// - another model;
    /// - a safetensors directory; or
    /// - a GGUF file.
    /// If you are creating a model from a safetensors directory or from a GGUF file, you must `create
    /// a blob` for each of the files and then use the file name and SHA256 digest associated with each
    /// blob in the files field.
    ///
    /// # Parameters
    /// - `model`: The model to create.
    /// - `from`: (optional) Name of an existing model to create the new model from.
    /// - `files`: (optional) A dictionary of file names to SHA256 digests of blobs to create the model from.
    /// - `adapters`: (optional) A dictionary of file names to SHA256 digests of blobs for LORA adapters.
    /// - `template`: (optional) The prompt template for the model.
    /// - `license`: (optional) A string or list of strings containing the license or licenses for the model.
    /// - `system`: (optional) A string containing the system prompt for the model.
    /// - `parameters`: (optional) A dictionary of parameters for the model.
    /// - `messages`: (optional) A list of messages objects used to create a conversation.
    /// - `quantize`: (optional) Quantize a non-quantized (e.g. float16) model.
    /// - `stream`: (optional) If `false` the response will be returned as a single response object, rather than a stream of objects.
    ///
    /// # Errors
    /// - `OllamaError::RequestError`: There is an error with the request.
    /// - `OllamaError::DecodeError`: There is an error decoding the response.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use ollama_native::Ollama;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let ollama = Ollama::new("http://localhost:11434");
    ///
    ///     // Create a new model from an existing model.
    ///     let _ = ollama
    ///         .create_model("mario")
    ///         .from("llama3.1:8b")
    ///         .system("You are Mario from Super Mario Bros.")
    ///         .await?;
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn create_model(&self, model: &str) -> Action<CreateModelRequest, CreateModelResponse> {
        Action::<CreateModelRequest, CreateModelResponse>::new(self.client.clone(), model)
    }

    /// List models that are available locally.
    ///
    /// # Errors
    /// - `OllamaError::RequestError`: There is an error with the request.
    /// - `OllamaError::DecodeError`: There is an error decoding the response.
    pub fn list_local_models(&self) -> Action<ListLocalModelsRequest, ListLocalModelsResponse> {
        Action::<ListLocalModelsRequest, ListLocalModelsResponse>::new(self.client.clone())
    }

    /// List models that are currently loaded into memory.
    /// # Errors
    /// - `OllamaError::RequestError`: There is an error with the request.
    /// - `OllamaError::DecodeError`: There is an error decoding the response.
    pub fn list_running_models(
        &self,
    ) -> Action<ListRunningModelsRequest, ListRunningModelsResponse> {
        Action::<ListRunningModelsRequest, ListRunningModelsResponse>::new(self.client.clone())
    }

    /// Show information about a model including details, modelfile, template, parameters, license, system prompt.
    /// # Parameters
    /// - `model`: Name of the model to show.
    /// - `verbose`: (optional) If set to `true`, returns full data for verbose response fields
    ///
    /// # Errors
    /// - `OllamaError::RequestError`: There is an error with the request.
    /// - `OllamaError::DecodeError`: There is an error decoding the response.
    pub fn show_model_information(
        &self,
        model: &str,
    ) -> Action<ShowModelInformationRequest, ShowModelInformationResponse> {
        Action::<ShowModelInformationRequest, ShowModelInformationResponse>::new(
            self.client.clone(),
            model,
        )
    }

    /// Copy a model. Creates a model with another name from an existing model.
    ///
    /// # Errors
    /// - `OllamaError::ModelDoesNotExist`: The model does not exist.
    /// - `OllamaError::RequestError`: There is an error with the request.
    /// - `OllamaError::DecodeError`: There is an error decoding the response.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use ollama_native::{Ollama, OllamaError};
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let ollama = Ollama::new("http://localhost:11434");
    ///
    ///     match ollama.copy_model("llama3.2", "llama3-backup").await {
    ///        Ok(_) => println!("Model copied successfully"),
    ///        Err(OllamaError::ModelDoesNotExist) => println!("Model does not exist"),
    ///        Err(e) => println!("Error copying model: {e}"),
    ///    }
    /// }
    pub fn copy_model(
        &self,
        source: &str,
        destination: &str,
    ) -> Action<CopyModelRequest, CopyModelResponse> {
        Action::<CopyModelRequest, CopyModelResponse>::new(self.client.clone(), source, destination)
    }

    /// Delete a model and its data.
    pub fn delete_model(&self, model: &str) -> Action<DeleteModelRequest, DeleteModelResponse> {
        Action::<DeleteModelRequest, DeleteModelResponse>::new(self.client.clone(), model)
    }

    /// Download a model from the ollama library. Cancelled pulls are resumed
    /// from where they left off, and multiple calls will share the same download progress.
    pub fn pull_model(&self, model: &str) -> Action<PullModelRequest, PullModelResponse> {
        Action::<PullModelRequest, PullModelResponse>::new(self.client.clone(), model)
    }

    /// Upload a model to a model library. Requires registering for ollama.ai and adding a public key first.
    pub fn push_model(&self, model: &str) -> Action<PushModelRequest, PushModelResponse> {
        Action::<PushModelRequest, PushModelResponse>::new(self.client.clone(), model)
    }

    /// Generate embeddings from a model.
    pub fn generate_embedding(
        &self,
        model: &str,
        prompt: &str,
    ) -> Action<GenerateEmbeddingRequest, GenerateEmbeddingResponse> {
        Action::<GenerateEmbeddingRequest, GenerateEmbeddingResponse>::new(
            self.client.clone(),
            model,
            prompt,
        )
    }

    /// Generate embeddings from a model.
    pub fn generate_embeddings(
        &self,
        model: &str,
    ) -> Action<GenerateEmbeddingsRequest, GenerateEmbeddingsResponse> {
        Action::<GenerateEmbeddingsRequest, GenerateEmbeddingsResponse>::new(
            self.client.clone(),
            model,
        )
    }

    /// Ensures that the file blob (Binary Large Object) used with create a model exists on the server.
    /// This checks your Ollama server and not ollama.com.
    ///
    /// # Parameters
    /// - `digest`: The SHA256 digest of the blob.
    ///
    /// # Returns
    /// - `Ok(CheckBlobExistsResponse {})` if blob exists.
    /// - `Err(OllamaError::BlobDoesNotExist)` if blob does not exist.
    pub fn check_blob_exists(
        &self,
        digest: &str,
    ) -> Action<CheckBlobExistsRequest, CheckBlobExistsResponse> {
        Action::<CheckBlobExistsRequest, CheckBlobExistsResponse>::new(self.client.clone(), digest)
    }

    /// Push a file to the Ollama server to create a "blob" (Binary Large Object).
    ///
    /// # Parameters
    /// - `file`: The file you want to push.
    /// - `digest`: The expected SHA256 digest of the file.
    ///
    /// # Returns
    /// - `Ok(PushBlobRequest {})` if the blob was successfully created.
    /// - `Err(OllamaError::UnexpectedDigest)` if the digest used is not expected.
    pub fn push_blob(&self, file: &str, digest: &str) -> Action<PushBlobRequest, PushBlobResponse> {
        Action::<PushBlobRequest, PushBlobResponse>::new(self.client.clone(), file, digest)
    }
}

#[cfg(test)]
mod tests {

    use serde::Serialize;
    use tokio::io::{AsyncWriteExt, stdout};
    use tokio_stream::StreamExt;

    use crate::{
        abi::Message,
        client::{IntoStream, OllamaStream, ollama::Ollama},
    };

    #[tokio::test]
    #[ignore]
    async fn generate_should_work() {
        let ollama = Ollama::new(mock_config());
        let resp = ollama.generate("llama3.1:8b", "hello").await.unwrap();
        println!("{:?}", resp);
    }

    #[tokio::test]
    #[ignore]
    async fn generate_stream_should_work() {
        let ollama = Ollama::new(mock_config());
        let mut s = ollama
            .generate("llama3.1:8b", "Tell me a joke about sharks")
            .stream()
            .await
            .unwrap();

        let mut out = stdout();
        while let Some(item) = s.next().await {
            out.write(item.unwrap().response.as_bytes()).await.unwrap();
            out.flush().await.unwrap();
        }

        out.write(b"\n").await.unwrap();
        out.flush().await.unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn chat_should_work() {
        let ollama = Ollama::new(mock_config());
        let weather_tool = r#"{"type": "function","function": {"name": "get_current_weather","description": "Get the current weather for a location"}}"#;
        let jokel_tool = r#"{"type": "function","function": {"name": "tell_joke","description": "Tell a joke about a topic given by user"}}"#;

        let resp = ollama
            .chat("llama3.1:8b")
            .system_message("You are an expert on sharks")
            .user_message("Tell me a joke about sharks?")
            .tool(&weather_tool)
            .tool(&jokel_tool)
            .await
            .unwrap();
        println!("{:?}", resp);
    }

    #[tokio::test]
    #[ignore]
    async fn chat_stream_should_work() {
        let ollama = Ollama::new(mock_config());
        let mut s = ollama
            .chat("llama3.2:1b")
            .system_message("You are an expiret on sharks")
            .user_message("Who are you")
            .stream()
            .await
            .unwrap();

        let mut out = stdout();
        while let Some(item) = s.next().await {
            out.write(item.unwrap().message.unwrap().content.as_bytes())
                .await
                .unwrap();
            out.flush().await.unwrap();
        }

        out.write(b"\n").await.unwrap();
        out.flush().await.unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn chat_with_format_should_work() {
        let ollama = Ollama::new(mock_config());
        let format = r#"
{
    "type": "object",
    "properties": {
        "age": {
            "type": "integer"
        },
        "available": {
            "type": "boolean"
        }
    },
    "required": [
        "age",
        "available"
    ]
}"#;

        let resp = ollama
            .chat("llama3.1:8b")
            .user_message("Ollama is 22 years old and is busy saving the world. Respond using JSON")
            .format(format)
            .await
            .unwrap();
        println!("{resp:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn chat_with_tools_should_work() {
        let tool = r#"
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the weather for, e.g. San Francisco, CA"
                        },
                        "format": {
                            "type": "string",
                            "description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location", "format"]
                }
            }
        }"#;
        let ollama: Ollama = Ollama::new(mock_config());
        let resp = ollama
            .chat("llama3.1:8b")
            .user_message("What is the weather today in Shanghai?")
            .tool(tool)
            .await
            .unwrap();
        println!("resp: {resp:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn chat_with_images_should_work() {
        let ollama = Ollama::new(mock_config());
        let image = "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";
        let message = Message::new_user("What's in the image").image(&image);
        let response = ollama.chat("llava").message(&message).await.unwrap();
        println!("{response:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn pull_a_model_should_work() {
        let ollama = Ollama::new(mock_config());
        let stream = ollama.pull_model("llama3.2").stream().await.unwrap();
        print_stream(stream).await;
    }

    #[tokio::test]
    #[ignore]
    async fn push_a_model_should_work() {
        let ollama = Ollama::new(mock_config());
        let stream = ollama
            .push_model("mattw/pygmalion:latest")
            .stream()
            .await
            .unwrap();

        print_stream(stream).await;
    }

    #[tokio::test]
    #[ignore]
    async fn create_model_should_work() {
        let ollama = Ollama::new(mock_config());
        let resp = ollama
            .create_model("yuanshen")
            .from("llama3.1:8b")
            .system("You are a model who is good at playing Yuanshen")
            .await
            .unwrap();

        println!("resp: {resp:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn create_model_with_stream_should_work() {
        let ollama = Ollama::new(mock_config());
        let stream = ollama
            .create_model("yuanshsdf")
            .from("llama3.2:1b")
            .system("You are a model who is good at playing Yuanshen")
            .stream()
            .await
            .unwrap();
        print_stream(stream).await;
    }

    #[tokio::test]
    async fn show_model_information_should_work() {
        let ollama = Ollama::new(mock_config());
        let model_info = ollama.show_model_information("llama3.1:8b").await.unwrap();
        println!("{model_info:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn list_local_models_should_work() {
        let ollama = Ollama::new(mock_config());
        let local_models = ollama.list_local_models().await.unwrap();
        println!("local_models: {local_models:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn list_running_models_should_work() {
        let ollama = Ollama::new(mock_config());
        let resp = ollama.list_running_models().await.unwrap();
        println!("{resp:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn copy_a_model_should_work() {
        let ollama = Ollama::new(mock_config());
        let _ = ollama
            .copy_model("llama3.1:8b", "llama3.1:another")
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn copy_a_model_does_not_exist_should_raise_error() {
        let ollama = Ollama::new(mock_config());
        let err = ollama
            .copy_model("llama3.9:18b", "llama3.1:another")
            .await
            .err()
            .unwrap();

        assert_eq!(err.to_string(), "model does not exist");
    }

    #[tokio::test]
    #[ignore]
    async fn delete_a_model_should_work() {
        let ollama = Ollama::new(mock_config());
        let _ = ollama.delete_model("yuanshsdf:latest").await.unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn delete_a_model_does_not_exist_should_raise_error() {
        let ollama = Ollama::new(mock_config());
        let err = ollama
            .delete_model("a_model_does_not_exist")
            .await
            .err()
            .unwrap();
        assert_eq!(err.to_string(), "model does not exist")
    }

    #[tokio::test]
    #[ignore]
    async fn version_should_work() {
        let ollama = Ollama::new(mock_config());
        let version = ollama.version().await.unwrap();
        println!("version: {version:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn generate_embedding_should_work() {
        let ollama = Ollama::new(mock_config());
        let resp = ollama
            .generate_embedding("llama3.2:1b", "Here is an article about llamas...")
            .await
            .unwrap();
        println!("{resp:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn generate_embeddings_should_work() {
        let ollama = Ollama::new(mock_config());
        let resp = ollama
            .generate_embeddings("llama3.2:1b")
            .input("Why the sky is blue")
            .input("How are you")
            .inputs(&vec!["haha".to_string()])
            .await
            .unwrap();
        println!("{resp:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn check_blob_exists_should_work() {
        let ollama = Ollama::new(mock_config());
        let resp = ollama
            .check_blob_exists(
                "sha256:29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2",
            )
            .await;

        assert_eq!(resp.err().unwrap().to_string(), "blob does not exist");
    }

    #[tokio::test]
    async fn push_blob_should_work() {
        let ollama = Ollama::new(mock_config());
        let _ = ollama.push_blob("xx", "digest:lxlkc").await;
    }

    fn mock_config() -> &'static str {
        "http://localhost:11434"
    }

    async fn print_stream<T: Serialize>(mut resp: OllamaStream<T>) {
        let mut out = stdout();
        out.flush().await.unwrap();

        while let Some(item) = resp.next().await {
            match item {
                Ok(item) => {
                    let serialized = serde_json::to_string(&item).unwrap();
                    out.write(format!("{}\n", serialized).as_bytes())
                        .await
                        .unwrap();
                    out.flush().await.unwrap();
                }
                Err(e) => panic!("{}", e),
            }
        }

        out.write(b"\n").await.unwrap();
        out.flush().await.unwrap();
    }
}
//     }

//     fn mock_config() -> OllamaConfig {
//         OllamaConfig::from_url("http://localhost:11434")
//     }
// }

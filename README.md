# ollama-native 🐑
ollama-native is a minimalist Ollama Rust SDK that provides the most basic functionality for interacting with Ollama.

## Goals 🎯
- ✅ Provide access to the core [Ollama API][ollama-api-doc] functions for interacting with models.<br>
- ❌ The project does not include any business-specific functionality like _**chat with history**_.<br>

For users who need features like chat with history, these functionalities can be implemented at the business layer of your application, where you can manage conversation state and context across requests. Alternatively, you may choose to use other Ollama SDKs that provide these higher-level features.

## Features
- **Minimal Functionality**: Offers the core functionalities of Ollama without extra features or complexity.
- **Rusty APIs**: Utilizes chainable methods, making the API simple, concise, and idiomatic to Rust.
- 
## APIs
- [x] Generate a completion
- [x] Generate a chat completion
- [x] Create a Model
- [x] List Local Models
- [x] Show Model Information
- [x] Delete a Model
- [x] Pull a Model
- [x] Push a Model
- [x] Generate Embeddings
- [x] List Running Models
- [x] Version
- [x] Check if a Blob Exists
- [x] Push a Blob

## Usage Examples

### Generate a completion
```rust
use ollama_native::Ollama;

let ollama = Ollama::new("http://localhost:11434");

let response = ollama
    .generate("llama3.1:8b", "Tell me a joke about sharks")
    .seed(5)
    .temperature(3.2)
    .await?;

println!("{}", response.response);
```

### Generate request (Streaming)
Enable `stream` feature:
```
cargo add ollama-native --features stream
```
```rust
use ollama_native::{IntoStream, Ollama};
use tokio::io::AsyncWriteExt;
use tokio_stream::StreamExt;

let ollama = Ollama::new("http://localhost:11434")

let mut stream = ollama
    .generate("llama3.1:8b", "Tell me a joke about sharks")
    .stream()
    .await?;

let mut out = tokio::io::stdout();
while let Some(Ok(item)) = stream.next().await {
    out.write(item.response.as_bytes()).await?;
    out.flush().await?;
}
```

### Structured Ouput
See [structured outputs example][structured-outputs-example] for more details.
```rust
// JSON mode
let resposne = ollama
    .generate(
        "llama3.1:8b",
        "Ollama is 22 years old and is busy saving the world. Respond using JSON",
    )
    .format("json") // Use "json" to get the response in JSON format.
    .await?;

// Specified JSON format.
let output_format = r#"
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

let resposne = ollama
    .generate(
        "llama3.1:8b",
        "Ollama is 22 years old and is busy saving the world. Respond using JSON",
    )
    .format(&output_format)
    .await?;
```

## Examples
- [x] [Generate Completions][generate-completion]
- [x] [Generate Chat Completions (Streaming)][chat-request-stream]
- [x] [Generate Chat Completions with Images][chat-with-images]
- [x] [Generate Embeddings][generate-embeddings]
- [x] [Structured Outputs (JSON)][structured-outputs-example]
- [ ] Chat with History

## License
This project is licensed under the [MIT license][license].

[examples]: https://github.com/ZBcheng/examples
[generate-completion]: https://github.com/ZBcheng/examples]/generate-completions.rs
[chat-request-stream]: https://github.com/ZBcheng/examples]/chat-request-stream.rs
[chat-with-images]: https://github.com/ZBcheng/examples]/chat-with-images.rs
[generate-embeddings]: https://github.com/ZBcheng/examples]/generate-embeddings.rs
[structured-outputs-example]: https://github.com/ZBcheng/examples]/structured_outputs.rs
[ollama-api-doc]: https://github.com/ollama/ollama/blob/main/docs/api.md
[license]: https://github.com/ZBcheng/ollama-native/blob/main/LICENSE
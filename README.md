# ollama-native 🐑
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ZBcheng/ollama-native/rust.yml)][workflow]
[![GitHub Release](https://img.shields.io/github/v/release/ZBcheng/ollama-native)][release]
[![Crates.io Version](https://img.shields.io/crates/v/ollama-native?color=%23D400FF)][crates-io]
[![GitHub License](https://img.shields.io/github/license/ZBCheng/ollama-native)][license]

ollama-native is a minimalist Ollama Rust SDK that provides the most basic functionality for interacting with Ollama.

## Goals 🎯
- ✅ Provide access to the core [Ollama API][ollama-api-doc] functions for interacting with models.
- ❌ The project does not include any business-specific functionality like _**chat with history**_.

> [!TIP]
> For users who need features like chat with history, these functionalities can be implemented at the business layer of your application ([chat-with-history-example][chat-with-history]). Alternatively, you may choose to use other Ollama SDKs that provide these higher-level features.

## APIs 📝
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


## Features 🧬
- **Minimal Functionality**: Offers the core functionalities of Ollama without extra features or complexity.
- **Rusty APIs**: Utilizes chainable methods, making the API simple, concise, and idiomatic to Rust.

### API Design
<table>
    <thead><tr>
        <th ></th>
        <th style="text-align: center;">❌</th>
        <th style="text-align: center;">✅</th>
    </tr></thead>
<tbody>
<tr>
<th>Completion</th>
</td><td>

```rust
let options = OptionsBuilder::new()
    .stop("stop")
    .num_predict(42)
    .seed(42)
    .build();

let request = GenerateRequestBuilder::new()
    .model("llama3.1:8b")
    .prompt("Tell me a joke")
    .options(options)
    .build();

let response = ollama.generate(request).await?;
```

</td><td>

```rust
let response = ollama
    .generate("llama3.1:8b", "Tell me a joke")
    .stop("stop")
    .num_predict(42)
    .seed(42)
    .await?;
```

</td></tr>
<tr>
<th>Streaming Response</th>
</td><td>

```rust
let options = OptionsBuilder::new()
    .stop("stop")
    .num_predict(42)
    .seed(42)
    .build();

let request = GenerateStreamRequestBuilder::new()
    .model("llama3.1:8b")
    .prompt("Tell me a joke")
    .options(options)
    .build();

let stream = ollama.generate_stream(request).await?;
```

</td><td>

```rust
let stream = ollama
    .generate("llama3.1:8b", "Tell me a joke")
    .stop("stop")
    .num_predict(42)
    .seed(42)
    .stream() // Specify streaming response.
    .await?;
```

</td></tr>
</tbody></table>

## Usage 🔦
### Add dependencies
default features (generate, chat, version)
```sh
cargo add ollama-native
```

`stream` features
```sh
cargo add ollama-native --features stream
```
`model` features (create models, pull models...)
```sh
cargo add ollama-native --features model
```

### Generate a completion
```rust
use ollama_native::Ollama;

let ollama = Ollama::new("http://localhost:11434");

let response = ollama
    .generate("llama3.1:8b", "Tell me a joke about sharks")
    .seed(5)
    .temperature(3.2)
    .await?;
```

### Generate request (Streaming)
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
> [!TIP]
> See [structured outputs example][structured-outputs] for more details.
#### JSON Mode
```rust
// JSON mode
let resposne = ollama
    .generate(
        "llama3.1:8b",
        "Ollama is 22 years old and is busy saving the world.",
    )
    .format("json") // Use "json" to get the response in JSON format.
    .await?;
```

#### Specified JSON Format
```rust
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
        "Ollama is 22 years old and is busy saving the world.",
    )
    .format(&output_format)
    .await?;
```

## Examples 📖
- [x] [Generate Completions][generate-completion]
- [x] [Generate Chat Completions (Streaming)][chat-request-stream]
- [x] [Generate Chat Completions with Images][chat-with-images]
- [x] [Generate Embeddings][generate-embeddings]
- [x] [Structured Outputs (JSON)][structured-outputs]
- [x] [Chat with History][chat-with-history]

## License 📄
This project is licensed under the [MIT license][license].

[examples]: https://github.com/ZBcheng/ollama-native/tree/main/examples
[generate-completion]: https://github.com/ZBcheng/ollama-native/blob/main/examples/generate_completions.rs
[chat-request-stream]: https://github.com/ZBcheng/ollama-native/blob/main/examples/chat_request_stream.rs
[chat-with-images]: https://github.com/ZBcheng/ollama-native/blob/main/examples/chat_with_images.rs
[generate-embeddings]: https://github.com/ZBcheng/ollama-native/blob/main/examples/generate_embeddings.rs
[structured-outputs]: https://github.com/ZBcheng/ollama-native/blob/main/examples/structured_outputs.rs
[chat-with-history]: https://github.com/ZBcheng/ollama-native/blob/main/examples/chat_with_history.rs
[ollama-api-doc]: https://github.com/ollama/ollama/blob/main/docs/api.md
[workflow]: https://github.com/ZBcheng/ollama-native/blob/main/.github/workflows/rust.yml
[release]: https://github.com/ZBcheng/ollama-native/releases
[crates-io]: https://crates.io/crates/ollama-native
[license]: https://github.com/ZBcheng/ollama-native/blob/main/LICENSE

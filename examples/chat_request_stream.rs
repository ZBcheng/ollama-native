use ollama_native::{
    Message, Ollama,
    abi::completion::chat::ChatResponse,
    action::{IntoStream, OllamaStream},
};
use tokio::io::AsyncWriteExt;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define a helper function to print the responses.
    let print_stream = async |mut stream: OllamaStream<ChatResponse>| {
        let mut out = tokio::io::stdout();
        while let Some(Ok(item)) = stream.next().await {
            let content = item.message.unwrap().content;
            out.write(content.as_bytes()).await.unwrap();
            out.flush().await.unwrap();
        }

        out.write(b"\n").await.unwrap();
        out.flush().await.unwrap();
    };

    let ollama = Ollama::new("http://localhost:11434");
    let stream = ollama
        .chat("llama3.1:8b")
        .system_message("You are a robot that is good at telling jokes")
        .user_message("Who are you")
        .stream() // Specify that we want to stream the responses.
        .await?;

    print_stream(stream).await;

    // Or use the `messages` method to pass multiple messages at once.
    let messages = vec![
        Message::new_system("You are a robot that is good at telling jokes"),
        Message::new_user("Who are you"),
    ];

    let stream = ollama
        .chat("llama3.1:8b")
        .messages(messages)
        .stream() // Specify that we want to stream the responses.
        .await?;

    print_stream(stream).await;

    Ok(())
}

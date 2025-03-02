use std::sync::Arc;

use ollama_native::{Message, Ollama, action::IntoStream};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt},
    sync::RwLock,
};
use tokio_stream::StreamExt;

type Error = Box<dyn std::error::Error>;

struct Manager {
    ollama: Ollama,
    model: String,
    history: Arc<RwLock<Vec<Message>>>,
}

impl Manager {
    async fn chat(&self, input: &str) -> Result<(), Error> {
        let history = self.load_history().await;

        let mut stream = self
            .ollama
            .chat(&self.model)
            .messages(&history)
            .user_message(input)
            .stream()
            .await?;

        let mut out = tokio::io::stdout();
        let mut content = String::new();
        while let Some(Ok(item)) = stream.next().await {
            let content_chunk = item.message.unwrap().content;
            content.push_str(&content_chunk);
            out.write(content_chunk.as_bytes()).await?;
            out.flush().await?;
        }

        out.write(b"\n\n").await?;
        out.flush().await?;

        self.update_history(input, &content).await;

        Ok(())
    }

    async fn update_history(&self, input: &str, content: &str) {
        let mut history = self.history.write().await;
        history.push(Message::new_user(input));
        history.push(Message::new_assistant(content));
    }

    async fn load_history(&self) -> Vec<Message> {
        self.history.read().await.clone()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = ollama_native::Ollama::new("http://localhost:11434");

    let manager = Manager {
        ollama,
        model: "llama3.1:8b".to_string(),
        history: Arc::new(RwLock::new(vec![])),
    };

    let mut reader = tokio::io::BufReader::new(tokio::io::stdin());
    let mut input = String::new();

    loop {
        input.clear();

        println!("Input:");
        let _ = reader.read_line(&mut input).await?;
        match input.trim() {
            "/bye" => break,
            content => manager.chat(&content).await?,
        }
    }

    Ok(())
}

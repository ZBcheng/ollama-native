use ollama_native::Ollama;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::new("http://localhost:11434");
    let response = ollama
        .generate("llama3.1:8b", "Tell me a joke about sharks")
        .await?;
    println!("{}", response.response);

    Ok(())
}

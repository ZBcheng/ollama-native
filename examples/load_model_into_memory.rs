use ollama_native::Ollama;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::new("http://localhost:11434");

    // Load llama3.1:8b to memory.
    let response = ollama.generate("llama3.1:8b").load().await?;
    println!("{response:?}");

    // Unload llama3.1:8b from memory.
    let response = ollama.generate("llama3.1:8b").unload().await?;
    println!("{response:?}");

    Ok(())
}

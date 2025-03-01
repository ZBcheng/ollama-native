use ollama_native::Ollama;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::new("http://localhost:11434");
    let response = ollama
        .generate_embeddings("llama3.1:8b")
        .input("Why the sky is blue?")
        .input("Why the sea is salty?")
        .await?;

    println!("{:?}", response);

    // Or use the `inputs` method to pass multiple inputs at once.
    let inputs = vec!["Why the sky is blue?", "Why the sea is salty?"];
    let response = ollama
        .generate_embeddings("llama3.1:8b")
        .inputs(&inputs)
        .seed(5) // Set seed to 5.
        .min_p(3.2) // Set min_p to 3.2.
        .await?;

    println!("{:?}", response);
    Ok(())
}

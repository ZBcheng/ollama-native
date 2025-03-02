use ollama_native::{Ollama, OllamaError};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::new("http://localhost:11434");

    // JSON mode
    let resposne = ollama
        .generate(
            "llama3.1:8b",
            "Ollama is 22 years old and is busy saving the world. Respond using JSON",
        )
        .format("json") // Use "json" to get the response in JSON format.
        .await?;
    println!("JSON mode:\n{}\n", resposne.response);

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

    println!("structured JSON output:\n{}\n", resposne.response);

    // If the output format is an invalid JSON format, an error will be returned.
    let invalid_output_format = r"invalid format";
    match ollama
        .generate("llama3.1:8b", "Tell me a joke about sharks")
        .format(invalid_output_format)
        .await
    {
        Ok(_) => println!("This should not happen"),
        Err(OllamaError::InvalidFormat(e)) => println!("Got error: {}", e), // OllamaError::InvalidFormat("invalid format schema: expected value at line 1 column 1".to_string())
        Err(e) => println!("Unexpected error: {}", e),
    }

    Ok(())
}

use ollama_native::{Ollama, OllamaError};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ollama = Ollama::new("http://localhost:11434");

    // JSON mode
    let resposne = ollama
        .generate("llama3.1:8b")
        .prompt("Ollama is 22 years old and is busy saving the world. Respond using JSON")
        .json() // Get the response in JSON format.
        .await?;
    println!("JSON mode:\n{}\n", resposne.response);

    // Specified JSON format.
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
    let resposne = ollama
        .generate("llama3.1:8b")
        .prompt("Ollama is 22 years old and is busy saving the world. Respond using JSON")
        .format(format)
        .await?;

    println!("structured JSON output:\n{}\n", resposne.response);

    // If the output format is an invalid JSON format, an error will be returned.
    let invalid_output_format = r"invalid JSON format";
    match ollama
        .generate("llama3.1:8b")
        .prompt("Tell me a joke about sharks")
        .format(invalid_output_format)
        .await
    {
        Ok(_) => println!("This should not happen"),
        Err(OllamaError::InvalidFormat(e)) => println!("Got error: {}", e), // OllamaError::InvalidFormat("invalid JSON schema: expected value at line 1 column 1".to_string())
        Err(e) => println!("Unexpected error: {}", e),
    }

    Ok(())
}

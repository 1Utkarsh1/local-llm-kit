# Local LLM Kit

An OpenAI-like interface for local Large Language Models. This package allows you to interact with local LLMs using an API similar to OpenAI's, including support for chat completions, function calling, streaming responses, and more.

**Created by Utkarsh Rajput (GitHub: [1Utkarsh1](https://github.com/1Utkarsh1))** - A powerful tool for working with local language models with all the convenience of the OpenAI API.

## Features

- 🗣️ **Chat and Completion API**: Similar to OpenAI's API
- 🧩 **Multiple Backend Support**: Works with Hugging Face Transformers and llama.cpp
- 🛠️ **Function Calling**: Register Python functions that LLMs can call
- 📱 **Streaming Support**: Stream responses token by token
- 🧠 **Memory Management**: Auto-truncation and context management
- 📊 **Logprobs**: Get token probabilities for generations
- 📝 **Prompt Formatting**: Supports various model template formats (Llama, Mistral, etc.)
- 🌐 **JSON Mode**: Enforce structured JSON output

## Installation

### Basic Installation

```bash
pip install local-llm-kit
```

### With Backend Support

```bash
# For Hugging Face Transformers support
pip install "local-llm-kit[transformers]"

# For llama.cpp support
pip install "local-llm-kit[llamacpp]"

# For all backends
pip install "local-llm-kit[all]"
```

## Quick Start

### Chat with a Model

```python
from local_llm_kit import chat

response = chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about the solar system."}
    ],
    model_path="path/to/your/model",  # Local model path or HF model name
    temperature=0.7,
    max_tokens=512
)

print(response["choices"][0]["message"]["content"])
```

### Text Completion

```python
from local_llm_kit import complete

response = complete(
    prompt="The solar system consists of",
    model_path="path/to/your/model",
    temperature=0.7,
    max_tokens=512
)

print(response["choices"][0]["text"])
```

### Function Calling

```python
from local_llm_kit import LLM

# Define a function with its schema
def get_weather(location: str, unit: str = "celsius"):
    """Get the weather for a location."""
    # In a real app, this would call a weather API
    return {"temperature": 22, "unit": unit, "description": f"Sunny in {location}"}

weather_function = {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature"
            }
        },
        "required": ["location"]
    }
}

# Create LLM instance
llm = LLM(model_path="path/to/your/model")

# Register the function
llm.add_function(
    name="get_weather",
    schema=weather_function,
    implementation=get_weather
)

# Chat with function calling
response = llm.chat(
    messages=[
        {"role": "user", "content": "What's the weather like in Paris?"}
    ],
    # The function will be automatically called if the model chooses to use it
)

print(response["choices"][0]["message"]["content"])
```

### Streaming Responses

```python
from local_llm_kit import chat

for chunk in chat(
    messages=[
        {"role": "user", "content": "Write a short poem about nature."}
    ],
    model_path="path/to/your/model",
    stream=True
):
    # Process each token as it's generated
    if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)
```

## Command Line Interface

Local LLM Kit includes a CLI for easy interaction with models.

### Interactive Chat

```bash
local-llm-kit chat --model path/to/your/model --system "You are a helpful assistant."
```

### Text Completion

```bash
local-llm-kit complete --model path/to/your/model --prompt "Once upon a time,"
```

## Advanced Usage

### Using Different Backends

```python
from local_llm_kit import LLM

# Using Transformers backend
llm_transformers = LLM(
    model_path="mistralai/Mistral-7B-Instruct-v0.1",
    backend="transformers",
    backend_kwargs={"device": "cuda", "torch_dtype": "float16"}
)

# Using llama.cpp backend
llm_llamacpp = LLM(
    model_path="path/to/model.gguf",
    backend="llamacpp",
    backend_kwargs={"n_gpu_layers": -1, "n_ctx": 4096}
)
```

### JSON Mode

```python
from local_llm_kit import chat

response = chat(
    messages=[
        {"role": "user", "content": "Generate a JSON list of 3 planets with their diameter."}
    ],
    model_path="path/to/your/model",
    format="json"
)

import json
planets = json.loads(response["choices"][0]["message"]["content"])
print(planets)
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
# Mistral-common Experimental API

## Context

The `experimental` module provides access to a FastAPI server designed to handle tokenization and detokenization operations through a REST API. This server features:

1. **Tokenization**: Converting chat completion requests into token sequences
2. **Detokenization**: Converting token sequences back into human-readable formats to an [AssistantMessage][mistral_common.protocol.instruct.messages.AssistantMessage] object or a raw string.
3. **Generation**: Generating text from a [ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest] using a server backend.

This API serves as a bridge between different providers and tokenization needs regardless of the provider programming language.

## Features

1. **Tokenizer Version Support**: Handles all our tokenizer versions automatically
2. **Tool Call Parsing**: Automatically extracts tool calls from token sequences
3. **Think Chunk Support**: Handles think chunks for v13+ tokenizers
5. **OpenAI Compatibility**: Accepts OpenAI-formatted requests for easier integration

## Error Handling

The API provides detailed error messages for:

- Invalid token sequences
- Empty token lists
- Missing tokenizer configuration
- Validation errors in requests
- Tool call parsing failures

## Usage

### Installation

To use the experimental API server, you need to install mistral-common with the appropriate dependencies:

```bash
pip install mistral-common[server]
```

### Launching the Server

You can launch the server using the follwing CLI command:

```bash
mistral_common mistralai/Magistral-Small-2507 [validation_mode] \
--host 127.0.0.1 --port 8000 \
--generation-host 127.0.0.1 --generation-port 8080 --generation-backend llama_cpp \
--api-key "" --timeout 60
```

#### Command Line Options

- `tokenizer_path`: Path to the tokenizer. Can be a HuggingFace model ID or a local path.
- `validation_mode`: Validation mode to use, choices in: "test", "finetuning", "serving" (Optional, defaults to `"test"`)
- `--host`: API host (default: `127.0.0.1`)
- `--port`: API port (default: `0` - auto-selects available port)
- `--generation-host`: Generation server host (default: `127.0.0.1`)
- `--generation-port`: Generation server port (default: `8080`)
- `--generation-backend`: Generation backend to use (default: `llama_cpp`)
- `--api-key`: API key for the generation server (default: `""`)
- `--timeout`: Timeout for the generation server (default: `60`)

## Available Routes

### Documentation

- **Route**: `/` or `/docs`
- **Method**: GET
- **Description**: The Swagger documentation

The Swagger UI provides interactive documentation for all available endpoints.

### Tokenization

#### Tokenize Request

- **Route**: `/tokenize/request`
- **Method**: POST
- **Description**: Tokenizes a chat completion request
- **Request Body**: 
    - Chat completion request in either:
        - Mistral-common format ([ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest])
        - OpenAI-compatible format ([OpenAIChatCompletionRequest][mistral_common.protocol.openai.request.OpenAIChatCompletionRequest]). Not all values used by OpenAI are supported.
- **Response**: List of token IDs

Example requests:

```python
import requests

from fastapi.encoders import jsonable_encoder

from mistral_common.protocol.instruct.messages import SystemMessage, TextChunk, ThinkChunk, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# Simple request
response = requests.post("http://localhost:8000/tokenize/request", json={
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ]
})
print(response.json())
# [1, 3, 22177, 1044, 2606, 1584, 1636, 1063, 4]

# Complex request with tools and think chunks
system_message = SystemMessage(
    content=[
        TextChunk(
            text=(
                "First draft your thinking process (inner monologue) until you arrive at a response. Format your "
                "response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and "
                "the response in the same language as the input.\n\nYour thinking process must follow the template "
                "below:"
            )
        ),
        ThinkChunk(
            thinking=(
                "Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and "
                "as long as you want until you are confident to generate the response. Use the same language as the "
                "input."
            ),
            closed=True,
        ),
        TextChunk(text="Here, provide a self-contained response."),
    ],
)


response = requests.post(
    "http://localhost:8000/tokenize/request",
    json=jsonable_encoder(
        ChatCompletionRequest(messages=[
            system_message,
            UserMessage(content="How many 'r' are there in strawberry ?")
        ])
    ),
)
print(response.json())
# [1, 17, 10107, 19564, 2143, ..., 33681, 3082, 4]
```

### Detokenization

#### Detokenize to Assistant Message

- **Route**: `/detokenize/`
- **Method**: POST
- **Description**: Converts tokens to an assistant message with tool call parsing
- **Request Body**: 
    - List of token IDs
- **Response**: [AssistantMessage][mistral_common.protocol.instruct.messages.AssistantMessage] object with parsed content and tool calls

Example requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/detokenize/",
    json=[1, 3, 22177, 1044, 2606, 1584, 1636, 1063, 4]
)
print(response.json())
# {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Hello, how are you?'}], 'tool_calls': None, 'prefix': True}

# With think chunks
response = requests.post(
    "http://localhost:8000/detokenize/",
    json=[12598, 1639, 3648, 2314, 1494, 1046, 34, 6958, 1584, 1032, 1051, 1576, 1114, 1039, 1294, 51567, 33681, 3082, 35, 19587, 1051, 19587, 2]
)
print(response.json())
# {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Let me think about it.'}, {'type': 'thinking', 'thinking': "There are 3 'r' in strawberry ?", 'closed': True}, {'type': 'text', 'text': '$$3$$'}], 'tool_calls': None, 'prefix': False}

# With tool calls
response = requests.post(
    "http://localhost:8000/detokenize/",
    json=[9, 12296, 1095, 99571, 32, 38985, 1039, 3494, 4550, 1576, 1314, 3416, 33681, 2096, 1576, 27965, 4550, 1576, 1114, 1039, 27024, 2]
)
print(response.json())
# {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'null', 'type': 'function', 'function': {'name': 'count_letters', 'arguments': "{'word': 'strawberry', 'letter': 'r'}"}}], 'prefix': False}
```


#### Detokenize to String

- **Route**: `/detokenize/string`
- **Method**: POST
- **Description**: Converts tokens to a raw string
- **Request Body**:
    - `tokens`: List of token IDs
    - `special_token_policy`: Policy for handling special tokens. See [SpecialTokenPolicy][mistral_common.tokens.tokenizers.base.SpecialTokenPolicy]
- **Response**: Detokenized string

Example request:
```python
import requests

response = requests.post("http://localhost:8000/detokenize/string", json={
    "tokens":[1, 3, 22177, 1044, 2606, 1584, 1636, 1063, 4],
    "special_token_policy": 1,
})
print(response.json())
# <s>[INST]Hello, how are you?[/INST]
```

### Generation

- **Route**: `/v1/chat/completions`
- **Method**: POST
- **Description**: Generates text from a chat completion request. This endpoint forwards the request to the generation server.
- **Request Body**:
    - Chat completion request in either:
        - Mistral-common format ([ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest])
        - OpenAI-compatible format ([OpenAIChatCompletionRequest][mistral_common.protocol.openai.request.OpenAIChatCompletionRequest]). Not all values used by OpenAI are supported.
- **Response**: Chat completion response in OpenAI-compatible format

## Issues and feedback

If you encounter any issues or have feedback, please open an issue or discussion on the [GitHub repository](https://github.com/mistralai/mistral-common).
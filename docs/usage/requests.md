# Requests

To query an AI assistant like [Mistral's LeChat](https://chat.mistral.ai/chat) or ChatGPT you need to provide the following:

- The history of the conversation between the user, the assistant and the tool calls.
- The tools available to the assistant.
- The context of the request.

In `mistral-common`, we currently support the following requests types:

- Instruct requests:
    - [Chat completion requests](#chat-completion).
    - [Fill-In-the-Middle completion](#fim).
- [Transcription requests](#transcription-request)

Every instruct requests should be encoded with its corresponding `encode_function` function by the tokenizers.


## Chat completion

Chat completion consists in a conversation between a user and an assistant. The assistant can call tools to enrich its response. Some of our tokenizers also support the use of images (see the [images](./images.md))

Every chat completion request are defined via [ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest].

To perform actual task every requests should follow the following structure:

1. validate the request via a [MistralRequestValidator][mistral_common.protocol.instruct.validator.MistralRequestValidator]
2. normalize the requests via the [InstructRequestNormalizer][mistral_common.protocol.instruct.normalize.InstructRequestNormalizer].
3. encode the request.

Using the [MistralTokenizer.encode_chat_completion][mistral_common.tokens.tokenizers.mistral.MistralTokenizer.encode_chat_completion] method will perform all these steps for you.

Following this design ensures minimizing unexpected behavior from the user.

### Conversation

A conversation with a model is a sequence of messages. Each message can be a user message, an assistant message, a tool message or a system message. To ease the creation of these messages, we provide a set of Pydantic classes that you can use to create them:

- [UserMessage][mistral_common.protocol.instruct.messages.UserMessage]: a message from the user. Users are the ones that interact with the model.
- [AssistantMessage][mistral_common.protocol.instruct.messages.AssistantMessage]: a message from the assistant. The assistant is the model itself.
- [ToolMessage][mistral_common.protocol.instruct.messages.ToolMessage]: a message from a tool. Tools are functions that the model can call to get information to answer the user's question.
- [SystemMessage][mistral_common.protocol.instruct.messages.SystemMessage]: a message from the system. Also called `System Prompt`, it is a set of instructions that the model should follow to answer the user's question. This allows you to customize the behavior of the model.

### Tools

Tools are functions that the model can call to get information to answer the user's question. See the [Tools](./tools.md) section for more information.

### Example

Here is an example of a request where a user asks for the weather in Paris. The model is also given access to a `get_current_weather` tool to get the weather:

```python
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

request = ChatCompletionRequest(
    tools=[
        Tool(
            function=Function(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the user's location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            )
        )
    ],
    messages=[
        UserMessage(content="What's the weather like today in Paris"),
    ],
)

tokenizer = MistralTokenizer.v3()
tokenizer.encode_chat_completion(request)
```

## FIM

Fill In the Middle (FIM) is a task where the model is given a prefix and a suffix and is asked to fill in the middle. This is useful for code completion, where the model is given a prefix of code and is asked to complete the code.

A pydantic class [FIMRequest][mistral_common.protocol.fim.request.FIMRequest] is defined to ease the creation of these requests.

```python
from mistral_common.protocol.fim.request import FIMRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

request = FIMRequest(
    prompt="def hello_world():\n    print('Hello, world!')",
    suffix="\n\nhello_world()",
)

tokenizer = MistralTokenizer.v3()
tokenizer.encode_fim(request)
```

## Transcription

Transcription is a task where the model is given an audio file and returns the text corresponding to the speech contained in the audio.

A pydantic class [TranscriptionRequest][mistral_common.protocol.transcription.request.TranscriptionRequest] is defined to ease the creation of these requests.

```python
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from huggingface_hub import hf_hub_download

repo_id = "mistralai/voxtral-mini"
tokenizer = MistralTokenizer.from_hf_hub(repo_id)

obama_file = hf_hub_download("patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset")
audio = Audio.from_file(obama_file, strict=False)

audio = RawAudio.from_audio(audio)
request = TranscriptionRequest(model=repo_id, audio=audio, language="en")

tokenizer.encode_transcription(request)
```

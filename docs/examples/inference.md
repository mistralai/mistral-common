# Inference

We have a few examples of how to use the library with our models:

- [Chat Completion](#chat-completion)
  - [Text only](#text-only)
  - [Image](#image)
  - [Function calling](#function-calling)
  - [Audio](#audio)
- [Fill-in-the-middle (FIM) Completion](#fim)
- [Audio Transcription](#audio-transcription)

## Chat Completion

### Text-only

```python
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    TextChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

repo_id = "mistralai/Mistral-Large-Instruct-2411"
tokenizer = MistralTokenizer.from_hf_hub(repo_id)


messages = [
    UserMessage(content="What is the capital of France?"),
    AssistantMessage(content="The capital of France is Paris."),
    UserMessage(content="And the capital of Spain?"),
]

request = ChatCompletionRequest(messages=messages)
tokenized = tokenizer.encode_chat_completion(request)

# pass tokenized.tokens to your favorite model
print(tokenized.tokens)

# print text to visually see tokens
print(tokenized.text)
```


### Image

```python
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ImageURLChunk,
    SystemMessage,
    TextChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

repo_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
tokenizer = MistralTokenizer.from_hf_hub(repo_id)

system_prompt = """You are Mistral Small 3.2, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is 03-06-2025. You have the ability to process images."""

messages = [
    SystemMessage(content=system_prompt),
    UserMessage(content="What is the capital of France?"),
    AssistantMessage(content="The capital of France is Paris."),
    UserMessage(content=[TextChunk(text="Does this photo comes from there ?")]),
    UserMessage(
        content=[
            ImageURLChunk(
                image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques%2C_Paris_ao%C3%BBt_2014_%282%29.jpg/1280px-La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques%2C_Paris_ao%C3%BBt_2014_%282%29.jpg"
            )
        ]
    ),
]
request = ChatCompletionRequest(messages=messages)
tokenized = tokenizer.encode_chat_completion(request)

# pass tokenized.tokens to your favorite image model
print(tokenized.tokens)
print(tokenized.images)

# print text to visually see tokens
print(tokenized.text)
```

### Function calling

```python
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    TextChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.tool_calls import Function, Tool

repo_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
tokenizer = MistralTokenizer.from_hf_hub(repo_id)


messages = [UserMessage(content="What is the weather in France like?")]

tool = Tool(
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

request = ChatCompletionRequest(messages=messages, tools=[tool])
tokenized = tokenizer.encode_chat_completion(request)

# pass tokenized.tokens to your favorite agent model
print(tokenized.tokens)

# print text to visually see tokens
print(tokenized.text)
```

### Audio

```py
from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage, AssistantMessage, RawAudio
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.audio import Audio
from huggingface_hub import hf_hub_download

repo_id = "mistralai/Voxtral-Mini-3B-2507"
tokenizer = MistralTokenizer.from_hf_hub(repo_id)

obama_file = hf_hub_download("patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset")
bcn_file = hf_hub_download("patrickvonplaten/audio_samples", "bcn_weather.mp3", repo_type="dataset")

def file_to_chunk(file: str) -> AudioChunk:
    audio = Audio.from_file(file, strict=False)
    return AudioChunk.from_audio(audio)

text_chunk = TextChunk(text="Which speaker do you prefer between the two? Why? How are they different from each other?")
user_msg = UserMessage(content=[file_to_chunk(obama_file), file_to_chunk(bcn_file), text_chunk]).to_openai()


request = ChatCompletionRequest(messages=[user_msg])
tokenized = tokenizer.encode_chat_completion(request)

# pass tokenized.tokens to your favorite audio model
print(tokenized.tokens)
print(tokenized.audios)

# print text to visually see tokens
print(tokenized.text)
```

## FIM

```python
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.fim.request import FIMRequest

tokenizer = MistralTokenizer.from_hf_hub("mistralai/Codestral-22B-v0.1")

prefix = """def add("""
suffix = """    return sum"""

request = FIMRequest(prompt=prefix, suffix=suffix)

tokenized = tokenizer.encode_fim(request)

# pass tokenized.tokens to your favorite model
print(tokenized.tokens)

# print text to visually see tokens
print(tokenized.text)
```


## Audio Transcription

```python
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from huggingface_hub import hf_hub_download

repo_id = "mistralai/Voxtral-Mini-3B-2507"
tokenizer = MistralTokenizer.from_hf_hub(repo_id)

obama_file = hf_hub_download("patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset")
audio = Audio.from_file(obama_file, strict=False)

audio = RawAudio.from_audio(audio)
request = TranscriptionRequest(model=repo_id, audio=audio, language="en")

tokenized = tokenizer.encode_transcription(request)

# pass tokenized.tokens to your favorite audio model
print(tokenized.tokens)
print(tokenized.audios)

# print text to visually see tokens
print(tokenized.text)
```




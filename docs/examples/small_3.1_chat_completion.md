# Chat completion using Mistral Small 3.1

We will use [vLLM](https://github.com/vllm-project/vllm) to run [Mistral Small 3.1](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503).

## Setup

First, install `vLLM`:
```sh
pip install vllm
```

You don't need to install `mistral-common` as vLLM supports directly Mistral models and tokenizers !

In case you had already installed `mistral-common`, you can upgrade it to the latest version:
```sh
pip install --upgrade mistral-common
```

## Running the model

We recommend that you use Mistral-Small-3.1-24B-Instruct-2503 in a server/client setting.

### Launch the server

To launch the server, run the following command:

```sh
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --limit_mm_per_prompt 'image=10' --tensor-parallel-size 2
```

As you can see the following arguments need the `mistral` specific value:

- `--tokenizer_mode`
- `--config_format`
- `--load_format`
- `--tool-call-parser`

Thanks to that, internally vLLM will know to use the tokenizers and tool-calls that are defined in `mistral-common`, you can enjoy our full model features natively.

### Chat Completion

Let's use a [ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest] so that we can normalize and validate it before sending it to vLLM. Because `Mistral Small 3.1 24B Instruct` use the v7 tokenizer, we will use the [InstructRequestNormalizerV7][mistral_common.protocol.instruct.normalize.InstructRequestNormalizerV7] to normalize the request.

We will also use the *system prompt* defined in [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/SYSTEM_PROMPT.txt).


```python
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ImageURLChunk,
    SystemMessage,
    TextChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizerV7
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.protocol.instruct.validator import MistralRequestValidatorV5

validator = MistralRequestValidatorV5()
request_normalizer = InstructRequestNormalizerV7.normalizer()

system_prompt = """You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is 03-06-2025.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is 02-06-2025) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.
You cannot read nor transcribe audio files or videos."""


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

request = ChatCompletionRequest(
    messages=messages,
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
)
validator.validate_request(request) # No error means the request is valid
instruct_request = request_normalizer.from_chat_completion_request(request) # Normalize the request and convert it to an InstructRequest
print(instruct_request.messages) 
```

The messages are now normalized and the last two users messages are merged into one and you should see this:

```python
[SystemMessage(role='system', content='You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\nYou power an AI assistant called Le Chat.\nYour knowledge base was last updated on 2023-10-01.\nThe current date is 03-06-2025.\n\nWhen you\'re not sure about some information, you say that you don\'t have the information and don\'t make up anything.\nIf the user\'s question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").\nYou are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is 02-06-2025) and when asked about information at specific dates, you discard information that is at another date.\nYou follow these instructions in all languages, and always respond to the user in the language they use or request.\nNext sections describe the capabilities that you have.\n\n# WEB BROWSING INSTRUCTIONS\n\nYou cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.\n\n# MULTI-MODAL INSTRUCTIONS\n\nYou have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.\nYou cannot read nor transcribe audio files or videos.'),
  UserMessage(role='user', content='What is the capital of France?'),
  AssistantMessage(role='assistant', content='The capital of France is Paris.', tool_calls=None, prefix=False),
  UserMessage(role='user', content=[TextChunk(type='text', text='Does this photo comes from there ?'), ImageURLChunk(type='image_url', image_url='https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques%2C_Paris_ao%C3%BBt_2014_%282%29.jpg/1280px-La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques%2C_Paris_ao%C3%BBt_2014_%282%29.jpg')])]
```

Now let's convert the request to something vLLM can understand:

```python
vllm_request = instruct_request.to_openai(
    temperature=0.15,
    model="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)
```

Finally, let's send the request to vLLM:

```python
import json

import requests

url = "http://<your-url>:8000/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}

response = requests.post(url, headers=headers, data=json.dumps(vllm_request))
assistant_response_content = response.json()["choices"][0]["message"]["content"]
print(assistant_response_content)
```

Your response should look like this:
```
Yes, this photo is of Paris, France. The Eiffel Tower is prominently visible in the center of the image, which is an iconic landmark of Paris. The river in the foreground is the Seine, and the cityscape includes many of the characteristic buildings and architecture found in Paris.
```

Now let's try to use the `get_weather` tool:

```python
from mistral_common.protocol.instruct.tool_calls import ToolCall

def get_current_weather(location, format):
    if "Paris" in location:
        return {"temperature": "20", "unit": format, "description": "sunny"}
    else:
        return {"temperature": "10", "unit": format, "description": "rainy"}

new_request = instruct_request.model_copy(deep=True)

new_request.messages.append(AssistantMessage(content=assistant_response_content))
new_request.messages.append(UserMessage(content="Could you tell me what is the weather there ?"))

new_vllm_request = new_request.to_openai(
    temperature=0.15,
    model="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)

response = requests.post(url, headers=headers, data=json.dumps(new_vllm_request))
tool_call_content = response.json()["choices"][0]["message"]["tool_calls"]

tool_call = ToolCall(
    **tool_call_content[0]
)

assert tool_call.function.name == "get_current_weather"

print(
    get_current_weather(
        **json.loads(tool_call.function.arguments)
    )
)
```

You should see that the model correctly called the tool to give you the weather in Paris:

```python
{'temperature': '20', 'unit': 'celsius', 'description': 'sunny'}
```








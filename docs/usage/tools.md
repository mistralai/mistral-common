# Tools

To execute external tools, like searching the web, LLMs need to be given the ability to call such tools whenever they see fit. This is usually achieved by adding all available tools that the LLM can call into the context of the [ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest] and representing tool calls as well as tool message as its own objects.

## Define tools

To define tools in `mistral-common`, you use the [Tool][mistral_common.protocol.instruct.tool_calls.Tool] class. Let's define a simple tool that fetches the current weather for a given city.

The example function below is not a real call that calls the weather, but it very well be an call to a weather API. We hardcode random numbers here just for demonstrative purposes.

```python
from mistral_common.protocol.instruct.tool_calls import Tool, Function


def get_current_weather(location: str, format: str) -> int:
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return 22 if format == "celsius" else 72
    elif "san francisco" in location.lower():
        return 14 if format == "celsius" else 57
    elif "paris" in location.lower():
        return 18 if format == "celsius" else 64
    else:
        return 20 if format == "celsius" else 68

weather_tool = Tool(
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
```

## Tool calling and messages

When a model decides to call a tool, it will add a [ToolCall][mistral_common.protocol.instruct.tool_calls.ToolCall] to the [AssistantMessage][mistral_common.protocol.instruct.messages.AssistantMessage]. The tool call will have an `id` and a [FunctionCall][mistral_common.protocol.instruct.tool_calls.FunctionCall] with the name of the function and the arguments to pass to it.

```python
from mistral_common.protocol.instruct.messages import AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    FunctionCall,
    Tool,
    ToolCall,
)


AssistantMessage(
    content=None,
    tool_calls=[
        ToolCall(
            id="VvvODy9mT",
            function=FunctionCall(
                name="get_current_weather",
                arguments='{"location": "Paris, France", "format": "celsius"}',
            ),
        )
    ],
)
```

Then you can execute the function and return the result in a [ToolMessage][mistral_common.protocol.instruct.messages.ToolMessage]. The [ToolMessage][mistral_common.protocol.instruct.messages.ToolMessage] must have the same `id` as the [ToolCall][mistral_common.protocol.instruct.tool_calls.ToolCall] it's responding to.

```python
from mistral_common.protocol.instruct.messages import ToolMessage


ToolMessage(tool_call_id="VvvODy9mT", name="get_current_weather", content="22")
```

## Putting it all together

Now we can put it all together and create a [ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest] with the tools and the messages. We can then use the [MistralTokenizer][mistral_common.tokens.tokenizers.mistral.MistralTokenizer] to encode the request.

```python
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    FunctionCall,
    Tool,
    ToolCall,
)

request = ChatCompletionRequest(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What's the weather like in Paris?"),
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="VvvODy9mT",
                    function=FunctionCall(
                        name="get_current_weather",
                        arguments='{"location": "Paris, France", "format": "celsius"}',
                    ),
                )
            ],
        ),
        ToolMessage(tool_call_id="VvvODy9mT", name="get_current_weather", content="22"),
    ],
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

tokenizer = MistralTokenizer.v3()
tokens = tokenizer.encode_chat_completion(request).tokens
```

Now the model can use the tool result to answer the user's question.

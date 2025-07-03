from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pytest
from openai.resources.chat.completions.completions import Completions
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam as OpenAIAssistantMessage,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam as OpenAIImageChunk,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam as OpenAITextChunk,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam as OpenAIToolCall,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam as OpenAISystemMessage,
)
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam as OpenAIToolMessage
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam as OpenAITool
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam as OpenAIUserMessage
from PIL import Image

from mistral_common.protocol.instruct.converters import _OPENAI_COMPLETION_FIELDS, _check_openai_fields_names
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    ImageChunk,
    ImageURL,
    ImageURLChunk,
    SystemMessage,
    TextChunk,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.instruct.request import InstructRequest

CURRENT_FILE_PATH = Path(__file__).resolve()
ROOT_PATH = CURRENT_FILE_PATH.parents[1]
LOGO_PATH = ROOT_PATH / "docs" / "assets" / "logo_favicon.png"


def test_openai_chat_fields() -> None:
    completions_create_inspect_keys = set(signature(Completions.create).parameters.keys())
    completions_create_inspect_keys.remove("self")
    assert _OPENAI_COMPLETION_FIELDS == completions_create_inspect_keys


def test_check_openai_fields_names() -> None:
    # Valid names
    valid_names = {"temperature"}
    _check_openai_fields_names(valid_names, {"temperature"})

    # Invalid openai names
    valid_names = {"temperature"}
    with pytest.raises(ValueError):
        _check_openai_fields_names(valid_names, {"temperature", "max_tokens"})

    # Invalid name
    with pytest.raises(ValueError):
        _check_openai_fields_names(valid_names, {"invalid_name"})


def test_convert_image_chunk() -> None:
    image = Image.open(LOGO_PATH.as_posix())
    chunk = ImageChunk(image=image)

    openai_image = chunk.to_openai()
    assert openai_image["type"] == "image_url"
    assert isinstance(openai_image["image_url"], dict)
    assert openai_image["image_url"]["url"].startswith("data:image/png;base64,")

    assert isinstance(ImageChunk.from_openai(openai_image), ImageChunk)

    typeddict_openai = OpenAIImageChunk(**openai_image)  # type: ignore[typeddict-item]

    assert isinstance(ImageChunk.from_openai(typeddict_openai), ImageChunk)


def test_convert_text_chunk() -> None:
    chunk = TextChunk(text="Hello")
    text_openai = chunk.to_openai()

    text_openai == {"type": "text", "text": "Hello"}

    assert TextChunk.from_openai(text_openai) == chunk

    typeddict_openai = OpenAITextChunk(**chunk.to_openai())  # type: ignore[typeddict-item]
    assert TextChunk.from_openai(typeddict_openai) == chunk


@pytest.mark.parametrize(
    ["openai_image_url_chunk", "image_url_chunk"],
    [
        (
            OpenAIImageChunk(
                type="image_url",
                image_url={
                    "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                    "detail": "auto",
                },
            ),
            ImageURLChunk(
                image_url=ImageURL(
                    url="https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                    detail="auto",
                )
            ),
        ),
        (
            OpenAIImageChunk(
                type="image_url",
                image_url={
                    "url": "data:image/png;base64,iVBORw0",
                },
            ),
            ImageURLChunk(
                image_url="data:image/png;base64,iVBORw0",
            ),
        ),
    ],
)
def test_convert_image_url_chunk(openai_image_url_chunk: Dict, image_url_chunk: ImageURLChunk) -> None:
    assert image_url_chunk.to_openai() == openai_image_url_chunk
    if not isinstance(image_url_chunk.image_url, ImageURL):
        image_url_from_openai = ImageURLChunk.from_openai(openai_image_url_chunk)
        assert isinstance(image_url_from_openai.image_url, ImageURL)
        assert image_url_from_openai.image_url.url == image_url_chunk.image_url
        assert image_url_from_openai.type == image_url_chunk.type
    else:
        assert ImageURLChunk.from_openai(openai_image_url_chunk) == image_url_chunk

    typeddict_openai = OpenAIImageChunk(**openai_image_url_chunk)  # type: ignore[typeddict-item]
    if not isinstance(image_url_chunk.image_url, ImageURL):
        image_url_chunk.image_url = ImageURL(url=image_url_chunk.image_url, detail=None)

    assert ImageURLChunk.from_openai(typeddict_openai) == image_url_chunk


def test_convert_tool() -> None:
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

    tool_openai = tool.to_openai()
    assert tool_openai == (
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
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
            },
        }
    )
    assert Tool.from_openai(tool.to_openai()) == tool

    typeddict_openai = OpenAITool(**tool.to_openai())  # type: ignore[typeddict-item]
    assert Tool.from_openai(typeddict_openai) == tool


def test_convert_tool_call() -> None:
    tool_call = ToolCall(
        id="VvvODy9mT",
        function=FunctionCall(
            name="get_current_weather",
            arguments='{"location": "Paris, France", "format": "celsius"}',
        ),
    )
    tool_call_openai = tool_call.to_openai()

    tool_call_openai == (
        {
            "id": "VvvODy9mT",
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "arguments": '{"location": "Paris, France", "format": "celsius"}',
            },
        }
    )
    assert ToolCall.from_openai(tool_call.to_openai()) == tool_call

    typeddict_openai = OpenAIToolCall(**tool_call.to_openai())  # type: ignore[typeddict-item]
    assert ToolCall.from_openai(typeddict_openai) == tool_call


@pytest.mark.parametrize(
    ["openai_message", "message"],
    [
        ({"role": "user", "content": "Hello"}, UserMessage(content="Hello")),
        (
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            UserMessage(content=[TextChunk(text="Hello")]),
        ),
        (
            OpenAIUserMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                            "detail": "auto",
                        },
                    },
                ],
            ),
            UserMessage(
                content=[
                    TextChunk(text="Describe this image"),
                    ImageURLChunk(
                        image_url=ImageURL(
                            url="https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                            detail="auto",
                        )
                    ),
                ]
            ),
        ),
        (OpenAIAssistantMessage(role="assistant", content="Hi"), AssistantMessage(content="Hi")),
        (
            OpenAIAssistantMessage(
                role="assistant",
                content="Hi",
                tool_calls=[
                    {
                        "id": "VvvODy9mT",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "Paris, France", "format": "celsius"}',
                        },
                    }
                ],
            ),
            AssistantMessage(
                content="Hi",
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
        ),
        (
            OpenAIToolMessage(role="tool", content="22", tool_call_id="VvvODy9mT"),
            ToolMessage(tool_call_id="VvvODy9mT", content="22"),
        ),
        (
            OpenAISystemMessage(role="system", content="You are a helpful assistant."),
            SystemMessage(content="You are a helpful assistant."),
        ),
    ],
)
def test_convert_openai_message_to_message_and_back(openai_message: Dict, message: ChatMessage) -> None:
    assert type(message).from_openai(openai_message) == message
    assert message.to_openai() == openai_message


@pytest.mark.parametrize(
    ["request_cls"],
    [
        (ChatCompletionRequest,),
        (InstructRequest,),
    ],
)
@pytest.mark.parametrize(
    ["openai_messages", "messages", "openai_tools", "tools"],
    [
        (
            [
                OpenAISystemMessage({"role": "system", "content": "You are a helpful assistant."}),
                OpenAIUserMessage({"role": "user", "content": "What's the weather like in Paris?"}),
                OpenAIAssistantMessage(
                    {
                        "role": "assistant",
                        "content": "Let me think...",
                        "tool_calls": [
                            {
                                "id": "VvvODy9mT",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "Paris, France", "format": "celsius"}',
                                },
                            }
                        ],
                    }
                ),
                OpenAIToolMessage({"role": "tool", "content": "22", "tool_call_id": "VvvODy9mT"}),
            ],
            [
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="What's the weather like in Paris?"),
                AssistantMessage(
                    content="Let me think...",
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
            [
                OpenAITool(
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get the current weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "format": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                        "description": "The temperature unit to use. Infer this from the user's location.",  # noqa: E501
                                    },
                                },
                                "required": ["location", "format"],
                            },
                        },
                    }
                )
            ],
            [
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
        ),
        (
            [
                OpenAISystemMessage({"role": "system", "content": "You are a helpful assistant."}),
                OpenAIUserMessage({"role": "user", "content": "What's the weather like in Paris?"}),
                OpenAIAssistantMessage(
                    {
                        "role": "assistant",
                        "content": "How should I know?",
                    }
                ),
            ],
            [
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="What's the weather like in Paris?"),
                AssistantMessage(
                    content="How should I know?",
                ),
            ],
            None,
            None,
        ),
        (
            [
                OpenAIUserMessage({"role": "user", "content": "What's the weather like in Paris?"}),
                OpenAIAssistantMessage(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "VvvODy9mT",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "Paris, France", "format": "celsius"}',
                                },
                            }
                        ],
                    }
                ),
                OpenAIToolMessage({"role": "tool", "content": "22", "tool_call_id": "VvvODy9mT"}),
            ],
            [
                UserMessage(content="What's the weather like in Paris?"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="VvvODy9mT",
                            function=FunctionCall(
                                name="get_current_weather",
                                arguments='{"location": "Paris, France", "format": "celsius"}',
                            ),
                        )
                    ]
                ),
                ToolMessage(tool_call_id="VvvODy9mT", name="get_current_weather", content="22"),
            ],
            [
                OpenAITool(
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get the current weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "format": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                        "description": "The temperature unit to use. Infer this from the user's location.",  # noqa: E501
                                    },
                                },
                                "required": ["location", "format"],
                            },
                        },
                    }
                )
            ],
            [
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
        ),
    ],
)
def test_convert_requests(
    openai_messages: List[Dict[str, Any]],
    messages: List[ChatMessage],
    openai_tools: Optional[List[Dict[str, Any]]],
    tools: Optional[List[Tool]],
    request_cls: Type[Union[ChatCompletionRequest, InstructRequest]],
) -> None:
    request: Union[ChatCompletionRequest, InstructRequest]
    if request_cls == ChatCompletionRequest:
        request = ChatCompletionRequest(
            messages=messages,
            tools=tools,
        )
    else:
        request = InstructRequest(
            messages=messages,
            available_tools=tools,
        )

    openai_request = request.to_openai(stream=True)

    assert openai_request["messages"] == openai_messages
    if tools is not None:
        assert openai_request["tools"] == openai_tools
    else:
        assert "tools" not in openai_request

    if isinstance(request, ChatCompletionRequest):
        assert openai_request["temperature"] == 0.7

    stream = openai_request.pop("stream")
    assert stream is True

    reconstructed_request: Union[ChatCompletionRequest, InstructRequest] = type(request).from_openai(**openai_request)

    for i, reconstructed_message in enumerate(reconstructed_request.messages):
        if isinstance(reconstructed_message, (SystemMessage, UserMessage, AssistantMessage)):
            assert reconstructed_message == messages[i]
        elif isinstance(reconstructed_message, ToolMessage):
            assert reconstructed_message.model_dump(exclude={"name"}) == messages[i].model_dump(exclude={"name"})

    if tools is not None:
        reconstructed_tools = (
            reconstructed_request.tools
            if isinstance(reconstructed_request, ChatCompletionRequest)
            else reconstructed_request.available_tools
        )
        assert isinstance(tools, list)
        assert isinstance(reconstructed_tools, list)

        # Not using zip below because of mypy not recognizing reconstructed_tools as a list of Tools.
        assert len(tools) == len(reconstructed_tools)
        for i in range(len(tools)):
            assert reconstructed_tools[i] == tools[i]

from copy import deepcopy
from typing import Any

import pytest
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam as OpenAIToolCall,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam as OpenAITool
from pydantic import ValidationError

from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall


@pytest.mark.parametrize(
    ("openai_function", "expected"),
    [
        pytest.param(
            {"name": "do_nothing", "unknown_field": "ignored"},
            Function(name="do_nothing", description="", parameters={}),
            id="missing-description-and-parameters",
        ),
        pytest.param(
            {"name": "do_nothing", "description": None, "parameters": None},
            Function(name="do_nothing", description="", parameters={}),
            id="null-description-and-parameters",
        ),
    ],
)
def test_function_from_openai_defaults_missing_or_null_fields(
    openai_function: dict[str, Any], expected: Function
) -> None:
    assert Function.from_openai(openai_function) == expected


def test_function_from_openai_rejects_invalid_recognized_parameters() -> None:
    with pytest.raises(ValidationError, match="parameters"):
        Function.from_openai({"name": "run", "parameters": "not an object"})


def test_tool_round_trip() -> None:
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
            strict=True,
        )
    )
    expected_openai = {
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
            "strict": True,
        },
    }

    assert tool.to_openai() == expected_openai
    assert Tool.from_openai(tool.to_openai()) == tool
    assert Tool.from_openai(OpenAITool(**expected_openai)) == tool  # type: ignore[typeddict-item]


def test_tool_from_openai_defaults_fields_and_does_not_mutate_input() -> None:
    openai_tool: dict[str, Any] = {
        "type": "function",
        "function": {"name": "do_nothing", "unknown_field": "ignored"},
        "extra_openai_field": True,
    }
    original_openai_tool = deepcopy(openai_tool)

    tool = Tool.from_openai(openai_tool)

    assert tool == Tool(function=Function(name="do_nothing", description="", parameters={}))
    assert openai_tool == original_openai_tool


def test_tool_from_openai_drops_unknown_fields() -> None:
    openai_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "",
            "parameters": {"type": "object"},
            "strict": False,
        },
        "extra_openai_field": True,
    }

    assert Tool.from_openai(openai_tool) == Tool(
        function=Function(name="get_weather", description="", parameters={"type": "object"})
    )


def test_tool_call_round_trip() -> None:
    tool_call = ToolCall(
        id="VvvODy9mT",
        function=FunctionCall(
            name="get_current_weather",
            arguments='{"location": "Paris, France", "format": "celsius"}',
        ),
    )
    expected_openai = {
        "id": "VvvODy9mT",
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "arguments": '{"location": "Paris, France", "format": "celsius"}',
        },
    }

    assert tool_call.to_openai() == expected_openai
    assert ToolCall.from_openai(tool_call.to_openai()) == tool_call
    assert ToolCall.from_openai(OpenAIToolCall(**expected_openai)) == tool_call  # type: ignore[typeddict-item]


@pytest.mark.parametrize(
    ("openai_tool_call", "expected"),
    [
        pytest.param(
            {
                "id": "call_123",
                "index": 0,
                "type": "function",
                "function": {"name": "foo", "arguments": "{}"},
            },
            ToolCall(id="call_123", function=FunctionCall(name="foo", arguments="{}")),
            id="tool-call-index",
        ),
        pytest.param(
            {
                "id": "c1",
                "index": 0,
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            },
            ToolCall(id="c1", function=FunctionCall(name="f", arguments="{}")),
            id="openai-extra-index-field",
        ),
    ],
)
def test_tool_call_from_openai_ignores_index(openai_tool_call: dict[str, Any], expected: ToolCall) -> None:
    assert ToolCall.from_openai(openai_tool_call) == expected


@pytest.mark.parametrize(
    "openai_tool_call",
    [
        pytest.param(
            {"id": "c1", "function": {"name": "f"}},
            id="missing-nested-arguments",
        ),
        pytest.param(
            {"id": "c1", "function": {"name": "f", "arguments": []}},
            id="invalid-nested-arguments",
        ),
    ],
)
def test_tool_call_from_openai_rejects_invalid_nested_function_call(openai_tool_call: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="function"):
        ToolCall.from_openai(openai_tool_call)

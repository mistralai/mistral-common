import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import Any, TypeAlias

import pytest
from pydantic import ValidationError

from mistral_common.exceptions import InvalidAssistantMessageException, InvalidMessageStructureException
from mistral_common.protocol.instruct.chunk import AudioURL, AudioURLChunk, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    ReasoningFieldFormat,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    JsonSchema,
    ReasoningEffort,
    ResponseFormat,
    ResponseFormats,
)
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    FunctionCall,
    FunctionName,
    NamedToolChoice,
    Tool,
    ToolCall,
    ToolChoiceEnum,
)

_RequestRoundTripInputs: TypeAlias = tuple[
    list[ChatMessage],
    list[dict[str, Any]],
    list[Tool] | None,
    list[dict[str, Any]] | None,
]


def _weather_tool_round_trip_values() -> tuple[list[Tool], list[dict[str, Any]]]:
    parameters: dict[str, Any] = {
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
    }
    tool = Tool(
        function=Function(
            name="get_current_weather",
            description="Get the current weather",
            parameters=deepcopy(parameters),
        )
    )
    openai_tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": deepcopy(parameters),
            "strict": False,
        },
    }

    return [tool], [openai_tool]


def _weather_tool_result_scenario() -> _RequestRoundTripInputs:
    tools, openai_tools = _weather_tool_round_trip_values()
    messages: list[ChatMessage] = [
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
    ]
    openai_messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like in Paris?"},
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
        },
        {"role": "tool", "content": "22", "tool_call_id": "VvvODy9mT"},
    ]

    return messages, openai_messages, tools, openai_tools


def _no_tool_conversation_scenario() -> _RequestRoundTripInputs:
    messages: list[ChatMessage] = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What's the weather like in Paris?"),
        AssistantMessage(content="How should I know?"),
    ]
    openai_messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like in Paris?"},
        {"role": "assistant", "content": "How should I know?"},
    ]

    return messages, openai_messages, None, None


def _weather_call_scenario() -> _RequestRoundTripInputs:
    tools, openai_tools = _weather_tool_round_trip_values()
    messages: list[ChatMessage] = [
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
    ]
    openai_messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather like in Paris?"},
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
        },
        {"role": "tool", "content": "22", "tool_call_id": "VvvODy9mT"},
    ]

    return messages, openai_messages, tools, openai_tools


def _audio_url_conversation_scenario() -> _RequestRoundTripInputs:
    sample_audio_url = "https://freetestdata.com/wp-content/uploads/2021/09/Free_Test_Data_100KB_MP3.mp3"
    base64_audio_url = "YXVkaW8="
    prefixed_audio_url = f"data:audio/wav;base64,{base64_audio_url}"
    messages: list[ChatMessage] = [
        UserMessage(content="Listen to this"),
        AssistantMessage(content="Pass the URL please."),
        UserMessage(
            content=[
                TextChunk(text="Here it is !"),
                AudioURLChunk(audio_url=AudioURL(url=sample_audio_url)),
                TextChunk(text="What do you think also of these ones?"),
                AudioURLChunk(audio_url=AudioURL(url=sample_audio_url)),
                AudioURLChunk(audio_url=AudioURL(url=base64_audio_url)),
                AudioURLChunk(audio_url=AudioURL(url=prefixed_audio_url)),
            ]
        ),
    ]
    openai_messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Listen to this"},
        {"role": "assistant", "content": "Pass the URL please."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here it is !"},
                {"type": "audio_url", "audio_url": {"url": sample_audio_url}},
                {"type": "text", "text": "What do you think also of these ones?"},
                {"type": "audio_url", "audio_url": {"url": sample_audio_url}},
                {"type": "audio_url", "audio_url": {"url": base64_audio_url}},
                {"type": "audio_url", "audio_url": {"url": prefixed_audio_url}},
            ],
        },
    ]

    return messages, openai_messages, None, None


class TestChatCompletionRequest:
    def test_negative_seed_reports_random_seed_location(self) -> None:
        messages = [UserMessage(content="foo")]

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(model="test-model", messages=messages, random_seed=-1)

        assert exc_info.value.errors()[0]["loc"] == ("random_seed",)

    def test_request_openai_round_trip_preserves_zero_seed_messages_and_tools(self) -> None:
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }
        openai_request: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "Checking the weather.",
                    "tool_calls": [
                        {
                            "id": "weather-call-1",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location": "Paris"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "weather-call-1", "content": "Sunny, 18 C"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather.",
                        "parameters": parameters,
                        "strict": True,
                    },
                }
            ],
            "seed": 0,
            "temperature": 0.25,
            "unsupported_outer_field": "discard this field",
        }
        original_request = deepcopy(openai_request)
        expected_messages: list[ChatMessage] = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is the weather in Paris?"),
            AssistantMessage(
                content="Checking the weather.",
                tool_calls=[
                    ToolCall(
                        id="weather-call-1",
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments='{"location": "Paris"}',
                        ),
                    )
                ],
            ),
            ToolMessage(tool_call_id="weather-call-1", content="Sunny, 18 C"),
        ]
        expected_tools = [
            Tool(
                function=Function(
                    name="get_current_weather",
                    description="Get the current weather.",
                    parameters=deepcopy(parameters),
                    strict=True,
                )
            )
        ]
        expected_openai_export: dict[str, Any] = {
            "temperature": 0.25,
            "top_p": 1.0,
            "response_format": {"type": "text"},
            "continue_final_message": False,
            "seed": 0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "Checking the weather.",
                    "tool_calls": [
                        {
                            "id": "weather-call-1",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location": "Paris"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "Sunny, 18 C", "tool_call_id": "weather-call-1"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                        "strict": True,
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": True,
            "parallel_tool_calls": True,
        }

        request = ChatCompletionRequest.from_openai(**openai_request)

        assert request.messages == expected_messages
        assert request.tools == expected_tools
        assert request.random_seed == 0
        assert request.temperature == 0.25
        assert "unsupported_outer_field" not in request.model_dump()

        openai_export = request.to_openai(stream=True)

        assert openai_export == expected_openai_export
        assert "unsupported_outer_field" not in openai_export
        assert openai_request == original_request

    def test_direct_continue_final_message_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="continue_final_message"):
            ChatCompletionRequest[ChatMessage](  # type: ignore[call-arg]
                messages=[UserMessage(content="foo"), AssistantMessage(content="bar")],
                continue_final_message=True,
            )


def test_request_from_openai_drops_unsupported_fields() -> None:
    request = ChatCompletionRequest.from_openai(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.5,
        stream=False,
        n=2,
        logprobs=True,
        frequency_penalty=0.1,
        unknown_field="value",
    )

    assert request == ChatCompletionRequest(messages=[UserMessage(content="Hello")], temperature=0.5)


def test_request_from_openai_rejects_conflicting_seed_names() -> None:
    with pytest.raises(ValueError, match="Cannot specify both `seed` and `random_seed`"):
        ChatCompletionRequest.from_openai(
            messages=[{"role": "user", "content": "Hello"}],
            seed=7,
            random_seed=7,
        )


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        pytest.param({}, None, id="absent"),
        pytest.param({"max_tokens": 5}, 5, id="max-tokens-only"),
        pytest.param({"max_completion_tokens": 7}, 7, id="max-completion-tokens-only"),
        pytest.param({"max_completion_tokens": None}, None, id="max-completion-tokens-null"),
        pytest.param({"max_tokens": 7, "max_completion_tokens": 7}, 7, id="both-equal"),
        pytest.param({"max_tokens": 7, "max_completion_tokens": None}, 7, id="null-alias-keeps-max-tokens"),
    ],
)
def test_request_from_openai_max_completion_tokens(limits: dict[str, Any], expected: int | None) -> None:
    request = ChatCompletionRequest.from_openai(messages=[{"role": "user", "content": "hi"}], **limits)

    assert request.max_tokens == expected
    assert ChatCompletionRequest.from_openai(**request.to_openai()) == request


def test_request_from_openai_rejects_conflicting_max_tokens() -> None:
    with pytest.raises(ValueError, match=r"`max_tokens` \(5\) and `max_completion_tokens` \(7\)"):
        ChatCompletionRequest.from_openai(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            max_completion_tokens=7,
        )


def test_request_from_openai_rejects_invalid_recognized_value() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        ChatCompletionRequest.from_openai(
            messages=[{"role": "user", "content": "Hello"}],
            temperature="not-a-number",
        )


def test_request_to_openai_forwards_reasoning_field_format() -> None:
    messages: list[ChatMessage] = [
        UserMessage(content="Hi"),
        AssistantMessage(content=[ThinkChunk(thinking="Let me think", closed=True), TextChunk(text="Done")]),
    ]
    request = ChatCompletionRequest(messages=messages)

    openai_request = request.to_openai(reasoning_field_format=ReasoningFieldFormat.reasoning)

    assistant_message = [message for message in openai_request["messages"] if message["role"] == "assistant"][0]
    assert assistant_message == {"role": "assistant", "reasoning": "Let me think", "content": "Done"}


def test_request_to_openai_rejects_interleaved_thinking() -> None:
    messages: list[ChatMessage] = [
        UserMessage(content="Hi"),
        AssistantMessage(content=[TextChunk(text="Done"), ThinkChunk(thinking="Let me think", closed=True)]),
    ]
    request = ChatCompletionRequest(messages=messages)

    with pytest.raises(InvalidAssistantMessageException, match="to_openai does not support interleaved thinking"):
        request.to_openai()


def test_request_from_openai_maps_continuation_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        request = ChatCompletionRequest.from_openai(
            messages=[
                {"role": "user", "content": "foo"},
                {"role": "assistant", "content": "bar"},
            ],
            continue_final_message=True,
        )

    assert isinstance(request.messages[-1], AssistantMessage)
    assert request.messages[-1].prefix is True


@pytest.mark.parametrize(
    ("legacy_value", "expected_prefix"),
    [
        pytest.param(1, True, id="integer-true"),
        pytest.param("true", True, id="string-true"),
        pytest.param(0, False, id="integer-false"),
        pytest.param("false", False, id="string-false"),
    ],
)
def test_request_from_openai_preserves_legacy_boolean_coercion(
    legacy_value: bool | int | str, expected_prefix: bool
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        request = ChatCompletionRequest.from_openai(
            messages=[
                {"role": "user", "content": "foo"},
                {"role": "assistant", "content": "bar"},
            ],
            continue_final_message=legacy_value,  # type: ignore[arg-type]
        )

    assert caught == []
    assert isinstance(request.messages[-1], AssistantMessage)
    assert request.messages[-1].prefix is expected_prefix


def test_request_from_openai_rejects_invalid_continuation_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError, match="valid boolean"):
            ChatCompletionRequest.from_openai(
                messages=[{"role": "user", "content": "foo"}],
                continue_final_message="not-a-bool",  # type: ignore[arg-type]
            )

    assert caught == []


def test_request_from_openai_rejects_true_continuation_for_non_assistant_final() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(InvalidMessageStructureException, match="requires final message to be an assistant"):
            ChatCompletionRequest.from_openai(
                messages=[
                    {"role": "user", "content": "foo"},
                    {"role": "user", "content": "bar"},
                ],
                continue_final_message=True,
            )

    assert caught == []


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(
            [UserMessage(content="foo"), AssistantMessage(content="bar", prefix=True)],
            True,
            id="prefixed-final-assistant",
        ),
        pytest.param(
            [UserMessage(content="foo"), AssistantMessage(content="bar")],
            False,
            id="unprefixed-final-assistant",
        ),
        pytest.param(
            [UserMessage(content="foo")],
            False,
            id="non-assistant-final",
        ),
    ],
)
def test_request_to_openai_derives_continuation_flag(messages: list[ChatMessage], expected: bool) -> None:
    request = ChatCompletionRequest(messages=deepcopy(messages))

    assert request.to_openai()["continue_final_message"] is expected


def test_prefixed_assistant_request_exports_continuation_without_message_prefix() -> None:
    messages: list[ChatMessage] = [UserMessage(content="foo"), AssistantMessage(content="bar", prefix=True)]
    request = ChatCompletionRequest(messages=messages)
    expected_openai_request = {
        "temperature": 0.7,
        "top_p": 1.0,
        "response_format": {"type": "text"},
        "continue_final_message": True,
        "messages": [
            {"role": "user", "content": "foo"},
            {"role": "assistant", "content": "bar"},
        ],
        "tool_choice": "auto",
    }

    exported = request.to_openai()

    assert exported == expected_openai_request
    expected_messages: list[ChatMessage] = [UserMessage(content="foo"), AssistantMessage(content="bar", prefix=True)]
    assert ChatCompletionRequest.from_openai(**exported) == ChatCompletionRequest(messages=expected_messages)


@pytest.mark.parametrize(
    ("tool_choice", "expected_openai", "expected_reconstructed"),
    [
        pytest.param(ToolChoiceEnum.auto, "auto", ToolChoiceEnum.auto.value, id="auto"),
        pytest.param(ToolChoiceEnum.none, "none", ToolChoiceEnum.none.value, id="none"),
        pytest.param(ToolChoiceEnum.required, "required", ToolChoiceEnum.required.value, id="required"),
        pytest.param(ToolChoiceEnum.any, "required", ToolChoiceEnum.required.value, id="any-maps-to-required"),
        pytest.param(
            NamedToolChoice(function=FunctionName(name="get_weather")),
            {"type": "function", "function": {"name": "get_weather"}},
            NamedToolChoice(function=FunctionName(name="get_weather")),
            id="named-tool",
        ),
    ],
)
def test_request_tool_choice_round_trip(
    tool_choice: ToolChoiceEnum | NamedToolChoice,
    expected_openai: str | dict[str, Any],
    expected_reconstructed: str | NamedToolChoice,
) -> None:
    request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], tool_choice=deepcopy(tool_choice))
    openai_request = request.to_openai()

    assert openai_request["tool_choice"] == expected_openai

    reconstructed = ChatCompletionRequest.from_openai(**openai_request)
    assert reconstructed.tool_choice == expected_reconstructed


@pytest.mark.parametrize(
    "scenario_factory",
    [
        pytest.param(_weather_tool_result_scenario, id="weather-tool-result"),
        pytest.param(_no_tool_conversation_scenario, id="no-tool-conversation"),
        pytest.param(_weather_call_scenario, id="weather-call-with-tool-result"),
        pytest.param(_audio_url_conversation_scenario, id="audio-url-conversation"),
    ],
)
@pytest.mark.parametrize(
    "reasoning_effort",
    [
        pytest.param(None, id="effort-absent"),
        pytest.param(ReasoningEffort.none, id="effort-none"),
        pytest.param(ReasoningEffort.high, id="effort-high"),
    ],
)
def test_request_openai_round_trip_preserves_message_scenarios(
    scenario_factory: Callable[[], _RequestRoundTripInputs], reasoning_effort: ReasoningEffort | None
) -> None:
    messages, expected_openai_messages, tools, expected_openai_tools = scenario_factory()
    expected_messages = deepcopy(messages)
    expected_tools = deepcopy(tools)
    request = ChatCompletionRequest(messages=messages, tools=tools, reasoning_effort=reasoning_effort)

    openai_request = request.to_openai(stream=True)

    expected_openai_request: dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 1.0,
        "response_format": {"type": "text"},
        "continue_final_message": False,
        "messages": expected_openai_messages,
        "tool_choice": "auto",
        "stream": True,
    }
    if expected_openai_tools is not None:
        expected_openai_request["tools"] = expected_openai_tools
        expected_openai_request["parallel_tool_calls"] = True
    if reasoning_effort is not None:
        expected_openai_request["reasoning_effort"] = reasoning_effort.value
    assert openai_request == expected_openai_request

    reconstructed_request = ChatCompletionRequest.from_openai(**openai_request)

    assert "stream" not in reconstructed_request.model_dump()
    assert reconstructed_request.temperature == 0.7
    assert len(reconstructed_request.messages) == len(expected_messages)
    for index, reconstructed_message in enumerate(reconstructed_request.messages):
        expected_message = expected_messages[index]
        if isinstance(expected_message, ToolMessage):
            assert reconstructed_message.model_dump(exclude={"name"}) == expected_message.model_dump(exclude={"name"})
        else:
            assert reconstructed_message == expected_message

    assert reconstructed_request.tools == expected_tools
    assert reconstructed_request.reasoning_effort == reasoning_effort


@pytest.mark.parametrize("parallel_tool_calls", [True, False], ids=["true", "false"])
def test_request_to_openai_with_tools_includes_parallel_tool_calls(parallel_tool_calls: bool) -> None:
    request = ChatCompletionRequest(
        messages=[UserMessage(content="Hello")],
        tools=[Tool(function=Function(name="get_weather", parameters={}))],
        parallel_tool_calls=parallel_tool_calls,
    )

    assert request.to_openai()["parallel_tool_calls"] is parallel_tool_calls


@pytest.mark.parametrize("tools", [None, []], ids=["no-tools", "empty-tools"])
def test_request_to_openai_without_tools_omits_parallel_tool_calls(tools: list[Tool] | None) -> None:
    request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], tools=tools, parallel_tool_calls=False)

    assert "parallel_tool_calls" not in request.to_openai()


def test_request_parallel_tool_calls_defaults_to_true() -> None:
    request = ChatCompletionRequest(messages=[UserMessage(content="Hello")])

    assert request.parallel_tool_calls is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(False, False, id="false"),
        pytest.param(True, True, id="true"),
        pytest.param(None, True, id="none-maps-to-default-true"),
    ],
)
def test_request_from_openai_parallel_tool_calls(value: bool | None, expected: bool) -> None:
    request = ChatCompletionRequest.from_openai(
        messages=[{"role": "user", "content": "Hello"}], parallel_tool_calls=value
    )

    assert request.parallel_tool_calls is expected


@pytest.mark.parametrize("field", ["temperature", "top_p"])
def test_request_from_openai_null_sampling_param_maps_to_default(field: str) -> None:
    null_param: dict[str, Any] = {field: None}
    request = ChatCompletionRequest.from_openai(messages=[{"role": "user", "content": "Hello"}], **null_param)

    assert request == ChatCompletionRequest(messages=[UserMessage(content="Hello")])


def test_request_to_openai_renames_custom_schema_to_schema() -> None:
    schema: dict[str, Any] = {"type": "object"}
    response_format = ResponseFormat(type=ResponseFormats.json_schema, json_schema=JsonSchema(name="x", schema=schema))

    exported = ChatCompletionRequest(messages=[UserMessage(content="hi")], response_format=response_format).to_openai()

    assert exported["response_format"]["json_schema"]["schema"] == schema
    assert "custom_schema" not in exported["response_format"]["json_schema"]


def test_request_from_openai_accepts_schema_key() -> None:
    schema: dict[str, Any] = {"type": "object"}

    request = ChatCompletionRequest.from_openai(
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
    )

    assert request.response_format is not None
    assert request.response_format.json_schema is not None
    assert request.response_format.json_schema.custom_schema == schema


@pytest.mark.parametrize(
    ("strict", "expected"),
    [
        pytest.param({"strict": True}, True, id="true"),
        pytest.param({"strict": False}, False, id="false"),
        pytest.param({"strict": None}, False, id="none-maps-to-default-false"),
        pytest.param({}, False, id="absent"),
    ],
)
def test_request_from_openai_json_schema_strict(strict: dict[str, bool | None], expected: bool) -> None:
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {"name": "x", "schema": {"type": "object"}, **strict},
    }
    original = deepcopy(response_format)

    request = ChatCompletionRequest.from_openai(
        messages=[{"role": "user", "content": "hi"}], response_format=response_format
    )

    assert response_format == original
    assert request.response_format is not None
    assert request.response_format.json_schema is not None
    assert request.response_format.json_schema.strict is expected
    assert ChatCompletionRequest.from_openai(**request.to_openai()) == request


def test_reasoning_effort_order() -> None:
    assert list(ReasoningEffort) == [
        ReasoningEffort.none,
        ReasoningEffort.minimal,
        ReasoningEffort.low,
        ReasoningEffort.medium,
        ReasoningEffort.high,
        ReasoningEffort.xhigh,
        ReasoningEffort.max,
    ]

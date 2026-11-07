import pickle

import pytest

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder

REQUEST: ChatCompletionRequest = ChatCompletionRequest(
    messages=[
        SystemMessage(content="Be brief."),
        UserMessage(content="Weather in Paris?"),
        AssistantMessage(
            tool_calls=[
                ToolCall(id="abcdefghi", function=FunctionCall(name="get_weather", arguments='{"city": "Paris"}'))
            ]
        ),
        ToolMessage(tool_call_id="abcdefghi", content="Sunny"),
    ],
    tools=[Tool(function=Function(name="get_weather", description="Get the weather", parameters={}))],
)


@pytest.mark.parametrize("version", list(TokenizerVersion))
def test_normalizer_and_instruct_request_pickle(version: TokenizerVersion) -> None:
    builder = ModelSettingsBuilder() if version >= TokenizerVersion.v15 else None
    normalizer = get_normalizer(version, builder)
    instruct_request = normalizer.from_chat_completion_request(REQUEST)

    restored_request = pickle.loads(pickle.dumps(instruct_request))
    restored_normalizer = pickle.loads(pickle.dumps(normalizer))

    assert restored_request == instruct_request
    assert restored_normalizer.from_chat_completion_request(REQUEST) == instruct_request


# Valid JSON that `json.loads` rejects with a plain ValueError (integer over the int digit limit).
_OVERSIZED_INT_JSON = '{"a":' + "1" * 5000 + "}"


@pytest.mark.parametrize(argnames="version", argvalues=[TokenizerVersion.v3, TokenizerVersion.v7, TokenizerVersion.v13])
def test_keeps_json_with_oversized_int_unchanged(version: TokenizerVersion) -> None:
    r"""Pass tool-call arguments and tool content that json.loads rejects through unchanged."""
    request = ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(
                tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="b", arguments=_OVERSIZED_INT_JSON))]
            ),
            ToolMessage(content=_OVERSIZED_INT_JSON, tool_call_id="123456789"),
        ]
    )

    _, assistant, tool = get_normalizer(version=version).from_chat_completion_request(request).messages

    assert isinstance(assistant, AssistantMessage) and assistant.tool_calls is not None
    assert assistant.tool_calls[0].function.arguments == _OVERSIZED_INT_JSON
    assert isinstance(tool, ToolMessage)
    assert tool.content == _OVERSIZED_INT_JSON

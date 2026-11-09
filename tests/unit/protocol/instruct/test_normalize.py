import pickle

import pytest

from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage
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

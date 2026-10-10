import pytest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.exceptions import InvalidToolException

def test_validator_rejects_trailing_newline_in_tool_name():
    tokenizer = MistralTokenizer.v7()
    request = ChatCompletionRequest(
        messages=[UserMessage(content="hi")],
        tools=[Tool(function=Function(name="get_weather\n", parameters={"type": "object", "properties": {}}))],
    )
    with pytest.raises(InvalidToolException):
        tokenizer.encode_chat_completion(request)

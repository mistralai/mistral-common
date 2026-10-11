import pytest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.exceptions import InvalidToolException, InvalidFunctionCallException
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

def test_function_name_with_trailing_newline_fails():
    tokenizer = MistralTokenizer.v7()
    
    with pytest.raises(InvalidToolException):
        tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[UserMessage(content="hi")],
                tools=[Tool(function=Function(name="get_weather\n", parameters={"type": "object", "properties": {}}))],
            )
        )

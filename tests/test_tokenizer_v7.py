import pytest

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizerV7
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidatorV5,
    ValidationMode,
)
from mistral_common.tokens.tokenizers.base import (
    InstructRequest,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV7
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import quick_vocab
from tests.test_tokenizer_v7_audio import get_tekkenizer_with_audio


@pytest.fixture
def no_audio_tekkenizer() -> InstructTokenizerV7:
    tokenizer = Tekkenizer(
        quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS),
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v7,
    )
    return InstructTokenizerV7(tokenizer)


@pytest.fixture
def with_audio_tekkenizer() -> InstructTokenizerV7:
    return get_tekkenizer_with_audio()


@pytest.mark.parametrize("tekkenizer", ["no_audio_tekkenizer", "with_audio_tekkenizer"])
def test_assistant_tool_call_and_content(request: pytest.FixtureRequest, tekkenizer: str) -> None:
    tokenizer = request.getfixturevalue(tekkenizer)
    instruct_request: InstructRequest = InstructRequest(
        available_tools=[
            Tool(function=Function(name="t1", parameters={})),
            Tool(function=Function(name="t2", parameters={})),
        ],
        messages=[
            UserMessage(content="a"),
            AssistantMessage(
                content="b1b2",
                tool_calls=[
                    ToolCall(id="000000000", function=FunctionCall(name="t1", arguments="{}")),
                    ToolCall(id="111111111", function=FunctionCall(name="t2", arguments="{}")),
                ],
            ),
        ],
    )
    tokenized = tokenizer.encode_instruct(instruct_request)
    tokens = tokenized.tokens

    # make sure it also works end to end
    tools = instruct_request.available_tools
    exclude = {"system_prompt", "truncate_at_max_tokens", "available_tools", "settings"}
    chat_completion_request = ChatCompletionRequest(**instruct_request.model_dump(exclude=exclude), tools=tools)
    validator = MistralRequestValidatorV5(mode=ValidationMode.finetuning)
    normalizer = InstructRequestNormalizerV7.normalizer()

    mistral_tokenizer = MistralTokenizer(tokenizer, validator, normalizer)
    tokens_2 = mistral_tokenizer.encode_chat_completion(chat_completion_request)

    assert tokens == tokens_2.tokens


@pytest.mark.parametrize("tekkenizer", ["no_audio_tekkenizer", "with_audio_tekkenizer"])
def test_assistant_tool_call_and_content_end_to_end(request: pytest.FixtureRequest, tekkenizer: str) -> None:
    tokenizer = request.getfixturevalue(tekkenizer)
    instruct_request: InstructRequest = InstructRequest(
        available_tools=[
            Tool(function=Function(name="t1", parameters={})),
            Tool(function=Function(name="t2", parameters={})),
        ],
        messages=[
            UserMessage(content="a"),
            AssistantMessage(
                content="b1b2",
                tool_calls=[
                    ToolCall(id="000000000", function=FunctionCall(name="t1", arguments="{}")),
                    ToolCall(id="111111111", function=FunctionCall(name="t2", arguments="{}")),
                ],
            ),
            ToolMessage(content="r1", tool_call_id="000000000"),
            ToolMessage(content="r2", tool_call_id="111111111"),
        ],
    )
    tokens = tokenizer.encode_instruct(instruct_request).tokens

    tools = instruct_request.available_tools
    exclude = {"system_prompt", "truncate_at_max_tokens", "available_tools", "settings"}
    chat_completion_request = ChatCompletionRequest(
        **instruct_request.model_dump(exclude=exclude), model="test-model", tools=tools
    )
    validator = MistralRequestValidatorV5(mode=ValidationMode.serving)
    normalizer = InstructRequestNormalizerV7.normalizer()

    mistral_tokenizer = MistralTokenizer(tokenizer, validator, normalizer)
    tokens_2 = mistral_tokenizer.encode_chat_completion(chat_completion_request)

    assert tokens == tokens_2.tokens

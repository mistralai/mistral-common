import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from mistral_common.protocol.instruct.chunk import ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.protocol.instruct.tool_calls import FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13, InstructTokenizerV14
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab

EXPECTED_TOKENS_V14: list[int] = [
    1,
    17,
    183,
    18,
    5,
    191,
    223,
    134,
    216,
    221,
    212,
    201,
    134,
    158,
    132,
    134,
    202,
    217,
    210,
    199,
    216,
    205,
    211,
    210,
    134,
    144,
    132,
    134,
    202,
    217,
    210,
    199,
    216,
    205,
    211,
    210,
    134,
    158,
    132,
    223,
    134,
    210,
    197,
    209,
    201,
    134,
    158,
    132,
    134,
    209,
    197,
    216,
    204,
    195,
    205,
    210,
    216,
    201,
    214,
    212,
    214,
    201,
    216,
    201,
    214,
    134,
    144,
    132,
    134,
    200,
    201,
    215,
    199,
    214,
    205,
    212,
    216,
    205,
    211,
    210,
    134,
    158,
    132,
    134,
    171,
    201,
    216,
    132,
    216,
    204,
    201,
    132,
    218,
    197,
    208,
    217,
    201,
    132,
    211,
    202,
    132,
    197,
    210,
    132,
    197,
    214,
    205,
    216,
    204,
    209,
    201,
    216,
    205,
    199,
    132,
    201,
    220,
    212,
    214,
    201,
    215,
    215,
    205,
    211,
    210,
    146,
    134,
    144,
    132,
    134,
    212,
    197,
    214,
    197,
    209,
    201,
    216,
    201,
    214,
    215,
    134,
    158,
    132,
    223,
    134,
    216,
    221,
    212,
    201,
    134,
    158,
    132,
    134,
    211,
    198,
    206,
    201,
    199,
    216,
    134,
    144,
    132,
    134,
    212,
    214,
    211,
    212,
    201,
    214,
    216,
    205,
    201,
    215,
    134,
    158,
    132,
    223,
    134,
    201,
    220,
    212,
    214,
    201,
    215,
    215,
    205,
    211,
    210,
    134,
    158,
    132,
    223,
    134,
    216,
    221,
    212,
    201,
    134,
    158,
    132,
    134,
    215,
    216,
    214,
    205,
    210,
    203,
    134,
    144,
    132,
    134,
    200,
    201,
    215,
    199,
    214,
    205,
    212,
    216,
    205,
    211,
    210,
    134,
    158,
    132,
    134,
    177,
    197,
    216,
    204,
    132,
    201,
    220,
    212,
    214,
    201,
    215,
    215,
    205,
    211,
    210,
    146,
    134,
    225,
    225,
    225,
    225,
    225,
    193,
    6,
    37,
    223,
    134,
    214,
    201,
    197,
    215,
    211,
    210,
    205,
    210,
    203,
    195,
    201,
    202,
    202,
    211,
    214,
    216,
    134,
    158,
    132,
    134,
    204,
    205,
    203,
    204,
    134,
    225,
    38,
    3,
    185,
    149,
    4,
    165,
    149,
    9,
    170,
    149,
    32,
    223,
    225,
    9,
    170,
    150,
    32,
    223,
    225,
    2,
    7,
    182,
    149,
    8,
    7,
    182,
    150,
    8,
    165,
    150,
    2,
    3,
    185,
    150,
    4,
]
EXPECTED_TEXT_V14: str = r'<s>[SYSTEM_PROMPT]S[/SYSTEM_PROMPT][AVAILABLE_TOOLS][{"type": "function", "function": {"name": "math_interpreter", "description": "Get the value of an arithmetic expression.", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression."}}}}}][/AVAILABLE_TOOLS][MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS][INST]U1[/INST]A1[TOOL_CALLS]F1[ARGS]{}[TOOL_CALLS]F2[ARGS]{}</s>[TOOL_RESULTS]R1[/TOOL_RESULTS][TOOL_RESULTS]R2[/TOOL_RESULTS]A2</s>[INST]U2[/INST]'  # noqa: E501


@pytest.fixture
def available_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "math_interpreter",
                "description": "Get the value of an arithmetic expression.",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression.",
                        }
                    },
                },
            },
        }
    ]


@pytest.fixture
def messages() -> list[BaseMessage]:
    return [
        SystemMessage(content="S"),
        UserMessage(content="U1"),
        AssistantMessage(
            content="A1",
            tool_calls=[
                ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{}")),
                ToolCall(id="999999999", function=FunctionCall(name="F2", arguments="{}")),
            ],
        ),
        ToolMessage(content="R1", tool_call_id="123456789"),
        ToolMessage(content="R2", tool_call_id="999999999"),
        AssistantMessage(content="A2"),
        UserMessage(content="U2"),
    ]


def _get_tekkenizer(version: TokenizerVersion) -> Tekkenizer:
    special_tokens = get_special_tokens(version, add_think=True)
    return Tekkenizer(
        quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=version,
    )


@pytest.fixture(scope="module")
def v13_tokenizer() -> InstructTokenizerV13:
    return InstructTokenizerV13(
        tokenizer=_get_tekkenizer(version=TokenizerVersion.v13), image_encoder=None, audio_encoder=None
    )


@pytest.fixture(scope="module")
def v14_tokenizer() -> InstructTokenizerV14:
    return InstructTokenizerV14(
        tokenizer=_get_tekkenizer(version=TokenizerVersion.v14), image_encoder=None, audio_encoder=None
    )


@pytest.fixture(scope="module")
def mistral_v14_tokenizer(v14_tokenizer: InstructTokenizerV14) -> MistralTokenizer:
    return MistralTokenizer(
        instruct_tokenizer=v14_tokenizer,
        validator=get_validator(version=TokenizerVersion.v14, mode=ValidationMode.test),
        request_normalizer=get_normalizer(version=TokenizerVersion.v14),
    )


def test_tools_and_reasoning_effort(
    mistral_v14_tokenizer: MistralTokenizer, available_tools: list[Tool], messages: list[BaseMessage]
) -> None:
    request_reasoning_effort: InstructRequest = InstructRequest(
        messages=messages, available_tools=available_tools, reasoning_effort=ReasoningEffort.high
    )
    tokenized_v14 = mistral_v14_tokenizer.instruct_tokenizer.encode_instruct(request_reasoning_effort)

    assert tokenized_v14.text == EXPECTED_TEXT_V14, tokenized_v14.text
    assert tokenized_v14.tokens == EXPECTED_TOKENS_V14, tokenized_v14.tokens

    chat_request_reasoning_effort: ChatCompletionRequest = ChatCompletionRequest(
        messages=messages, tools=available_tools, reasoning_effort=ReasoningEffort.high
    )
    mistral_tokenized_v14 = mistral_v14_tokenizer.encode_chat_completion(chat_request_reasoning_effort)

    assert mistral_tokenized_v14.text == EXPECTED_TEXT_V14, mistral_tokenized_v14.text
    assert mistral_tokenized_v14.tokens == EXPECTED_TOKENS_V14, mistral_tokenized_v14.tokens


def test_no_tools_and_reasoning_effort(mistral_v14_tokenizer: MistralTokenizer, messages: list[BaseMessage]) -> None:
    request_reasoning_effort: InstructRequest = InstructRequest(
        messages=messages, available_tools=None, reasoning_effort=ReasoningEffort.none
    )
    tokenized_v14 = mistral_v14_tokenizer.instruct_tokenizer.encode_instruct(request_reasoning_effort)
    expected_text_no_tools = re.sub(r"\[AVAILABLE_TOOLS\].*?\[/AVAILABLE_TOOLS\]", "", EXPECTED_TEXT_V14).replace(
        "high", "none"
    )
    assert tokenized_v14.text == expected_text_no_tools, tokenized_v14.text

    chat_request_reasoning_effort: ChatCompletionRequest = ChatCompletionRequest(
        messages=messages, tools=None, reasoning_effort=ReasoningEffort.none
    )
    mistral_tokenized_v14 = mistral_v14_tokenizer.encode_chat_completion(chat_request_reasoning_effort)
    assert mistral_tokenized_v14.text == expected_text_no_tools, mistral_tokenized_v14.text


def test_reasoning_effort_v13(
    v13_tokenizer: InstructTokenizerV13,
    available_tools: list[Tool],
    messages: list[BaseMessage],
) -> None:
    # InstructRequest does not encode reasoning_effort
    request_reasoning_effort: InstructRequest = InstructRequest(
        messages=messages, available_tools=available_tools, reasoning_effort=ReasoningEffort.high
    )
    tokenized_v13 = v13_tokenizer.encode_instruct(request_reasoning_effort)
    expected_text_no_settings = re.sub(r"\[MODEL_SETTINGS\].*?\[/MODEL_SETTINGS\]", "", EXPECTED_TEXT_V14)
    assert tokenized_v13.text == expected_text_no_settings, tokenized_v13.text


def test_system_think_chunk_raises_v13_v14(
    v13_tokenizer: InstructTokenizerV13, v14_tokenizer: InstructTokenizerV14
) -> None:
    messages = [SystemMessage(content=[ThinkChunk(thinking="Hi")])]
    # InstructRequest does not encode reasoning_effort
    request_v13: InstructRequest = InstructRequest(messages=messages, reasoning_effort=None)
    request_v14: InstructRequest = InstructRequest(messages=messages, reasoning_effort=ReasoningEffort.high)
    _ = v13_tokenizer.encode_instruct(request_v13)
    with pytest.raises(ValueError, match=r"ThinkChunk in system message is not supported for tokenizers >= v14."):
        v14_tokenizer.encode_instruct(request_v14)

    mistral_tokenizer_v14: MistralTokenizer = MistralTokenizer(
        instruct_tokenizer=MagicMock(),  # Magic to make sure the validator is the one throwing the error.
        request_normalizer=get_normalizer(version=TokenizerVersion.v14),
        validator=get_validator(version=TokenizerVersion.v14, mode=ValidationMode.test),
    )
    chat_request_v14: ChatCompletionRequest = ChatCompletionRequest(
        messages=messages, reasoning_effort=ReasoningEffort.high
    )

    with pytest.raises(ValueError, match=r"ThinkChunk in system message is not supported for tokenizers >= v14."):
        mistral_tokenizer_v14.encode_chat_completion(chat_request_v14)

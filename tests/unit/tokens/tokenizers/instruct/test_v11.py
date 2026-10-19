r"""Direct formatting tests for the v11 instruct tokenizer."""

from collections.abc import Callable

import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV11
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.utils import assert_tokenized_equal, decode_keep

# fmt: off
_ONE_CALL_WITHOUT_ID_TOKENS = [1, 3, 197, 4, 9, 197, 195, 197, 195, 197, 32, 134, 198, 208, 197, 198, 208, 197, 134, 2]
_PREFIXED_TEXT_TOKENS = [1, 3, 197, 4, 134, 198, 208, 197, 198, 208, 197, 134]
_PREFIXED_TEXT_PREFIX_IDS = [134, 198, 208, 197, 198, 208, 197, 134]
_TWO_CALLS_WITHOUT_ID_TOKENS = [
    1, 3, 197, 4, 9, 197, 195, 197, 195, 197, 32, 134, 198, 208, 197, 198, 208, 197, 134, 9, 198, 32, 134, 198, 208,
    217, 134, 2
]
_CALL_WITH_ID_BEFORE_LAST_USER_TOKENS = [
    1, 3, 197, 4, 9, 197, 195, 197, 195, 197, 33, 165, 166, 167, 32, 134, 198, 208, 197, 198, 208, 197, 134, 2, 3,
    199, 4
]
_NON_ASCII_CALL_ARGUMENTS_TOKENS = [
    1, 3, 197, 4, 9, 202, 32, 223, 134, 220, 134, 158, 132, 134, 295, 269, 134, 225, 2
]
# fmt: on

_ONE_CALL_WITHOUT_ID_EXPECTED = Tokenized(tokens=_ONE_CALL_WITHOUT_ID_TOKENS)
_PREFIXED_TEXT_EXPECTED = Tokenized(tokens=_PREFIXED_TEXT_TOKENS, prefix_ids=_PREFIXED_TEXT_PREFIX_IDS)
_TWO_CALLS_WITHOUT_ID_EXPECTED = Tokenized(tokens=_TWO_CALLS_WITHOUT_ID_TOKENS)
_CALL_WITH_ID_BEFORE_LAST_USER_EXPECTED = Tokenized(tokens=_CALL_WITH_ID_BEFORE_LAST_USER_TOKENS)
_NON_ASCII_CALL_ARGUMENTS_EXPECTED = Tokenized(tokens=_NON_ASCII_CALL_ARGUMENTS_TOKENS)

_ONE_CALL_WITHOUT_ID_TEXT = '<s>[INST]a[/INST][TOOL_CALLS]a_a_a[ARGS]"blabla"</s>'
_PREFIXED_TEXT_TEXT = '<s>[INST]a[/INST]"blabla"'
_TWO_CALLS_WITHOUT_ID_TEXT = '<s>[INST]a[/INST][TOOL_CALLS]a_a_a[ARGS]"blabla"[TOOL_CALLS]b[ARGS]"blu"</s>'
_CALL_WITH_ID_BEFORE_LAST_USER_TEXT = '<s>[INST]a[/INST][TOOL_CALLS]a_a_a[CALL_ID]ABC[ARGS]"blabla"</s>[INST]c[/INST]'
_NON_ASCII_CALL_ARGUMENTS_TEXT = '<s>[INST]a[/INST][TOOL_CALLS]f[ARGS]{"x": "é"}</s>'

_THINK_NOT_IMPLEMENTED_MESSAGE = r"^Think not implemented for tokenizer < V13\.$"


@pytest.fixture(scope="module")
def tokenizer() -> InstructTokenizerV11:
    r"""Synthetic formatting-only tokenizer; its IDs are not released-vocabulary evidence."""
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v11),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v11,
    )
    return InstructTokenizerV11(tokenizer=tekkenizer)


def _build_user_then_assistant_request(
    assistant: AssistantMessage,
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by the supplied assistant message."""
    return InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="a"), assistant])


def _build_one_call_without_id_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build an assistant tool call without an ID."""
    return _build_user_then_assistant_request(
        assistant=AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"))])
    )


def _build_prefixed_text_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a prefixed assistant text turn."""
    return _build_user_then_assistant_request(assistant=AssistantMessage(content='"blabla"', prefix=True))


def _build_two_calls_without_id_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build an assistant turn with two tool calls without IDs."""
    return _build_user_then_assistant_request(
        assistant=AssistantMessage(
            tool_calls=[
                ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla")),
                ToolCall(function=FunctionCall(name="b", arguments="blu")),
            ]
        )
    )


def _build_call_with_id_before_last_user_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a tool call with an ID followed by the final user turn."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"), id="ABC")]),
            UserMessage(content="c"),
        ]
    )


def _build_non_ascii_call_arguments_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a tool call whose JSON arguments contain non-ASCII text."""
    return _build_user_then_assistant_request(
        assistant=AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="f", arguments='{"x": "é"}'))])
    )


def _build_assistant_think_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build assistant content with a think chunk followed by text."""
    return _build_user_then_assistant_request(
        assistant=AssistantMessage(content=[ThinkChunk(thinking="t"), TextChunk(text="b")])
    )


@pytest.mark.parametrize(
    argnames="request_builder, expected, expected_text",
    argvalues=[
        pytest.param(
            _build_one_call_without_id_request,
            _ONE_CALL_WITHOUT_ID_EXPECTED,
            _ONE_CALL_WITHOUT_ID_TEXT,
            id="one-call-without-id",
        ),
        pytest.param(
            _build_prefixed_text_request,
            _PREFIXED_TEXT_EXPECTED,
            _PREFIXED_TEXT_TEXT,
            id="prefixed-text",
        ),
        pytest.param(
            _build_two_calls_without_id_request,
            _TWO_CALLS_WITHOUT_ID_EXPECTED,
            _TWO_CALLS_WITHOUT_ID_TEXT,
            id="two-calls-without-id",
        ),
        pytest.param(
            _build_call_with_id_before_last_user_request,
            _CALL_WITH_ID_BEFORE_LAST_USER_EXPECTED,
            _CALL_WITH_ID_BEFORE_LAST_USER_TEXT,
            id="call-with-id-before-last-user",
        ),
        pytest.param(
            _build_non_ascii_call_arguments_request,
            _NON_ASCII_CALL_ARGUMENTS_EXPECTED,
            _NON_ASCII_CALL_ARGUMENTS_TEXT,
            id="non-ascii-call-arguments",
        ),
    ],
)
def test_encode_instruct(
    tokenizer: InstructTokenizerV11,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected: Tokenized,
    expected_text: str,
) -> None:
    r"""Preserve complete v11 instruct encodings for each request."""
    tokenized = tokenizer.encode_instruct(request=request_builder())
    assert_tokenized_equal(actual=tokenized, expected=expected)
    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text


@pytest.mark.parametrize(
    argnames="request_builder, expected_exception, expected_message",
    argvalues=[
        pytest.param(
            _build_assistant_think_request,
            TokenizerException,
            _THINK_NOT_IMPLEMENTED_MESSAGE,
            id="assistant-think",
        ),
    ],
)
def test_encode_instruct_rejected(
    tokenizer: InstructTokenizerV11,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected_exception: type[Exception],
    expected_message: str | None,
) -> None:
    r"""Preserve direct v11 exception types and specified messages."""
    instruct_request = request_builder()

    with pytest.raises(expected_exception=expected_exception, match=expected_message):
        tokenizer.encode_instruct(request=instruct_request)

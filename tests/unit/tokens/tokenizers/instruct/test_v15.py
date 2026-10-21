r"""Direct formatting tests for the v15 instruct tokenizer."""

from collections.abc import Callable

import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.chunk import ContentChunk, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import ChatMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest, ModelSettings, ReasoningEffort
from mistral_common.protocol.instruct.tool_calls import Tool
from mistral_common.tokens.tokenizers.base import Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV15
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.utils import assert_tokenized_equal, decode_keep

# fmt: off
_SYSTEM_TEXT_TOKENS = [
    1, 17, 183, 149, 18, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216,
    134, 158, 132, 134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4
]
# fmt: on

_SYSTEM_TEXT_EXPECTED = Tokenized(tokens=_SYSTEM_TEXT_TOKENS)
_SYSTEM_TEXT_TEXT = (
    '<s>[SYSTEM_PROMPT]S1[/SYSTEM_PROMPT][MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS][INST]a[/INST]'
)
_SYSTEM_THINK_MESSAGE = r"^ThinkChunk in system message is not supported for this model$"


@pytest.fixture(scope="module")
def tokenizer() -> InstructTokenizerV15:
    r"""Synthetic formatting-only tokenizer; its IDs are not released-vocabulary evidence."""
    model_settings_builder = ModelSettingsBuilder(
        reasoning_effort=EnumBuilder[ReasoningEffort](
            values=list(ReasoningEffort),
            accepts_none=True,
            default=list(ReasoningEffort)[0],
        )
    )
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v15, add_think=True),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v15,
        model_settings_builder=model_settings_builder,
    )
    return InstructTokenizerV15(tokenizer=tekkenizer)


def _build_system_request(content: str | list[ContentChunk]) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message followed by a user turn with high reasoning effort."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content=content), UserMessage(content="a")],
        settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
    )


def _build_system_text_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with string content."""
    return _build_system_request(content="S1")


def _build_system_text_chunk_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with one text chunk."""
    return _build_system_request(content=[TextChunk(text="S1")])


def _build_system_think_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message containing a think chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content=[ThinkChunk(thinking="Hi")])],
        settings=ModelSettings(reasoning_effort=ReasoningEffort.high),
    )


def _build_system_text_then_think_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with text followed by a think chunk."""
    return _build_system_request(content=[TextChunk(text="S1"), ThinkChunk(thinking="Hi")])


@pytest.mark.parametrize(
    argnames="request_builder, expected, expected_text",
    argvalues=[
        pytest.param(
            _build_system_text_request,
            _SYSTEM_TEXT_EXPECTED,
            _SYSTEM_TEXT_TEXT,
            id="system-text",
        ),
        pytest.param(
            _build_system_text_chunk_request,
            _SYSTEM_TEXT_EXPECTED,
            _SYSTEM_TEXT_TEXT,
            id="system-text-chunk",
        ),
    ],
)
def test_encode_instruct(
    tokenizer: InstructTokenizerV15,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected: Tokenized,
    expected_text: str,
) -> None:
    r"""Preserve complete v15 system encodings with model settings."""
    tokenized = tokenizer.encode_instruct(request=request_builder())
    assert_tokenized_equal(actual=tokenized, expected=expected)
    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text


@pytest.mark.parametrize(
    argnames="request_builder, expected_exception, expected_message",
    argvalues=[
        pytest.param(
            _build_system_think_request,
            TokenizerException,
            _SYSTEM_THINK_MESSAGE,
            id="system-think",
        ),
        pytest.param(
            _build_system_text_then_think_request,
            TokenizerException,
            _SYSTEM_THINK_MESSAGE,
            id="system-text-then-think",
        ),
    ],
)
def test_encode_instruct_rejected(
    tokenizer: InstructTokenizerV15,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected_exception: type[Exception],
    expected_message: str | None,
) -> None:
    r"""Preserve direct v15 exception types and specified messages."""
    instruct_request = request_builder()

    with pytest.raises(expected_exception=expected_exception, match=expected_message):
        tokenizer.encode_instruct(request=instruct_request)

r"""Direct formatting tests for the bundled v1 instruct tokenizer."""

from collections.abc import Callable

import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import InstructTokenizer, Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils import assert_tokenized_equal, decode_keep

# fmt: off
_NORMAL_TOKENS = [
    1, 733, 16289, 28793, 264, 733, 28748, 16289, 28793, 287, 2, 733, 16289, 28793, 277, 733, 28748, 16289, 28793, 281,
    2
]
_SYSTEM_SINGLETURN_TOKENS = [1, 733, 16289, 28793, 17121, 22526, 13, 13, 28708, 733, 28748, 16289, 28793]
_SYSTEM_MULTITURN_TOKENS = [
    1, 733, 16289, 28793, 17121, 22526, 13, 13, 28708, 733, 28748, 16289, 28793, 287, 2, 733, 16289, 28793, 277, 733,
    28748, 16289, 28793, 281, 2
]
_PREFIXED_FINAL_MESSAGE_TOKENS = [
    1, 733, 16289, 28793, 17121, 22526, 13, 13, 28708, 733, 28748, 16289, 28793, 287, 2, 733, 16289, 28793, 277, 733,
    28748, 16289, 28793, 281
]
# fmt: on

_NORMAL_EXPECTED = Tokenized(tokens=_NORMAL_TOKENS)
_SYSTEM_SINGLETURN_EXPECTED = Tokenized(tokens=_SYSTEM_SINGLETURN_TOKENS)
_SYSTEM_MULTITURN_EXPECTED = Tokenized(tokens=_SYSTEM_MULTITURN_TOKENS)
_PREFIXED_FINAL_MESSAGE_EXPECTED = Tokenized(tokens=_PREFIXED_FINAL_MESSAGE_TOKENS, prefix_ids=[281])

_NORMAL_TEXT = "<s>▁[INST]▁a▁[/INST]▁b</s>▁[INST]▁c▁[/INST]▁d</s>"
_SYSTEM_SINGLETURN_TEXT = "<s>▁[INST]▁SYSTEM<0x0A><0x0A>a▁[/INST]"
_SYSTEM_MULTITURN_TEXT = "<s>▁[INST]▁SYSTEM<0x0A><0x0A>a▁[/INST]▁b</s>▁[INST]▁c▁[/INST]▁d</s>"
_PREFIXED_FINAL_MESSAGE_TEXT = "<s>▁[INST]▁SYSTEM<0x0A><0x0A>a▁[/INST]▁b</s>▁[INST]▁c▁[/INST]▁d"

_TOOLS_NOT_IMPLEMENTED_MESSAGE = r"^Tools not implemented for tokenizer V1$"


@pytest.fixture(scope="module")
def tokenizer() -> InstructTokenizer:
    r"""Provide the bundled v1 instruct tokenizer for formatting cases."""
    return MistralTokenizer.v1().instruct_tokenizer


def _build_normal_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the v1 multi-turn request without a system prompt."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ]
    )


def _build_system_singleturn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the v1 single-turn request with a system prompt."""
    return InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="a")], system_prompt="SYSTEM")


def _build_system_multiturn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the v1 multi-turn request with a system prompt."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ],
        system_prompt="SYSTEM",
    )


def _build_prefixed_final_message_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the v1 request with a prefixed final assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d", prefix=True),
        ],
        system_prompt="SYSTEM",
    )


def _build_tool_message_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v1 request containing a tool result."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), ToolMessage(content="d", tool_call_id="123456789")]
    )


def _build_assistant_tool_calls_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v1 request containing an assistant tool call."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="b", arguments="{}"))]),
        ]
    )


# The current " // None" message is a candidate defect and is deliberately not pinned.
def _build_empty_assistant_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v1 request containing an empty assistant message."""
    return InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="a"), AssistantMessage(content="")])


@pytest.mark.parametrize(
    argnames="request_builder, expected, expected_text",
    argvalues=[
        pytest.param(_build_normal_request, _NORMAL_EXPECTED, _NORMAL_TEXT, id="normal"),
        pytest.param(
            _build_system_singleturn_request,
            _SYSTEM_SINGLETURN_EXPECTED,
            _SYSTEM_SINGLETURN_TEXT,
            id="system-singleturn",
        ),
        pytest.param(
            _build_system_multiturn_request,
            _SYSTEM_MULTITURN_EXPECTED,
            _SYSTEM_MULTITURN_TEXT,
            id="system-multiturn",
        ),
        pytest.param(
            _build_prefixed_final_message_request,
            _PREFIXED_FINAL_MESSAGE_EXPECTED,
            _PREFIXED_FINAL_MESSAGE_TEXT,
            id="prefixed-final-message",
        ),
    ],
)
def test_encode_instruct(
    tokenizer: InstructTokenizer,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected: Tokenized,
    expected_text: str,
) -> None:
    r"""Preserve the complete v1 instruct encoding for each request shape."""
    tokenized = tokenizer.encode_instruct(request=request_builder())
    assert_tokenized_equal(actual=tokenized, expected=expected)
    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text


@pytest.mark.parametrize(
    argnames="request_builder, expected_exception, expected_message",
    argvalues=[
        pytest.param(
            _build_tool_message_request,
            TokenizerException,
            _TOOLS_NOT_IMPLEMENTED_MESSAGE,
            id="tool-message",
        ),
        pytest.param(
            _build_assistant_tool_calls_request,
            TokenizerException,
            _TOOLS_NOT_IMPLEMENTED_MESSAGE,
            id="assistant-tool-calls",
        ),
        pytest.param(_build_empty_assistant_request, TokenizerException, None, id="empty-assistant"),
    ],
)
def test_encode_instruct_rejected(
    tokenizer: InstructTokenizer,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected_exception: type[Exception],
    expected_message: str | None,
) -> None:
    r"""Preserve direct v1 exception types and specified messages."""
    request = request_builder()

    with pytest.raises(expected_exception=expected_exception, match=expected_message):
        tokenizer.encode_instruct(request=request)

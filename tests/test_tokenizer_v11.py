import pytest

from mistral_common.exceptions import InvalidAssistantMessageException
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
)
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV11
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import _quick_vocab, get_special_tokens


@pytest.fixture(scope="session")
def tekkenizer() -> InstructTokenizerV11:
    special_tokens = get_special_tokens(TokenizerVersion.v11)
    tokenizer = Tekkenizer(
        _quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v11,
    )
    return InstructTokenizerV11(tokenizer)


def test_special_tokens(tekkenizer: InstructTokenizerV11) -> None:
    assert tekkenizer.ARGS == 32
    assert tekkenizer.CALL_ID == 33
    assert tekkenizer.TOOL_CALLS == 9


def test_tokenize_assistant_message(tekkenizer: InstructTokenizerV11) -> None:
    tokens = tekkenizer.encode_assistant_message(
        AssistantMessage(
            tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"))],
        ),
        is_before_last_user_message=False,
        continue_message=False,
    )
    assert tokens == [
        tekkenizer.TOOL_CALLS,
        197,
        195,
        197,
        195,
        197,
        tekkenizer.ARGS,
        134,
        198,
        208,
        197,
        198,
        208,
        197,
        134,
        2,
    ]
    assert tekkenizer.tokenizer.to_string(tokens) == ('[TOOL_CALLS]a_a_a[ARGS]"blabla"</s>')


def test_tokenize_assistant_message_continue_message(tekkenizer: InstructTokenizerV11) -> None:
    tokens = tekkenizer.encode_assistant_message(
        AssistantMessage(
            content='"blabla"',
        ),
        is_before_last_user_message=False,
        continue_message=True,
    )
    assert tokens == [
        134,
        198,
        208,
        197,
        198,
        208,
        197,
        134,
    ]
    assert tekkenizer.tokenizer.to_string(tokens) == ('"blabla"')

    with pytest.raises(
        InvalidAssistantMessageException,
        match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
    ):
        tekkenizer.encode_assistant_message(
            AssistantMessage(
                content='"blabla"',
                prefix=True,
            ),
            is_before_last_user_message=False,
            continue_message=True,
        )


def test_tokenize_assistant_messages(tekkenizer: InstructTokenizerV11) -> None:
    tokens = tekkenizer.encode_assistant_message(
        AssistantMessage(
            tool_calls=[
                ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla")),
                ToolCall(function=FunctionCall(name="b", arguments="blu")),
            ],
        ),
        is_before_last_user_message=False,
        continue_message=False,
    )
    assert tokens == [
        tekkenizer.TOOL_CALLS,
        197,
        195,
        197,
        195,
        197,
        tekkenizer.ARGS,
        134,
        198,
        208,
        197,
        198,
        208,
        197,
        134,
        tekkenizer.TOOL_CALLS,
        198,
        tekkenizer.ARGS,
        134,
        198,
        208,
        217,
        134,
        2,
    ]
    assert tekkenizer.tokenizer.to_string(tokens) == ('[TOOL_CALLS]a_a_a[ARGS]"blabla"[TOOL_CALLS]b[ARGS]"blu"</s>')


def test_tokenize_assistant_message_train(tekkenizer: InstructTokenizerV11) -> None:
    tokens = tekkenizer.encode_assistant_message(
        AssistantMessage(
            tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"), id="ABC")],
        ),
        is_before_last_user_message=True,
        continue_message=False,
    )
    assert tokens == [
        tekkenizer.TOOL_CALLS,
        197,
        195,
        197,
        195,
        197,
        tekkenizer.CALL_ID,
        165,
        166,
        167,
        tekkenizer.ARGS,
        134,
        198,
        208,
        197,
        198,
        208,
        197,
        134,
        2,
    ]
    assert tekkenizer.tokenizer.to_string(tokens) == ('[TOOL_CALLS]a_a_a[CALL_ID]ABC[ARGS]"blabla"</s>')

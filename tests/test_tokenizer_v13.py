import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.chunk import (
    TextChunk,
    ThinkChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
)
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy, Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab


@pytest.fixture(scope="session")
def v13_tekkenizer() -> InstructTokenizerV13:
    special_tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v13, add_think=False)
    tokenizer = Tekkenizer(
        vocab=quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
    )
    return InstructTokenizerV13(tokenizer)


@pytest.fixture(scope="session")
def v13_tekkenizer_think() -> InstructTokenizerV13:
    special_tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v13, add_think=True)
    tokenizer = Tekkenizer(
        vocab=quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
    )
    return InstructTokenizerV13(tokenizer)


def test_encode_tool_message(v13_tekkenizer: InstructTokenizerV13) -> None:
    tool_message = ToolMessage(content="R1", tool_call_id="123456789")
    assert isinstance(v13_tekkenizer, InstructTokenizerV13)
    encoded, images, audios = v13_tekkenizer.encode_tool_message(
        message=tool_message, is_before_last_user_message=False
    )
    assert encoded == [7, 182, 149, 8]
    assert images == []
    assert audios == []

    tool_message = ToolMessage(content=[TextChunk(text="R1"), TextChunk(text="R2")], tool_call_id="123456789")
    assert isinstance(v13_tekkenizer, InstructTokenizerV13)
    encoded, images, audios = v13_tekkenizer.encode_tool_message(
        message=tool_message, is_before_last_user_message=False
    )
    assert encoded == [7, 182, 149, 182, 150, 8]
    assert images == []
    assert audios == []


def test_encode_think_chunk(v13_tekkenizer_think: InstructTokenizerV13) -> None:
    assert isinstance(v13_tekkenizer_think, InstructTokenizerV13)
    think_chunk = ThinkChunk(
        thinking="T1",
    )
    encoded = v13_tekkenizer_think.encode_think(think_chunk)
    assert (
        v13_tekkenizer_think.decode(tokens=encoded, special_token_policy=SpecialTokenPolicy.KEEP) == "[THINK]T1[/THINK]"
    )

    think_chunk = ThinkChunk(
        thinking="T1",
        closed=False,
    )
    encoded = v13_tekkenizer_think.encode_think(think_chunk)
    assert v13_tekkenizer_think.decode(tokens=encoded, special_token_policy=SpecialTokenPolicy.KEEP) == "[THINK]T1"


@pytest.mark.parametrize(
    argnames="message, expected",
    argvalues=[
        (
            AssistantMessage(content="A1"),
            "A1",
        ),
        (
            AssistantMessage(content="A1", prefix=True),
            "A1",
        ),
        (
            AssistantMessage(content=[TextChunk(text="A1")]),
            "A1",
        ),
        (
            AssistantMessage(content=[ThinkChunk(thinking="T1"), TextChunk(text="A1")]),
            "[THINK]T1[/THINK]A1",
        ),
        (
            AssistantMessage(
                content=[ThinkChunk(thinking="R1", closed=False), TextChunk(text="A1")],
                tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{'a': 1}"))],
            ),
            "[THINK]R1A1[TOOL_CALLS]F1[ARGS]\"{'a': 1}\"",
        ),
    ],
)
def test_tokenize_assistant_message(
    v13_tekkenizer_think: InstructTokenizerV13, message: AssistantMessage, expected: str
) -> None:
    tokens = v13_tekkenizer_think.encode_assistant_message(message=message, is_before_last_user_message=False)
    if not message.prefix:
        expected += "</s>"
    assert v13_tekkenizer_think.decode(tokens=tokens, special_token_policy=SpecialTokenPolicy.KEEP) == expected


def test_tokenize_assistant_message_error(v13_tekkenizer: InstructTokenizerV13) -> None:
    with pytest.raises(expected_exception=TokenizerException, match=r"Invalid assistant message"):
        v13_tekkenizer.encode_assistant_message(
            message=AssistantMessage(content="", tool_calls=[]), is_before_last_user_message=False
        )


@pytest.mark.parametrize(
    argnames="message, expected",
    argvalues=[
        (
            SystemMessage(content="S1"),
            "[SYSTEM_PROMPT]S1[/SYSTEM_PROMPT]",
        ),
        (
            SystemMessage(content=[TextChunk(text="S1"), ThinkChunk(thinking="TS"), TextChunk(text="S2")]),
            "[SYSTEM_PROMPT]S1[THINK]TS[/THINK]S2[/SYSTEM_PROMPT]",
        ),
        (
            SystemMessage(
                content=[
                    TextChunk(text="S1"),
                    TextChunk(text="S3"),
                    ThinkChunk(thinking="TS", closed=True),
                    ThinkChunk(thinking="TS", closed=True),
                    TextChunk(text="S2"),
                ]
            ),
            "[SYSTEM_PROMPT]S1S3[THINK]TS[/THINK][THINK]TS[/THINK]S2[/SYSTEM_PROMPT]",
        ),
        (
            SystemMessage(
                content=[
                    TextChunk(text="S1"),
                    TextChunk(text="S3"),
                    ThinkChunk(thinking="TS", closed=False),
                ]
            ),
            "[SYSTEM_PROMPT]S1S3[THINK]TS[/SYSTEM_PROMPT]",
        ),
    ],
)
def test_encode_system_message(
    v13_tekkenizer_think: InstructTokenizerV13, message: SystemMessage, expected: str
) -> None:
    encoded, audios = v13_tekkenizer_think.encode_system_message(message)
    assert v13_tekkenizer_think.decode(tokens=encoded, special_token_policy=SpecialTokenPolicy.KEEP) == expected
    assert audios == []

r"""Direct formatting tests for the v13 instruct tokenizer."""

from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.chunk import AudioChunk, ContentChunk, ImageChunk, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import SpecialTokens, Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.image import DATASET_MEAN, DATASET_STD, ImageConfig, ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.utils import assert_tokenized_equal, decode_keep

# fmt: off
_TOOL_RESULT_TEXT_TOKENS = [1, 3, 197, 4, 9, 170, 149, 32, 223, 225, 2, 7, 182, 149, 8]
_TOOL_RESULT_TEXT_CHUNKS_TOKENS = [1, 3, 197, 4, 9, 170, 149, 32, 223, 225, 2, 7, 182, 149, 182, 150, 8]
_CLOSED_THINK_TOKENS = [1, 3, 197, 4, 35, 184, 149, 36, 2]
_OPEN_THINK_TOKENS = [1, 3, 197, 4, 35, 184, 149, 2]
_ASSISTANT_TEXT_TOKENS = [1, 3, 197, 4, 165, 149, 2]
_PREFIXED_ASSISTANT_TEXT_TOKENS = [1, 3, 197, 4, 165, 149]
_PREFIXED_ASSISTANT_TEXT_PREFIX_IDS = [165, 149]
_CLOSED_THINK_THEN_TEXT_TOKENS = [1, 3, 197, 4, 35, 184, 149, 36, 165, 149, 2]
_OPEN_THINK_TEXT_AND_CALL_TOKENS = [
    1, 3, 197, 4, 35, 182, 149, 165, 149, 9, 170, 149, 32, 134, 223, 139, 197, 139, 158, 132, 149, 225, 134, 2
]
_SYSTEM_TEXT_TOKENS = [1, 17, 183, 149, 18, 3, 197, 4]
_SYSTEM_THINK_BETWEEN_TEXT_TOKENS = [1, 17, 183, 149, 35, 184, 183, 36, 183, 150, 18, 3, 197, 4]
_SYSTEM_TWO_CLOSED_THINKS_TOKENS = [
    1, 17, 183, 149, 183, 151, 35, 184, 183, 36, 35, 184, 183, 36, 183, 150, 18, 3, 197, 4
]
_SYSTEM_OPEN_THINK_LAST_TOKENS = [1, 17, 183, 149, 183, 151, 35, 184, 183, 18, 3, 197, 4]
_NON_ASCII_CALL_ARGUMENTS_TOKENS = [
    1, 3, 197, 4, 9, 202, 32, 223, 134, 220, 134, 158, 132, 134, 295, 269, 134, 225, 2
]
_TOOLS_BEFORE_FIRST_USER_TOKENS = [
    1, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144,
    132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132,
    134, 202, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134, 134,
    144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 193, 6,
    3, 197, 4, 198, 2, 3, 199, 4
]
_SYSTEM_THEN_USER_AUDIO_TOKENS = [1, 17, 204, 201, 208, 208, 211, 18, 3, 25, 24, 24, 4]
_TOOL_RESULT_IMAGE_TOKENS = [1, 3, 197, 4, 9, 170, 149, 32, 223, 225, 2, 7, 10, 10, 12, 10, 10, 13, 8]
# fmt: on

_RED_4X4_IMAGE = np.broadcast_to(
    array=(
        (
            np.array(object=(255, 0, 0), dtype=np.float32) / np.float32(255.0)
            - np.array(object=DATASET_MEAN, dtype=np.float32)
        )
        / np.array(object=DATASET_STD, dtype=np.float32)
    )[:, None, None],
    shape=(3, 4, 4),
)
_SILENT_AUDIO = Audio(audio_array=np.zeros(shape=2400, dtype=np.float32), sampling_rate=24_000, format="wav")

_TOOL_RESULT_TEXT_EXPECTED = Tokenized(tokens=_TOOL_RESULT_TEXT_TOKENS)
_TOOL_RESULT_TEXT_CHUNKS_EXPECTED = Tokenized(tokens=_TOOL_RESULT_TEXT_CHUNKS_TOKENS)
_CLOSED_THINK_EXPECTED = Tokenized(tokens=_CLOSED_THINK_TOKENS)
_OPEN_THINK_EXPECTED = Tokenized(tokens=_OPEN_THINK_TOKENS)
_ASSISTANT_TEXT_EXPECTED = Tokenized(tokens=_ASSISTANT_TEXT_TOKENS)
_PREFIXED_ASSISTANT_TEXT_EXPECTED = Tokenized(
    tokens=_PREFIXED_ASSISTANT_TEXT_TOKENS, prefix_ids=_PREFIXED_ASSISTANT_TEXT_PREFIX_IDS
)
_CLOSED_THINK_THEN_TEXT_EXPECTED = Tokenized(tokens=_CLOSED_THINK_THEN_TEXT_TOKENS)
_OPEN_THINK_TEXT_AND_CALL_EXPECTED = Tokenized(tokens=_OPEN_THINK_TEXT_AND_CALL_TOKENS)
_SYSTEM_TEXT_EXPECTED = Tokenized(tokens=_SYSTEM_TEXT_TOKENS)
_SYSTEM_THINK_BETWEEN_TEXT_EXPECTED = Tokenized(tokens=_SYSTEM_THINK_BETWEEN_TEXT_TOKENS)
_SYSTEM_TWO_CLOSED_THINKS_EXPECTED = Tokenized(tokens=_SYSTEM_TWO_CLOSED_THINKS_TOKENS)
_SYSTEM_OPEN_THINK_LAST_EXPECTED = Tokenized(tokens=_SYSTEM_OPEN_THINK_LAST_TOKENS)
_NON_ASCII_CALL_ARGUMENTS_EXPECTED = Tokenized(tokens=_NON_ASCII_CALL_ARGUMENTS_TOKENS)
_TOOLS_BEFORE_FIRST_USER_EXPECTED = Tokenized(tokens=_TOOLS_BEFORE_FIRST_USER_TOKENS)
_SYSTEM_THEN_USER_AUDIO_EXPECTED = Tokenized(tokens=_SYSTEM_THEN_USER_AUDIO_TOKENS, audios=[_SILENT_AUDIO])
_TOOL_RESULT_IMAGE_EXPECTED = Tokenized(tokens=_TOOL_RESULT_IMAGE_TOKENS, images=[_RED_4X4_IMAGE])

_TOOL_RESULT_TEXT_TEXT = "<s>[INST]a[/INST][TOOL_CALLS]F1[ARGS]{}</s>[TOOL_RESULTS]R1[/TOOL_RESULTS]"
_TOOL_RESULT_TEXT_CHUNKS_TEXT = "<s>[INST]a[/INST][TOOL_CALLS]F1[ARGS]{}</s>[TOOL_RESULTS]R1R2[/TOOL_RESULTS]"
_CLOSED_THINK_TEXT = "<s>[INST]a[/INST][THINK]T1[/THINK]</s>"
_OPEN_THINK_TEXT = "<s>[INST]a[/INST][THINK]T1</s>"
_ASSISTANT_TEXT_TEXT = "<s>[INST]a[/INST]A1</s>"
_PREFIXED_ASSISTANT_TEXT_TEXT = "<s>[INST]a[/INST]A1"
_CLOSED_THINK_THEN_TEXT_TEXT = "<s>[INST]a[/INST][THINK]T1[/THINK]A1</s>"
_OPEN_THINK_TEXT_AND_CALL_TEXT = "<s>[INST]a[/INST][THINK]R1A1[TOOL_CALLS]F1[ARGS]\"{'a': 1}\"</s>"
_SYSTEM_TEXT_TEXT = "<s>[SYSTEM_PROMPT]S1[/SYSTEM_PROMPT][INST]a[/INST]"
_SYSTEM_THINK_BETWEEN_TEXT_TEXT = "<s>[SYSTEM_PROMPT]S1[THINK]TS[/THINK]S2[/SYSTEM_PROMPT][INST]a[/INST]"
_SYSTEM_TWO_CLOSED_THINKS_TEXT = (
    "<s>[SYSTEM_PROMPT]S1S3[THINK]TS[/THINK][THINK]TS[/THINK]S2[/SYSTEM_PROMPT][INST]a[/INST]"
)
_SYSTEM_OPEN_THINK_LAST_TEXT = "<s>[SYSTEM_PROMPT]S1S3[THINK]TS[/SYSTEM_PROMPT][INST]a[/INST]"
_NON_ASCII_CALL_ARGUMENTS_TEXT = '<s>[INST]a[/INST][TOOL_CALLS]f[ARGS]{"x": "é"}</s>'
_TOOLS_BEFORE_FIRST_USER_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "f", "description": "", '
    '"parameters": {}}}][/AVAILABLE_TOOLS][INST]a[/INST]b</s>[INST]c[/INST]'
)
_SYSTEM_THEN_USER_AUDIO_TEXT = "<s>[SYSTEM_PROMPT]hello[/SYSTEM_PROMPT][INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_TOOL_RESULT_IMAGE_TEXT = (
    "<s>[INST]a[/INST][TOOL_CALLS]F1[ARGS]{}</s>[TOOL_RESULTS][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][/TOOL_RESULTS]"
)

_INVALID_ASSISTANT_MESSAGE = r"^Invalid assistant message:"


@pytest.fixture(scope="module")
def no_think_tokenizer() -> InstructTokenizerV13:
    r"""Synthetic formatting-only tokenizer; its IDs are not released-vocabulary evidence."""
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v13, add_think=False),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
    )
    return InstructTokenizerV13(tokenizer=tekkenizer)


@pytest.fixture(scope="module")
def with_think_tokenizer() -> InstructTokenizerV13:
    r"""Synthetic formatting-only tokenizer; its IDs are not released-vocabulary evidence."""
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v13, add_think=True),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
    )
    return InstructTokenizerV13(tokenizer=tekkenizer)


@pytest.fixture(scope="module")
def audio_tokenizer() -> InstructTokenizerV13:
    r"""Synthetic formatting-only tokenizer; its IDs are not released-vocabulary evidence."""
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v13, add_think=False, add_audio=True),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
    )
    return InstructTokenizerV13(
        tokenizer=tekkenizer,
        audio_encoder=AudioEncoder(
            audio_config=AudioConfig(
                sampling_rate=24_000,
                frame_rate=12.5,
                encoding_config=AudioSpectrogramConfig(num_mel_bins=128, window_size=400, hop_length=160),
            ),
            special_ids=SpecialAudioIDs(
                audio=tekkenizer.get_special_token(SpecialTokens.audio.value),
                begin_audio=tekkenizer.get_special_token(SpecialTokens.begin_audio.value),
                streaming_pad=None,
                text_to_audio=None,
                audio_to_text=None,
            ),
        ),
    )


@pytest.fixture(scope="module")
def image_tokenizer() -> InstructTokenizerV13:
    r"""Synthetic formatting-only tokenizer; its IDs are not released-vocabulary evidence."""
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v13, add_think=False),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v13,
    )
    return InstructTokenizerV13(
        tokenizer=tekkenizer,
        image_encoder=ImageEncoder(
            image_config=ImageConfig(image_patch_size=2, max_image_size=1024),
            special_ids=SpecialImageIDs(
                img=tekkenizer.get_special_token("[IMG]"),
                img_break=tekkenizer.get_special_token("[IMG_BREAK]"),
                img_end=tekkenizer.get_special_token("[IMG_END]"),
            ),
        ),
    )


def _build_user_then_assistant_request(
    assistant: AssistantMessage,
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by the supplied assistant message."""
    return InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="a"), assistant])


def _build_tool_result_request(content: str | list[ContentChunk]) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with one assistant call and its tool result."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{}"))]),
            ToolMessage(content=content, tool_call_id="123456789"),
        ]
    )


def _build_system_request(content: str | list[ContentChunk]) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message followed by the user turn."""
    return InstructRequest[ChatMessage, Tool](messages=[SystemMessage(content=content), UserMessage(content="a")])


def _build_tool_result_text_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose tool result is the text `R1`."""
    return _build_tool_result_request(content="R1")


def _build_tool_result_text_chunks_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose tool result has two text chunks."""
    return _build_tool_result_request(content=[TextChunk(text="R1"), TextChunk(text="R2")])


def _build_closed_think_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a closed assistant think chunk request."""
    return _build_user_then_assistant_request(assistant=AssistantMessage(content=[ThinkChunk(thinking="T1")]))


def _build_open_think_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build an open assistant think chunk request."""
    return _build_user_then_assistant_request(
        assistant=AssistantMessage(content=[ThinkChunk(thinking="T1", closed=False)])
    )


def _build_assistant_text_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build an assistant turn with plain text content."""
    return _build_user_then_assistant_request(assistant=AssistantMessage(content="A1"))


def _build_prefixed_assistant_text_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a prefixed assistant turn with plain text content."""
    return _build_user_then_assistant_request(assistant=AssistantMessage(content="A1", prefix=True))


def _build_assistant_text_chunk_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build an assistant turn containing one text chunk."""
    return _build_user_then_assistant_request(assistant=AssistantMessage(content=[TextChunk(text="A1")]))


def _build_closed_think_then_text_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a closed think chunk followed by assistant text."""
    return _build_user_then_assistant_request(
        assistant=AssistantMessage(content=[ThinkChunk(thinking="T1"), TextChunk(text="A1")])
    )


def _build_open_think_text_and_call_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build assistant content with an open think chunk, text and a tool call."""
    return _build_user_then_assistant_request(
        assistant=AssistantMessage(
            content=[ThinkChunk(thinking="R1", closed=False), TextChunk(text="A1")],
            tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{'a': 1}"))],
        )
    )


def _build_system_text_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with plain text content."""
    return _build_system_request(content="S1")


def _build_system_think_between_text_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with text on both sides of a think chunk."""
    return _build_system_request(content=[TextChunk(text="S1"), ThinkChunk(thinking="TS"), TextChunk(text="S2")])


def _build_system_two_closed_thinks_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with two closed think chunks between text."""
    return _build_system_request(
        content=[
            TextChunk(text="S1"),
            TextChunk(text="S3"),
            ThinkChunk(thinking="TS", closed=True),
            ThinkChunk(thinking="TS", closed=True),
            TextChunk(text="S2"),
        ]
    )


def _build_system_open_think_last_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message ending with an open think chunk."""
    return _build_system_request(
        content=[TextChunk(text="S1"), TextChunk(text="S3"), ThinkChunk(thinking="TS", closed=False)]
    )


def _build_non_ascii_call_arguments_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a tool call with an ID and non-ASCII JSON arguments."""
    return _build_user_then_assistant_request(
        assistant=AssistantMessage(
            tool_calls=[
                ToolCall(id="123456789", function=FunctionCall(name="f", arguments='{"x": "é"}')),
            ]
        )
    )


def _build_tools_before_first_user_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a tool listing and conversation with two user messages."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), AssistantMessage(content="b"), UserMessage(content="c")],
        available_tools=[Tool(function=Function(name="f", description="", parameters={}))],
    )


def _build_system_then_user_audio_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message followed by a user audio chunk."""
    audio = Audio(audio_array=np.zeros(shape=1600, dtype=np.float32), sampling_rate=16_000, format="wav")
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content="hello"), UserMessage(content=[AudioChunk.from_audio(audio=audio)])]
    )


def _build_tool_result_image_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a tool result containing one red image."""
    return _build_tool_result_request(content=[ImageChunk(image=Image.new(mode="RGB", size=(4, 4), color="red"))])


def _build_empty_assistant_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with an empty assistant message."""
    return _build_user_then_assistant_request(assistant=AssistantMessage(content="", tool_calls=[]))


@pytest.mark.parametrize(
    argnames="config, request_builder, expected, expected_text",
    argvalues=[
        pytest.param(
            "no_think",
            _build_tool_result_text_request,
            _TOOL_RESULT_TEXT_EXPECTED,
            _TOOL_RESULT_TEXT_TEXT,
            id="no-think-tool-result-text",
        ),
        pytest.param(
            "no_think",
            _build_tool_result_text_chunks_request,
            _TOOL_RESULT_TEXT_CHUNKS_EXPECTED,
            _TOOL_RESULT_TEXT_CHUNKS_TEXT,
            id="no-think-tool-result-text-chunks",
        ),
        pytest.param(
            "with_think",
            _build_closed_think_request,
            _CLOSED_THINK_EXPECTED,
            _CLOSED_THINK_TEXT,
            id="with-think-closed-think",
        ),
        pytest.param(
            "with_think", _build_open_think_request, _OPEN_THINK_EXPECTED, _OPEN_THINK_TEXT, id="with-think-open-think"
        ),
        pytest.param(
            "with_think",
            _build_assistant_text_request,
            _ASSISTANT_TEXT_EXPECTED,
            _ASSISTANT_TEXT_TEXT,
            id="with-think-assistant-text",
        ),
        pytest.param(
            "with_think",
            _build_prefixed_assistant_text_request,
            _PREFIXED_ASSISTANT_TEXT_EXPECTED,
            _PREFIXED_ASSISTANT_TEXT_TEXT,
            id="with-think-prefixed-assistant-text",
        ),
        pytest.param(
            "with_think",
            _build_assistant_text_chunk_request,
            _ASSISTANT_TEXT_EXPECTED,
            _ASSISTANT_TEXT_TEXT,
            id="with-think-assistant-text-chunk",
        ),
        pytest.param(
            "with_think",
            _build_closed_think_then_text_request,
            _CLOSED_THINK_THEN_TEXT_EXPECTED,
            _CLOSED_THINK_THEN_TEXT_TEXT,
            id="with-think-closed-think-then-text",
        ),
        pytest.param(
            "with_think",
            _build_open_think_text_and_call_request,
            _OPEN_THINK_TEXT_AND_CALL_EXPECTED,
            _OPEN_THINK_TEXT_AND_CALL_TEXT,
            id="with-think-open-think-text-and-call",
        ),
        pytest.param(
            "with_think",
            _build_system_text_request,
            _SYSTEM_TEXT_EXPECTED,
            _SYSTEM_TEXT_TEXT,
            id="with-think-system-text",
        ),
        pytest.param(
            "with_think",
            _build_system_think_between_text_request,
            _SYSTEM_THINK_BETWEEN_TEXT_EXPECTED,
            _SYSTEM_THINK_BETWEEN_TEXT_TEXT,
            id="with-think-system-think-between-text",
        ),
        pytest.param(
            "with_think",
            _build_system_two_closed_thinks_request,
            _SYSTEM_TWO_CLOSED_THINKS_EXPECTED,
            _SYSTEM_TWO_CLOSED_THINKS_TEXT,
            id="with-think-system-two-closed-thinks",
        ),
        pytest.param(
            "with_think",
            _build_system_open_think_last_request,
            _SYSTEM_OPEN_THINK_LAST_EXPECTED,
            _SYSTEM_OPEN_THINK_LAST_TEXT,
            id="with-think-system-open-think-last",
        ),
        pytest.param(
            "no_think",
            _build_non_ascii_call_arguments_request,
            _NON_ASCII_CALL_ARGUMENTS_EXPECTED,
            _NON_ASCII_CALL_ARGUMENTS_TEXT,
            id="no-think-non-ascii-call-arguments",
        ),
        pytest.param(
            "no_think",
            _build_tools_before_first_user_request,
            _TOOLS_BEFORE_FIRST_USER_EXPECTED,
            _TOOLS_BEFORE_FIRST_USER_TEXT,
            id="no-think-tools-before-first-user",
        ),
        pytest.param(
            "audio",
            _build_system_then_user_audio_request,
            _SYSTEM_THEN_USER_AUDIO_EXPECTED,
            _SYSTEM_THEN_USER_AUDIO_TEXT,
            id="audio-system-then-user-audio",
        ),
        pytest.param(
            "image",
            _build_tool_result_image_request,
            _TOOL_RESULT_IMAGE_EXPECTED,
            _TOOL_RESULT_IMAGE_TEXT,
            id="image-tool-result-image",
        ),
    ],
)
def test_encode_instruct(
    request: pytest.FixtureRequest,
    config: str,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected: Tokenized,
    expected_text: str,
) -> None:
    r"""Preserve complete v13 instruct encodings for each configuration and request."""
    tokenizer: InstructTokenizerV13 = request.getfixturevalue(argname=f"{config}_tokenizer")
    tokenized = tokenizer.encode_instruct(request=request_builder())
    assert_tokenized_equal(actual=tokenized, expected=expected)
    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text


@pytest.mark.parametrize(
    argnames="config, request_builder, expected_exception, expected_message",
    argvalues=[
        pytest.param(
            "no_think",
            _build_empty_assistant_request,
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="no-think-empty-assistant",
        ),
    ],
)
def test_encode_instruct_rejected(
    request: pytest.FixtureRequest,
    config: str,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected_exception: type[Exception],
    expected_message: str | None,
) -> None:
    r"""Preserve direct v13 exception types and specified messages."""
    tokenizer: InstructTokenizerV13 = request.getfixturevalue(argname=f"{config}_tokenizer")
    instruct_request = request_builder()

    with pytest.raises(expected_exception=expected_exception, match=expected_message):
        tokenizer.encode_instruct(request=instruct_request)

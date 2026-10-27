r"""Direct formatting tests for the v11 instruct tokenizer."""

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import SpecialTokens, Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.image import DATASET_MEAN, DATASET_STD, ImageConfig, ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV11
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.unit.tokens.tokenizers.instruct.request_builders import (
    build_arbitrary_tool_call_id_request,
    build_assistant_list_content_request,
    build_assistant_think_request,
    build_assistant_tool_calls_request,
    build_audio_user_request,
    build_empty_assistant_request,
    build_image_user_request,
    build_multi_audio_user_request,
    build_multi_image_user_request,
    build_multiturn_audio_request,
    build_multiturn_image_request,
    build_normal_request,
    build_system_and_user_words_request,
    build_system_audio_request,
    build_system_message_multiturn_request,
    build_system_message_prefixed_final_request,
    build_system_message_request,
    build_system_message_tools_multiturn_request,
    build_system_then_audio_user_request,
    build_text_then_audio_user_request,
    build_text_then_image_user_request,
    build_tool_message_request,
    build_tools_multiturn_request,
    build_tools_singleturn_request,
    build_truncation_last_user_at_limit_request,
    build_user_audio_url_request,
    build_user_image_url_request,
    build_user_then_assistant_request,
)
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

# fmt: off
_NORMAL_TOKENS = [1, 3, 197, 4, 198, 2, 3, 199, 4, 200, 2]
_TOOLS_SINGLETURN_TOKENS = [
    1, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144,
    132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134,
    216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132,
    134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225,
    193, 6, 3, 197, 4]
_TOOLS_MULTITURN_TOKENS = [
    1, 3, 197, 4, 198, 2, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211,
    210, 134, 144, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134,
    158, 132, 134, 216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210,
    134, 158, 132, 134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223,
    225, 225, 225, 144, 132, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210,
    134, 144, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158,
    132, 134, 216, 211, 211, 208, 150, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134,
    158, 132, 134, 150, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225,
    225, 225, 193, 6, 3, 199, 4, 200, 2]
_SYSTEM_MESSAGE_TOKENS = [1, 17, 215, 18, 3, 197, 4]
_SYSTEM_MESSAGE_MULTITURN_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 3, 197, 4, 198, 2, 3, 199, 4, 200, 2
]
_SYSTEM_MESSAGE_PREFIXED_FINAL_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 3, 197, 4, 198, 2, 3, 199, 4, 200
]
_SYSTEM_MESSAGE_TOOLS_MULTITURN_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 3, 197, 4, 198, 2, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132,
    134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134,
    158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 211, 211, 208, 149, 134, 144, 132, 134, 200,
    201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134, 149, 134, 144, 132, 134, 212, 197, 214,
    197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 193, 6, 3, 199, 4, 200, 2
]
_TOOL_MESSAGE_TOKENS = [1, 3, 197, 4, 7, 149, 150, 151, 152, 153, 154, 155, 156, 157, 19, 200, 8]
_ASSISTANT_TOOL_CALLS_TOKENS = [1, 3, 197, 4, 9, 198, 33, 149, 150, 151, 152, 153, 154, 155, 156, 157, 32, 223, 225, 2]
_ASSISTANT_LIST_CONTENT_TOKENS = [1, 3, 197, 4, 198, 2]
_SYSTEM_AND_USER_WORDS_TOKENS = [
    1, 17, 204, 201, 208, 208, 211, 132, 219, 211, 214, 208, 200, 18, 3, 203, 211, 211, 200, 132, 200, 197, 221, 4
]
_TRUNCATION_LAST_USER_AT_LIMIT_TOKENS = [1, 3, 197, 197, 197, 197, 197, 197, 197, 4]
_ARBITRARY_TOOL_CALL_ID_X_TOKENS = [1, 3, 197, 4, 9, 202, 33, 220, 32, 223, 225, 2, 7, 220, 19, 198, 8]
_ARBITRARY_TOOL_CALL_ID_CALL_ID_TOKENS = [
    1, 3, 197, 4, 9, 202, 33, 199, 197, 208, 208, 147, 205, 200, 145, 149, 32, 223, 225, 2, 7, 199, 197,
    208, 208, 147, 205, 200, 145, 149, 19, 198, 8
]
_SYSTEM_AUDIO_TOKENS = [
    1, 17, 183, 221, 215, 216, 201, 209, 132, 219, 205, 216, 204, 132, 199, 211, 210, 216, 201, 210, 216,
    25, 24, 24, 18, 3, 172, 201, 208, 208, 211, 4
]
_USER_AUDIO_URL_TOKENS = [
    1, 3, 172, 201, 214, 201, 132, 205, 215, 132, 199, 211, 210, 216, 201, 210, 216, 25, 24, 24, 4
]
_USER_IMAGE_URL_TOKENS = [
    1, 3, 10, 10, 12, 10, 10, 13, 172, 201, 214, 201, 132, 205, 215, 132, 199, 211, 210, 216, 201, 210, 216, 4
]
# fmt: on

_NORMAL_EXPECTED = Tokenized(tokens=_NORMAL_TOKENS)
_TOOLS_SINGLETURN_EXPECTED = Tokenized(tokens=_TOOLS_SINGLETURN_TOKENS)
_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_TOOLS_MULTITURN_TOKENS)
_SYSTEM_MESSAGE_EXPECTED = Tokenized(tokens=_SYSTEM_MESSAGE_TOKENS)
_SYSTEM_MESSAGE_MULTITURN_EXPECTED = Tokenized(tokens=_SYSTEM_MESSAGE_MULTITURN_TOKENS)
_SYSTEM_MESSAGE_PREFIXED_FINAL_EXPECTED = Tokenized(tokens=_SYSTEM_MESSAGE_PREFIXED_FINAL_TOKENS, prefix_ids=[200])
_SYSTEM_MESSAGE_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_SYSTEM_MESSAGE_TOOLS_MULTITURN_TOKENS)
_TOOL_MESSAGE_EXPECTED = Tokenized(tokens=_TOOL_MESSAGE_TOKENS)
_ASSISTANT_TOOL_CALLS_EXPECTED = Tokenized(tokens=_ASSISTANT_TOOL_CALLS_TOKENS)
_ASSISTANT_LIST_CONTENT_EXPECTED = Tokenized(tokens=_ASSISTANT_LIST_CONTENT_TOKENS)
_SYSTEM_AND_USER_WORDS_EXPECTED = Tokenized(tokens=_SYSTEM_AND_USER_WORDS_TOKENS)
_TRUNCATION_LAST_USER_AT_LIMIT_EXPECTED = Tokenized(tokens=_TRUNCATION_LAST_USER_AT_LIMIT_TOKENS)
_ARBITRARY_TOOL_CALL_ID_X_EXPECTED = Tokenized(tokens=_ARBITRARY_TOOL_CALL_ID_X_TOKENS)
_ARBITRARY_TOOL_CALL_ID_CALL_ID_EXPECTED = Tokenized(tokens=_ARBITRARY_TOOL_CALL_ID_CALL_ID_TOKENS)

_NORMAL_TEXT = "<s>[INST]a[/INST]b</s>[INST]c[/INST]d</s>"
_TOOLS_SINGLETURN_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}]'
    "[/AVAILABLE_TOOLS][INST]a[/INST]"
)
_TOOLS_MULTITURN_TEXT = (
    '<s>[INST]a[/INST]b</s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", '
    '"parameters": {}}}, {"type": "function", "function": {"name": "tool2", "description": "2", "parameters": {}}}]'
    "[/AVAILABLE_TOOLS][INST]c[/INST]d</s>"
)
_SYSTEM_MESSAGE_TEXT = "<s>[SYSTEM_PROMPT]s[/SYSTEM_PROMPT][INST]a[/INST]"
_SYSTEM_MESSAGE_MULTITURN_TEXT = "<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT][INST]a[/INST]b</s>[INST]c[/INST]d</s>"
_SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT = "<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT][INST]a[/INST]b</s>[INST]c[/INST]d"
_SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT = (
    "<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT][INST]a[/INST]b</s>"
    '[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}]'
    "[/AVAILABLE_TOOLS][INST]c[/INST]d</s>"
)
_TOOL_MESSAGE_TEXT = "<s>[INST]a[/INST][TOOL_RESULTS]123456789[TOOL_CONTENT]d[/TOOL_RESULTS]"
_ASSISTANT_TOOL_CALLS_TEXT = "<s>[INST]a[/INST][TOOL_CALLS]b[CALL_ID]123456789[ARGS]{}</s>"
_ASSISTANT_LIST_CONTENT_TEXT = "<s>[INST]a[/INST]b</s>"
_SYSTEM_AND_USER_WORDS_TEXT = "<s>[SYSTEM_PROMPT]hello world[/SYSTEM_PROMPT][INST]good day[/INST]"
_TRUNCATION_LAST_USER_AT_LIMIT_TEXT = "<s>[INST]aaaaaaa[/INST]"
_ARBITRARY_TOOL_CALL_ID_X_TEXT = (
    "<s>[INST]a[/INST][TOOL_CALLS]f[CALL_ID]x[ARGS]{}</s>[TOOL_RESULTS]x[TOOL_CONTENT]b[/TOOL_RESULTS]"
)
_ARBITRARY_TOOL_CALL_ID_CALL_ID_TEXT = (
    "<s>[INST]a[/INST][TOOL_CALLS]f[CALL_ID]call/id-1[ARGS]{}</s>[TOOL_RESULTS]call/id-1[TOOL_CONTENT]b[/TOOL_RESULTS]"
)
_SYSTEM_AUDIO_TEXT = (
    "<s>[SYSTEM_PROMPT]System with content[BEGIN_AUDIO][AUDIO][AUDIO][/SYSTEM_PROMPT][INST]Hello[/INST]"
)
_USER_AUDIO_URL_TEXT = "<s>[INST]Here is content[BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_USER_IMAGE_URL_TEXT = "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]Here is content[/INST]"

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
_BLUE_4X4_IMAGE = np.broadcast_to(
    array=(
        (
            np.array(object=(0, 0, 255), dtype=np.float32) / np.float32(255.0)
            - np.array(object=DATASET_MEAN, dtype=np.float32)
        )
        / np.array(object=DATASET_STD, dtype=np.float32)
    )[:, None, None],
    shape=(3, 4, 4),
)
_SILENT_AUDIO = Audio(audio_array=np.zeros(shape=2400, dtype=np.float32), sampling_rate=24_000, format="wav")

_IMAGE_USER_TOKENS = [1, 3, 10, 10, 12, 10, 10, 13, 197, 4]
_AUDIO_USER_TOKENS = [1, 3, 25, 24, 24, 4]
_TEXT_THEN_AUDIO_USER_TOKENS = [1, 3, 197, 25, 24, 24, 4]
_MULTI_IMAGE_USER_TOKENS = [1, 3, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 4]
_MULTITURN_IMAGE_TOKENS = [1, 3, 10, 10, 12, 10, 10, 13, 4, 198, 2, 3, 10, 10, 12, 10, 10, 13, 4, 200, 2]
_MULTI_AUDIO_USER_TOKENS = [1, 3, 25, 24, 24, 25, 24, 24, 4]
_MULTITURN_AUDIO_TOKENS = [1, 3, 25, 24, 24, 4, 198, 2, 3, 25, 24, 24, 4, 200, 2]

_IMAGE_USER_EXPECTED = Tokenized(tokens=_IMAGE_USER_TOKENS, images=[_RED_4X4_IMAGE])
_AUDIO_USER_EXPECTED = Tokenized(tokens=_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO])
_TEXT_THEN_AUDIO_USER_EXPECTED = Tokenized(tokens=_TEXT_THEN_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO])
_MULTI_IMAGE_USER_EXPECTED = Tokenized(tokens=_MULTI_IMAGE_USER_TOKENS, images=[_RED_4X4_IMAGE, _BLUE_4X4_IMAGE])
_MULTITURN_IMAGE_EXPECTED = Tokenized(tokens=_MULTITURN_IMAGE_TOKENS, images=[_RED_4X4_IMAGE, _RED_4X4_IMAGE])
_MULTI_AUDIO_USER_EXPECTED = Tokenized(tokens=_MULTI_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO, _SILENT_AUDIO])
_MULTITURN_AUDIO_EXPECTED = Tokenized(tokens=_MULTITURN_AUDIO_TOKENS, audios=[_SILENT_AUDIO, _SILENT_AUDIO])
_SYSTEM_AUDIO_EXPECTED = Tokenized(tokens=_SYSTEM_AUDIO_TOKENS, audios=[_SILENT_AUDIO])
_USER_AUDIO_URL_EXPECTED = Tokenized(tokens=_USER_AUDIO_URL_TOKENS, audios=[_SILENT_AUDIO])
_USER_IMAGE_URL_EXPECTED = Tokenized(tokens=_USER_IMAGE_URL_TOKENS, images=[_RED_4X4_IMAGE])

_IMAGE_USER_TEXT = "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]a[/INST]"
_AUDIO_USER_TEXT = "<s>[INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_TEXT_THEN_AUDIO_USER_TEXT = "<s>[INST]a[BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_MULTI_IMAGE_USER_TEXT = (
    "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][/INST]"
)
_MULTITURN_IMAGE_TEXT = (
    "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][/INST]b</s>"
    "[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][/INST]d</s>"
)
_MULTI_AUDIO_USER_TEXT = "<s>[INST][BEGIN_AUDIO][AUDIO][AUDIO][BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_MULTITURN_AUDIO_TEXT = "<s>[INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]b</s>[INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]d</s>"

_INVALID_ASSISTANT_MESSAGE = r"^Invalid assistant message:"
_THINK_NOT_IMPLEMENTED_MESSAGE = r"^Think not implemented for tokenizer < V13\.$"
_SYSTEM_AUDIO_NOT_ALLOWED_MESSAGE = r"^System messages are not yet allowed when audio is present$"


@pytest.fixture(scope="module")
def default_tokenizer() -> InstructTokenizerV11:
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


@pytest.fixture(scope="module")
def image_tokenizer() -> InstructTokenizerV11:
    r"""Synthetic v11 tokenizer with an image encoder; its IDs are not released-vocabulary evidence."""
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v11),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v11,
    )
    return InstructTokenizerV11(
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


@pytest.fixture(scope="module")
def audio_tokenizer() -> InstructTokenizerV11:
    r"""Synthetic v11 tokenizer with an audio encoder; its IDs are not released-vocabulary evidence."""
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v11, add_audio=True),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v11,
    )
    return InstructTokenizerV11(
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


def _build_one_call_without_id_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build an assistant tool call without an ID."""
    return build_user_then_assistant_request(
        assistant=AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"))]),
        settings=None,
    )


def _build_prefixed_text_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a prefixed assistant text turn."""
    return build_user_then_assistant_request(assistant=AssistantMessage(content='"blabla"', prefix=True), settings=None)


def _build_two_calls_without_id_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build an assistant turn with two tool calls without IDs."""
    return build_user_then_assistant_request(
        assistant=AssistantMessage(
            tool_calls=[
                ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla")),
                ToolCall(function=FunctionCall(name="b", arguments="blu")),
            ]
        ),
        settings=None,
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
    return build_user_then_assistant_request(
        assistant=AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="f", arguments='{"x": "é"}'))]),
        settings=None,
    )


@pytest.mark.parametrize(
    argnames="config, request_builder, expected, expected_text",
    argvalues=[
        pytest.param(
            "default",
            _build_one_call_without_id_request,
            _ONE_CALL_WITHOUT_ID_EXPECTED,
            _ONE_CALL_WITHOUT_ID_TEXT,
            id="default-one-call-without-id",
        ),
        pytest.param(
            "default",
            _build_prefixed_text_request,
            _PREFIXED_TEXT_EXPECTED,
            _PREFIXED_TEXT_TEXT,
            id="default-prefixed-text",
        ),
        pytest.param(
            "default",
            _build_two_calls_without_id_request,
            _TWO_CALLS_WITHOUT_ID_EXPECTED,
            _TWO_CALLS_WITHOUT_ID_TEXT,
            id="default-two-calls-without-id",
        ),
        pytest.param(
            "default",
            _build_call_with_id_before_last_user_request,
            _CALL_WITH_ID_BEFORE_LAST_USER_EXPECTED,
            _CALL_WITH_ID_BEFORE_LAST_USER_TEXT,
            id="default-call-with-id-before-last-user",
        ),
        pytest.param(
            "default",
            _build_non_ascii_call_arguments_request,
            _NON_ASCII_CALL_ARGUMENTS_EXPECTED,
            _NON_ASCII_CALL_ARGUMENTS_TEXT,
            id="default-non-ascii-call-arguments",
        ),
        pytest.param(
            "default",
            partial(build_normal_request, settings=None),
            _NORMAL_EXPECTED,
            _NORMAL_TEXT,
            id="default-shared-normal",
        ),
        pytest.param(
            "default",
            partial(build_tools_singleturn_request, settings=None),
            _TOOLS_SINGLETURN_EXPECTED,
            _TOOLS_SINGLETURN_TEXT,
            id="default-shared-tools-singleturn",
        ),
        pytest.param(
            "default",
            partial(build_tools_multiturn_request, settings=None),
            _TOOLS_MULTITURN_EXPECTED,
            _TOOLS_MULTITURN_TEXT,
            id="default-shared-tools-multiturn",
        ),
        pytest.param(
            "default",
            partial(build_system_message_request, settings=None),
            _SYSTEM_MESSAGE_EXPECTED,
            _SYSTEM_MESSAGE_TEXT,
            id="default-shared-system-message",
        ),
        pytest.param(
            "default",
            partial(build_system_message_multiturn_request, settings=None),
            _SYSTEM_MESSAGE_MULTITURN_EXPECTED,
            _SYSTEM_MESSAGE_MULTITURN_TEXT,
            id="default-shared-system-message-multiturn",
        ),
        pytest.param(
            "default",
            partial(build_system_message_prefixed_final_request, settings=None),
            _SYSTEM_MESSAGE_PREFIXED_FINAL_EXPECTED,
            _SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT,
            id="default-shared-system-message-prefixed-final",
        ),
        pytest.param(
            "default",
            partial(build_system_message_tools_multiturn_request, settings=None),
            _SYSTEM_MESSAGE_TOOLS_MULTITURN_EXPECTED,
            _SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT,
            id="default-shared-system-message-tools-multiturn",
        ),
        pytest.param(
            "default",
            partial(build_tool_message_request, settings=None),
            _TOOL_MESSAGE_EXPECTED,
            _TOOL_MESSAGE_TEXT,
            id="default-shared-tool-message",
        ),
        pytest.param(
            "default",
            partial(build_assistant_tool_calls_request, settings=None),
            _ASSISTANT_TOOL_CALLS_EXPECTED,
            _ASSISTANT_TOOL_CALLS_TEXT,
            id="default-shared-assistant-tool-calls",
        ),
        pytest.param(
            "default",
            partial(build_assistant_list_content_request, settings=None),
            _ASSISTANT_LIST_CONTENT_EXPECTED,
            _ASSISTANT_LIST_CONTENT_TEXT,
            id="default-shared-assistant-list-content",
        ),
        pytest.param(
            "default",
            partial(build_system_and_user_words_request, settings=None),
            _SYSTEM_AND_USER_WORDS_EXPECTED,
            _SYSTEM_AND_USER_WORDS_TEXT,
            id="default-shared-system-and-user-words",
        ),
        pytest.param(
            "default",
            partial(build_truncation_last_user_at_limit_request, settings=None),
            _TRUNCATION_LAST_USER_AT_LIMIT_EXPECTED,
            _TRUNCATION_LAST_USER_AT_LIMIT_TEXT,
            id="default-shared-truncation-last-user-at-limit",
        ),
        pytest.param(
            "image",
            partial(build_image_user_request, settings=None),
            _IMAGE_USER_EXPECTED,
            _IMAGE_USER_TEXT,
            id="image-shared-image-user",
        ),
        pytest.param(
            "image",
            partial(build_text_then_image_user_request, settings=None),
            _IMAGE_USER_EXPECTED,
            _IMAGE_USER_TEXT,
            id="image-shared-text-then-image-user",
        ),
        pytest.param(
            "audio",
            partial(build_audio_user_request, settings=None),
            _AUDIO_USER_EXPECTED,
            _AUDIO_USER_TEXT,
            id="audio-shared-audio-user",
        ),
        pytest.param(
            "audio",
            partial(build_text_then_audio_user_request, settings=None),
            _TEXT_THEN_AUDIO_USER_EXPECTED,
            _TEXT_THEN_AUDIO_USER_TEXT,
            id="audio-shared-text-then-audio-user",
        ),
        pytest.param(
            "image",
            partial(build_multi_image_user_request, settings=None),
            _MULTI_IMAGE_USER_EXPECTED,
            _MULTI_IMAGE_USER_TEXT,
            id="image-shared-multi-image-user",
        ),
        pytest.param(
            "image",
            partial(build_multiturn_image_request, settings=None),
            _MULTITURN_IMAGE_EXPECTED,
            _MULTITURN_IMAGE_TEXT,
            id="image-shared-multiturn-image",
        ),
        pytest.param(
            "audio",
            partial(build_multi_audio_user_request, settings=None),
            _MULTI_AUDIO_USER_EXPECTED,
            _MULTI_AUDIO_USER_TEXT,
            id="audio-shared-multi-audio-user",
        ),
        pytest.param(
            "audio",
            partial(build_multiturn_audio_request, settings=None),
            _MULTITURN_AUDIO_EXPECTED,
            _MULTITURN_AUDIO_TEXT,
            id="audio-shared-multiturn-audio",
        ),
        pytest.param(
            "default",
            partial(build_arbitrary_tool_call_id_request, "x", settings=None),
            _ARBITRARY_TOOL_CALL_ID_X_EXPECTED,
            _ARBITRARY_TOOL_CALL_ID_X_TEXT,
            id="default-shared-arbitrary-tool-call-id-x",
        ),
        pytest.param(
            "default",
            partial(build_arbitrary_tool_call_id_request, "call/id-1", settings=None),
            _ARBITRARY_TOOL_CALL_ID_CALL_ID_EXPECTED,
            _ARBITRARY_TOOL_CALL_ID_CALL_ID_TEXT,
            id="default-shared-arbitrary-tool-call-id-call-slash-id-1",
        ),
        pytest.param(
            "audio",
            partial(build_system_audio_request, settings=None),
            _SYSTEM_AUDIO_EXPECTED,
            _SYSTEM_AUDIO_TEXT,
            id="audio-shared-system-audio",
        ),
        pytest.param(
            "audio",
            partial(build_user_audio_url_request, settings=None),
            _USER_AUDIO_URL_EXPECTED,
            _USER_AUDIO_URL_TEXT,
            id="audio-shared-user-audio-url",
        ),
        pytest.param(
            "image",
            partial(build_user_image_url_request, settings=None),
            _USER_IMAGE_URL_EXPECTED,
            _USER_IMAGE_URL_TEXT,
            id="image-shared-user-image-url",
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
    r"""Preserve complete v11 instruct encodings for each configuration and request."""
    tokenizer: InstructTokenizerV11 = request.getfixturevalue(argname=f"{config}_tokenizer")
    tokenized = tokenizer.encode_instruct(request=request_builder())
    assert_tokenized_equal(actual=tokenized, expected=expected)
    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text


@pytest.mark.parametrize(
    argnames="config, request_builder, expected_exception, expected_message",
    argvalues=[
        pytest.param(
            "default",
            partial(build_assistant_think_request, settings=None),
            TokenizerException,
            _THINK_NOT_IMPLEMENTED_MESSAGE,
            id="default-shared-assistant-think",
        ),
        pytest.param(
            "default",
            partial(build_empty_assistant_request, settings=None),
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="default-shared-empty-assistant",
        ),
        pytest.param(
            "audio",
            partial(build_system_then_audio_user_request, settings=None),
            ValueError,
            _SYSTEM_AUDIO_NOT_ALLOWED_MESSAGE,
            id="audio-shared-system-then-audio-user-rejected",
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
    r"""Preserve direct v11 exception types and specified messages."""
    tokenizer: InstructTokenizerV11 = request.getfixturevalue(argname=f"{config}_tokenizer")
    instruct_request = request_builder()

    with pytest.raises(expected_exception=expected_exception, match=expected_message):
        tokenizer.encode_instruct(request=instruct_request)

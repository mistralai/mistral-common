r"""Direct formatting tests for the v15 instruct tokenizer."""

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from mistral_common.exceptions import InvalidRequestException, TokenizerException
from mistral_common.protocol.instruct.chunk import ContentChunk, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import ChatMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest, ModelSettings, ReasoningEffort
from mistral_common.protocol.instruct.tool_calls import Tool
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import SpecialTokens, Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.image import DATASET_MEAN, DATASET_STD, ImageConfig, ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV15
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.unit.tokens.tokenizers.instruct.request_builders import (
    build_arbitrary_tool_call_id_request,
    build_assistant_list_content_request,
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
    build_system_think_chunk_request,
    build_text_then_audio_user_request,
    build_text_then_image_user_request,
    build_tool_message_request,
    build_tools_multiturn_request,
    build_tools_singleturn_request,
    build_truncation_last_user_at_limit_request,
    build_user_audio_url_request,
    build_user_image_url_request,
)
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
_INVALID_ASSISTANT_MESSAGE = r"^Invalid assistant message:"
_REASONING_EFFORT_REQUIRED_MESSAGE = r"^reasoning_effort should be set for this model\.$"

_SETTINGS = ModelSettings(reasoning_effort=ReasoningEffort.high)
_SETTINGS_JSON = '{"reasoning_effort": "high"}'

# fmt: off
_SHARED_NORMAL_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4, 198, 2, 3, 199, 4, 200, 2]
_SHARED_TOOLS_SINGLETURN_TOKENS = [
    1, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144,
    132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134,
    216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132,
    134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225,
    193, 6, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158,
    132, 134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4]
_SHARED_TOOLS_MULTITURN_TOKENS = [
    1, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144,
    132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134,
    216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132,
    134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225,
    144, 132, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144, 132,
    134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216,
    211, 211, 208, 150, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134,
    150, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 193,
    6, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132, 134,
    204, 205, 203, 204, 134, 225, 38, 3, 197, 4, 198, 2, 3, 199, 4, 200, 2]
_SHARED_SYSTEM_MESSAGE_TOKENS = [
    1, 17, 215, 18, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216,
    134, 158, 132, 134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4]
_SHARED_SYSTEM_MESSAGE_MULTITURN_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201,
    202, 202, 211, 214, 216, 134, 158, 132, 134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4, 198, 2, 3, 199, 4,
    200, 2]
_SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201,
    202, 202, 211, 214, 216, 134, 158, 132, 134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4, 198, 2, 3, 199, 4,
    200]
_SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217,
    210, 199, 216, 205, 211, 210, 134, 144, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134,
    210, 197, 209, 201, 134, 158, 132, 134, 216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205,
    212, 216, 205, 211, 210, 134, 158, 132, 134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214,
    215, 134, 158, 132, 223, 225, 225, 225, 193, 6, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195,
    201, 202, 202, 211, 214, 216, 134, 158, 132, 134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4, 198, 2, 3, 199,
    4, 200, 2]
_SHARED_TOOL_MESSAGE_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4, 7, 200, 8]
_SHARED_ASSISTANT_TOOL_CALLS_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4, 9, 198, 32, 223, 225, 2]
_SHARED_ASSISTANT_LIST_CONTENT_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4, 198, 2]
_SHARED_SYSTEM_AND_USER_WORDS_TOKENS = [
    1, 17, 204, 201, 208, 208, 211, 132, 219, 211, 214, 208, 200, 18, 37, 223, 134, 214, 201, 197, 215, 211, 210,
    205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132, 134, 204, 205, 203, 204, 134, 225, 38, 3, 203,
    211, 211, 200, 132, 200, 197, 221, 4]
_SHARED_TRUNCATION_LAST_USER_AT_LIMIT_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 197, 197, 197, 197, 197, 197, 4]
_ARBITRARY_TOOL_CALL_ID_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 4, 9, 202, 32, 223, 225, 2, 7, 198, 8
]
_SYSTEM_AUDIO_TOKENS = [
    1, 17, 183, 221, 215, 216, 201, 209, 132, 219, 205, 216, 204, 132, 199, 211, 210, 216, 201, 210, 216,
    25, 24, 24, 18, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214,
    216, 134, 158, 132, 134, 204, 205, 203, 204, 134, 225, 38, 3, 172, 201, 208, 208, 211, 4
]
_USER_AUDIO_URL_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 172, 201, 214, 201, 132, 205, 215, 132, 199, 211, 210, 216, 201,
    210, 216, 25, 24, 24, 4
]
_USER_IMAGE_URL_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 10, 10, 12, 10, 10, 13, 172, 201, 214, 201, 132, 205, 215, 132,
    199, 211, 210, 216, 201, 210, 216, 4
]
_SYSTEM_THEN_AUDIO_USER_TOKENS = [
    1, 17, 204, 201, 208, 208, 211, 18, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201,
    202, 202, 211, 214, 216, 134, 158, 132, 134, 204, 205, 203, 204, 134, 225, 38, 3, 25, 24, 24, 4
]
# fmt: on

_SHARED_NORMAL_EXPECTED = Tokenized(tokens=_SHARED_NORMAL_TOKENS)
_SHARED_TOOLS_SINGLETURN_EXPECTED = Tokenized(tokens=_SHARED_TOOLS_SINGLETURN_TOKENS)
_SHARED_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_SHARED_TOOLS_MULTITURN_TOKENS)
_SHARED_SYSTEM_MESSAGE_EXPECTED = Tokenized(tokens=_SHARED_SYSTEM_MESSAGE_TOKENS)
_SHARED_SYSTEM_MESSAGE_MULTITURN_EXPECTED = Tokenized(tokens=_SHARED_SYSTEM_MESSAGE_MULTITURN_TOKENS)
_SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_EXPECTED = Tokenized(
    tokens=_SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_TOKENS, prefix_ids=[200]
)
_SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_TOKENS)
_SHARED_TOOL_MESSAGE_EXPECTED = Tokenized(tokens=_SHARED_TOOL_MESSAGE_TOKENS)
_SHARED_ASSISTANT_TOOL_CALLS_EXPECTED = Tokenized(tokens=_SHARED_ASSISTANT_TOOL_CALLS_TOKENS)
_SHARED_ASSISTANT_LIST_CONTENT_EXPECTED = Tokenized(tokens=_SHARED_ASSISTANT_LIST_CONTENT_TOKENS)
_SHARED_SYSTEM_AND_USER_WORDS_EXPECTED = Tokenized(tokens=_SHARED_SYSTEM_AND_USER_WORDS_TOKENS)
_SHARED_TRUNCATION_LAST_USER_AT_LIMIT_EXPECTED = Tokenized(tokens=_SHARED_TRUNCATION_LAST_USER_AT_LIMIT_TOKENS)

_SETTINGS_BLOCK = f"[MODEL_SETTINGS]{_SETTINGS_JSON}[/MODEL_SETTINGS]"
_SHARED_NORMAL_TEXT = f"<s>{_SETTINGS_BLOCK}[INST]a[/INST]b</s>[INST]c[/INST]d</s>"
_SHARED_TOOLS_SINGLETURN_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}]'
    f"[/AVAILABLE_TOOLS]{_SETTINGS_BLOCK}[INST]a[/INST]"
)
_SHARED_TOOLS_MULTITURN_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}, '
    '{"type": "function", "function": {"name": "tool2", "description": "2", "parameters": {}}}]'
    f"[/AVAILABLE_TOOLS]{_SETTINGS_BLOCK}[INST]a[/INST]b</s>[INST]c[/INST]d</s>"
)
_SHARED_SYSTEM_MESSAGE_TEXT = f"<s>[SYSTEM_PROMPT]s[/SYSTEM_PROMPT]{_SETTINGS_BLOCK}[INST]a[/INST]"
_SHARED_SYSTEM_MESSAGE_MULTITURN_TEXT = (
    f"<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT]{_SETTINGS_BLOCK}[INST]a[/INST]b</s>[INST]c[/INST]d</s>"
)
_SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT = (
    f"<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT]{_SETTINGS_BLOCK}[INST]a[/INST]b</s>[INST]c[/INST]d"
)
_SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT = (
    "<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT]"
    '[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}]'
    f"[/AVAILABLE_TOOLS]{_SETTINGS_BLOCK}[INST]a[/INST]b</s>[INST]c[/INST]d</s>"
)
_SHARED_TOOL_MESSAGE_TEXT = f"<s>{_SETTINGS_BLOCK}[INST]a[/INST][TOOL_RESULTS]d[/TOOL_RESULTS]"
_SHARED_ASSISTANT_TOOL_CALLS_TEXT = f"<s>{_SETTINGS_BLOCK}[INST]a[/INST][TOOL_CALLS]b[ARGS]{{}}</s>"
_SHARED_ASSISTANT_LIST_CONTENT_TEXT = f"<s>{_SETTINGS_BLOCK}[INST]a[/INST]b</s>"
_SHARED_SYSTEM_AND_USER_WORDS_TEXT = (
    f"<s>[SYSTEM_PROMPT]hello world[/SYSTEM_PROMPT]{_SETTINGS_BLOCK}[INST]good day[/INST]"
)
_SHARED_TRUNCATION_LAST_USER_AT_LIMIT_TEXT = f"<s>{_SETTINGS_BLOCK}[INST]aaaaaaa[/INST]"

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

# fmt: off
_IMAGE_USER_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 10, 10, 12, 10, 10, 13, 197, 4]
_AUDIO_USER_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 25, 24, 24, 4]
_TEXT_THEN_AUDIO_USER_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 197, 25, 24, 24, 4]
_MULTI_IMAGE_USER_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 4]
_MULTITURN_IMAGE_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 10, 10, 12, 10, 10, 13, 4, 198, 2, 3, 10, 10, 12, 10, 10, 13, 4, 200, 2]
_MULTI_AUDIO_USER_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 25, 24, 24, 25, 24, 24, 4]
_MULTITURN_AUDIO_TOKENS = [
    1, 37, 223, 134, 214, 201, 197, 215, 211, 210, 205, 210, 203, 195, 201, 202, 202, 211, 214, 216, 134, 158, 132,
    134, 204, 205, 203, 204, 134, 225, 38, 3, 25, 24, 24, 4, 198, 2, 3, 25, 24, 24, 4, 200, 2]
# fmt: on

_IMAGE_USER_EXPECTED = Tokenized(tokens=_IMAGE_USER_TOKENS, images=[_RED_4X4_IMAGE])
_AUDIO_USER_EXPECTED = Tokenized(tokens=_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO])
_TEXT_THEN_AUDIO_USER_EXPECTED = Tokenized(tokens=_TEXT_THEN_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO])
_MULTI_IMAGE_USER_EXPECTED = Tokenized(tokens=_MULTI_IMAGE_USER_TOKENS, images=[_RED_4X4_IMAGE, _BLUE_4X4_IMAGE])
_MULTITURN_IMAGE_EXPECTED = Tokenized(tokens=_MULTITURN_IMAGE_TOKENS, images=[_RED_4X4_IMAGE, _RED_4X4_IMAGE])
_MULTI_AUDIO_USER_EXPECTED = Tokenized(tokens=_MULTI_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO, _SILENT_AUDIO])
_MULTITURN_AUDIO_EXPECTED = Tokenized(tokens=_MULTITURN_AUDIO_TOKENS, audios=[_SILENT_AUDIO, _SILENT_AUDIO])
_ARBITRARY_TOOL_CALL_ID_EXPECTED = Tokenized(tokens=_ARBITRARY_TOOL_CALL_ID_TOKENS)
_SYSTEM_AUDIO_EXPECTED = Tokenized(tokens=_SYSTEM_AUDIO_TOKENS, audios=[_SILENT_AUDIO])
_USER_AUDIO_URL_EXPECTED = Tokenized(tokens=_USER_AUDIO_URL_TOKENS, audios=[_SILENT_AUDIO])
_USER_IMAGE_URL_EXPECTED = Tokenized(tokens=_USER_IMAGE_URL_TOKENS, images=[_RED_4X4_IMAGE])
_SYSTEM_THEN_AUDIO_USER_EXPECTED_V2 = Tokenized(tokens=_SYSTEM_THEN_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO])

_IMAGE_USER_TEXT = f"<s>{_SETTINGS_BLOCK}[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]a[/INST]"
_AUDIO_USER_TEXT = f"<s>{_SETTINGS_BLOCK}[INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_TEXT_THEN_AUDIO_USER_TEXT = f"<s>{_SETTINGS_BLOCK}[INST]a[BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_MULTI_IMAGE_USER_TEXT = (
    f"<s>{_SETTINGS_BLOCK}[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][/INST]"
)
_MULTITURN_IMAGE_TEXT = (
    f"<s>{_SETTINGS_BLOCK}[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][/INST]b</s>"
    f"[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][/INST]d</s>"
)
_MULTI_AUDIO_USER_TEXT = f"<s>{_SETTINGS_BLOCK}[INST][BEGIN_AUDIO][AUDIO][AUDIO][BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_MULTITURN_AUDIO_TEXT = (
    f"<s>{_SETTINGS_BLOCK}[INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]b</s>[INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]d</s>"
)
_ARBITRARY_TOOL_CALL_ID_TEXT = (
    f"<s>{_SETTINGS_BLOCK}[INST]a[/INST][TOOL_CALLS]f[ARGS]{{}}</s>[TOOL_RESULTS]b[/TOOL_RESULTS]"
)
_SYSTEM_AUDIO_TEXT = (
    f"<s>[SYSTEM_PROMPT]System with content[BEGIN_AUDIO][AUDIO][AUDIO][/SYSTEM_PROMPT]"
    f"{_SETTINGS_BLOCK}[INST]Hello[/INST]"
)
_USER_AUDIO_URL_TEXT = f"<s>{_SETTINGS_BLOCK}[INST]Here is content[BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_USER_IMAGE_URL_TEXT = f"<s>{_SETTINGS_BLOCK}[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]Here is content[/INST]"
_SYSTEM_THEN_AUDIO_USER_TEXT = (
    f"<s>[SYSTEM_PROMPT]hello[/SYSTEM_PROMPT]{_SETTINGS_BLOCK}[INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
)


@pytest.fixture(scope="module")
def default_tokenizer() -> InstructTokenizerV15:
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


@pytest.fixture(scope="module")
def image_tokenizer() -> InstructTokenizerV15:
    r"""Synthetic v15 tokenizer with an image encoder; its IDs are not released-vocabulary evidence."""
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
    return InstructTokenizerV15(
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
def audio_tokenizer() -> InstructTokenizerV15:
    r"""Synthetic v15 tokenizer with an audio encoder; its IDs are not released-vocabulary evidence."""
    model_settings_builder = ModelSettingsBuilder(
        reasoning_effort=EnumBuilder[ReasoningEffort](
            values=list(ReasoningEffort),
            accepts_none=True,
            default=list(ReasoningEffort)[0],
        )
    )
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v15, add_think=False, add_audio=True),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v15,
        model_settings_builder=model_settings_builder,
    )
    return InstructTokenizerV15(
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
    argnames="config, request_builder, expected, expected_text",
    argvalues=[
        pytest.param(
            "default",
            _build_system_text_request,
            _SYSTEM_TEXT_EXPECTED,
            _SYSTEM_TEXT_TEXT,
            id="default-system-text",
        ),
        pytest.param(
            "default",
            _build_system_text_chunk_request,
            _SYSTEM_TEXT_EXPECTED,
            _SYSTEM_TEXT_TEXT,
            id="default-system-text-chunk",
        ),
        pytest.param(
            "default",
            partial(build_normal_request, settings=_SETTINGS),
            _SHARED_NORMAL_EXPECTED,
            _SHARED_NORMAL_TEXT,
            id="default-shared-normal",
        ),
        pytest.param(
            "default",
            partial(build_tools_singleturn_request, settings=_SETTINGS),
            _SHARED_TOOLS_SINGLETURN_EXPECTED,
            _SHARED_TOOLS_SINGLETURN_TEXT,
            id="default-shared-tools-singleturn",
        ),
        pytest.param(
            "default",
            partial(build_tools_multiturn_request, settings=_SETTINGS),
            _SHARED_TOOLS_MULTITURN_EXPECTED,
            _SHARED_TOOLS_MULTITURN_TEXT,
            id="default-shared-tools-multiturn",
        ),
        pytest.param(
            "default",
            partial(build_system_message_request, settings=_SETTINGS),
            _SHARED_SYSTEM_MESSAGE_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_TEXT,
            id="default-shared-system-message",
        ),
        pytest.param(
            "default",
            partial(build_system_message_multiturn_request, settings=_SETTINGS),
            _SHARED_SYSTEM_MESSAGE_MULTITURN_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_MULTITURN_TEXT,
            id="default-shared-system-message-multiturn",
        ),
        pytest.param(
            "default",
            partial(build_system_message_prefixed_final_request, settings=_SETTINGS),
            _SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT,
            id="default-shared-system-message-prefixed-final",
        ),
        pytest.param(
            "default",
            partial(build_system_message_tools_multiturn_request, settings=_SETTINGS),
            _SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT,
            id="default-shared-system-message-tools-multiturn",
        ),
        pytest.param(
            "default",
            partial(build_tool_message_request, settings=_SETTINGS),
            _SHARED_TOOL_MESSAGE_EXPECTED,
            _SHARED_TOOL_MESSAGE_TEXT,
            id="default-shared-tool-message",
        ),
        pytest.param(
            "default",
            partial(build_assistant_tool_calls_request, settings=_SETTINGS),
            _SHARED_ASSISTANT_TOOL_CALLS_EXPECTED,
            _SHARED_ASSISTANT_TOOL_CALLS_TEXT,
            id="default-shared-assistant-tool-calls",
        ),
        pytest.param(
            "default",
            partial(build_assistant_list_content_request, settings=_SETTINGS),
            _SHARED_ASSISTANT_LIST_CONTENT_EXPECTED,
            _SHARED_ASSISTANT_LIST_CONTENT_TEXT,
            id="default-shared-assistant-list-content",
        ),
        pytest.param(
            "default",
            partial(build_system_and_user_words_request, settings=_SETTINGS),
            _SHARED_SYSTEM_AND_USER_WORDS_EXPECTED,
            _SHARED_SYSTEM_AND_USER_WORDS_TEXT,
            id="default-shared-system-and-user-words",
        ),
        pytest.param(
            "default",
            partial(build_truncation_last_user_at_limit_request, settings=_SETTINGS, truncate_at_max_tokens=39),
            _SHARED_TRUNCATION_LAST_USER_AT_LIMIT_EXPECTED,
            _SHARED_TRUNCATION_LAST_USER_AT_LIMIT_TEXT,
            id="default-shared-truncation-last-user-at-limit",
        ),
        pytest.param(
            "image",
            partial(build_image_user_request, settings=_SETTINGS),
            _IMAGE_USER_EXPECTED,
            _IMAGE_USER_TEXT,
            id="image-shared-image-user",
        ),
        pytest.param(
            "image",
            partial(build_text_then_image_user_request, settings=_SETTINGS),
            _IMAGE_USER_EXPECTED,
            _IMAGE_USER_TEXT,
            id="image-shared-text-then-image-user",
        ),
        pytest.param(
            "audio",
            partial(build_audio_user_request, settings=_SETTINGS),
            _AUDIO_USER_EXPECTED,
            _AUDIO_USER_TEXT,
            id="audio-shared-audio-user",
        ),
        pytest.param(
            "audio",
            partial(build_text_then_audio_user_request, settings=_SETTINGS),
            _TEXT_THEN_AUDIO_USER_EXPECTED,
            _TEXT_THEN_AUDIO_USER_TEXT,
            id="audio-shared-text-then-audio-user",
        ),
        pytest.param(
            "image",
            partial(build_multi_image_user_request, settings=_SETTINGS),
            _MULTI_IMAGE_USER_EXPECTED,
            _MULTI_IMAGE_USER_TEXT,
            id="image-shared-multi-image-user",
        ),
        pytest.param(
            "image",
            partial(build_multiturn_image_request, settings=_SETTINGS),
            _MULTITURN_IMAGE_EXPECTED,
            _MULTITURN_IMAGE_TEXT,
            id="image-shared-multiturn-image",
        ),
        pytest.param(
            "audio",
            partial(build_multi_audio_user_request, settings=_SETTINGS),
            _MULTI_AUDIO_USER_EXPECTED,
            _MULTI_AUDIO_USER_TEXT,
            id="audio-shared-multi-audio-user",
        ),
        pytest.param(
            "audio",
            partial(build_multiturn_audio_request, settings=_SETTINGS),
            _MULTITURN_AUDIO_EXPECTED,
            _MULTITURN_AUDIO_TEXT,
            id="audio-shared-multiturn-audio",
        ),
        pytest.param(
            "default",
            partial(build_arbitrary_tool_call_id_request, "x", settings=_SETTINGS),
            _ARBITRARY_TOOL_CALL_ID_EXPECTED,
            _ARBITRARY_TOOL_CALL_ID_TEXT,
            id="default-shared-arbitrary-tool-call-id",
        ),
        pytest.param(
            "audio",
            partial(build_system_audio_request, settings=_SETTINGS),
            _SYSTEM_AUDIO_EXPECTED,
            _SYSTEM_AUDIO_TEXT,
            id="audio-shared-system-audio",
        ),
        pytest.param(
            "audio",
            partial(build_user_audio_url_request, settings=_SETTINGS),
            _USER_AUDIO_URL_EXPECTED,
            _USER_AUDIO_URL_TEXT,
            id="audio-shared-user-audio-url",
        ),
        pytest.param(
            "image",
            partial(build_user_image_url_request, settings=_SETTINGS),
            _USER_IMAGE_URL_EXPECTED,
            _USER_IMAGE_URL_TEXT,
            id="image-shared-user-image-url",
        ),
        pytest.param(
            "audio",
            partial(build_system_then_audio_user_request, settings=_SETTINGS),
            _SYSTEM_THEN_AUDIO_USER_EXPECTED_V2,
            _SYSTEM_THEN_AUDIO_USER_TEXT,
            id="audio-shared-system-then-audio-user",
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
    r"""Preserve complete v15 encodings for each configuration and request."""
    tokenizer: InstructTokenizerV15 = request.getfixturevalue(argname=f"{config}_tokenizer")
    tokenized = tokenizer.encode_instruct(request=request_builder())
    assert_tokenized_equal(actual=tokenized, expected=expected)
    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text


@pytest.mark.parametrize(
    argnames="config, request_builder, expected_exception, expected_message",
    argvalues=[
        pytest.param(
            "default",
            _build_system_think_request,
            TokenizerException,
            _SYSTEM_THINK_MESSAGE,
            id="default-system-think",
        ),
        pytest.param(
            "default",
            _build_system_text_then_think_request,
            TokenizerException,
            _SYSTEM_THINK_MESSAGE,
            id="default-system-text-then-think",
        ),
        pytest.param(
            "default",
            partial(build_empty_assistant_request, settings=_SETTINGS),
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="default-shared-empty-assistant",
        ),
        pytest.param(
            "default",
            partial(build_normal_request, settings=None),
            InvalidRequestException,
            _REASONING_EFFORT_REQUIRED_MESSAGE,
            id="default-shared-normal-no-settings",
        ),
        pytest.param(
            "default",
            partial(build_system_think_chunk_request, settings=_SETTINGS),
            TokenizerException,
            _SYSTEM_THINK_MESSAGE,
            id="default-shared-system-think-chunk",
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
    r"""Preserve direct v15 exception types and specified messages."""
    tokenizer: InstructTokenizerV15 = request.getfixturevalue(argname=f"{config}_tokenizer")
    instruct_request = request_builder()

    with pytest.raises(expected_exception=expected_exception, match=expected_message):
        tokenizer.encode_instruct(request=instruct_request)

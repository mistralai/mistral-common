r"""Direct formatting tests for the v13 instruct tokenizer."""

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.messages import (
    ChatMessage,
)
from mistral_common.protocol.instruct.request import InstructRequest
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
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.unit.tokens.tokenizers.instruct.request_builders import (
    build_arbitrary_tool_call_id_request,
    build_assistant_list_content_request,
    build_assistant_text_chunk_request,
    build_assistant_text_request,
    build_assistant_tool_calls_request,
    build_audio_user_request,
    build_closed_think_request,
    build_closed_think_then_text_request,
    build_empty_assistant_request,
    build_image_user_request,
    build_multi_audio_user_request,
    build_multi_image_user_request,
    build_multiturn_audio_request,
    build_multiturn_image_request,
    build_non_ascii_call_arguments_request,
    build_normal_request,
    build_open_think_request,
    build_open_think_text_and_call_request,
    build_prefixed_text_request,
    build_system_and_user_words_request,
    build_system_audio_request,
    build_system_message_multiturn_request,
    build_system_message_prefixed_final_request,
    build_system_message_request,
    build_system_message_tools_multiturn_request,
    build_system_open_think_last_request,
    build_system_text_request,
    build_system_then_audio_user_request,
    build_system_think_between_text_request,
    build_system_two_closed_thinks_request,
    build_text_then_audio_user_request,
    build_text_then_image_user_request,
    build_tool_message_request,
    build_tool_result_image_request,
    build_tool_result_text_chunks_request,
    build_tool_result_text_request,
    build_tools_before_first_user_request,
    build_tools_multiturn_request,
    build_tools_singleturn_request,
    build_truncation_last_user_at_limit_request,
    build_user_audio_url_request,
    build_user_image_url_request,
)
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
_ARBITRARY_TOOL_CALL_ID_TOKENS = [1, 3, 197, 4, 9, 202, 32, 223, 225, 2, 7, 198, 8]
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
_SYSTEM_THEN_AUDIO_USER_TOKENS = [1, 17, 204, 201, 208, 208, 211, 18, 3, 25, 24, 24, 4]
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
_ARBITRARY_TOOL_CALL_ID_EXPECTED = Tokenized(tokens=_ARBITRARY_TOOL_CALL_ID_TOKENS)
_SYSTEM_AUDIO_EXPECTED = Tokenized(tokens=_SYSTEM_AUDIO_TOKENS, audios=[_SILENT_AUDIO])
_USER_AUDIO_URL_EXPECTED = Tokenized(tokens=_USER_AUDIO_URL_TOKENS, audios=[_SILENT_AUDIO])
_USER_IMAGE_URL_EXPECTED = Tokenized(tokens=_USER_IMAGE_URL_TOKENS, images=[_RED_4X4_IMAGE])
_SYSTEM_THEN_AUDIO_USER_EXPECTED_V2 = Tokenized(tokens=_SYSTEM_THEN_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO])
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

# fmt: off
_SHARED_NORMAL_TOKENS = [1, 3, 197, 4, 198, 2, 3, 199, 4, 200, 2]
_SHARED_TOOLS_SINGLETURN_TOKENS = [
    1, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144,
    132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134,
    216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132,
    134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225,
    193, 6, 3, 197, 4]
_SHARED_TOOLS_MULTITURN_TOKENS = [
    1, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144,
    132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134,
    216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132,
    134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225,
    144, 132, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144, 132,
    134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216,
    211, 211, 208, 150, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134,
    150, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 193,
    6, 3, 197, 4, 198, 2, 3, 199, 4, 200, 2]
_SHARED_SYSTEM_MESSAGE_TOKENS = [1, 17, 215, 18, 3, 197, 4]
_SHARED_SYSTEM_MESSAGE_MULTITURN_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 3, 197, 4, 198, 2, 3, 199, 4, 200, 2
]
_SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 3, 197, 4, 198, 2, 3, 199, 4, 200
]
_SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217,
    210, 199, 216, 205, 211, 210, 134, 144, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223,
    134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201, 215, 199,
    214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201,
    216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 193, 6, 3, 197, 4, 198, 2, 3, 199, 4, 200, 2
]
_SHARED_TOOL_MESSAGE_TOKENS = [1, 3, 197, 4, 7, 200, 8]
_SHARED_ASSISTANT_TOOL_CALLS_TOKENS = [1, 3, 197, 4, 9, 198, 32, 223, 225, 2]
_SHARED_ASSISTANT_LIST_CONTENT_TOKENS = [1, 3, 197, 4, 198, 2]
_SHARED_SYSTEM_AND_USER_WORDS_TOKENS = [
    1, 17, 204, 201, 208, 208, 211, 132, 219, 211, 214, 208, 200, 18, 3, 203, 211, 211, 200, 132, 200, 197, 221, 4
]
_SHARED_TRUNCATION_LAST_USER_AT_LIMIT_TOKENS = [1, 3, 197, 197, 197, 197, 197, 197, 197, 4]
_IMAGE_USER_TOKENS = [1, 3, 10, 10, 12, 10, 10, 13, 197, 4]
_AUDIO_USER_TOKENS = [1, 3, 25, 24, 24, 4]
_TEXT_THEN_AUDIO_USER_TOKENS = [1, 3, 197, 25, 24, 24, 4]
_MULTI_IMAGE_USER_TOKENS = [1, 3, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 4]
_MULTITURN_IMAGE_TOKENS = [1, 3, 10, 10, 12, 10, 10, 13, 4, 198, 2, 3, 10, 10, 12, 10, 10, 13, 4, 200, 2]
_MULTI_AUDIO_USER_TOKENS = [1, 3, 25, 24, 24, 25, 24, 24, 4]
_MULTITURN_AUDIO_TOKENS = [1, 3, 25, 24, 24, 4, 198, 2, 3, 25, 24, 24, 4, 200, 2]
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
_IMAGE_USER_EXPECTED = Tokenized(tokens=_IMAGE_USER_TOKENS, images=[_RED_4X4_IMAGE])
_AUDIO_USER_EXPECTED = Tokenized(tokens=_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO])
_TEXT_THEN_AUDIO_USER_EXPECTED = Tokenized(tokens=_TEXT_THEN_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO])
_MULTI_IMAGE_USER_EXPECTED = Tokenized(tokens=_MULTI_IMAGE_USER_TOKENS, images=[_RED_4X4_IMAGE, _BLUE_4X4_IMAGE])
_MULTITURN_IMAGE_EXPECTED = Tokenized(tokens=_MULTITURN_IMAGE_TOKENS, images=[_RED_4X4_IMAGE, _RED_4X4_IMAGE])
_MULTI_AUDIO_USER_EXPECTED = Tokenized(tokens=_MULTI_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO, _SILENT_AUDIO])
_MULTITURN_AUDIO_EXPECTED = Tokenized(tokens=_MULTITURN_AUDIO_TOKENS, audios=[_SILENT_AUDIO, _SILENT_AUDIO])

_SHARED_NORMAL_TEXT = "<s>[INST]a[/INST]b</s>[INST]c[/INST]d</s>"
_SHARED_TOOLS_SINGLETURN_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}]'
    "[/AVAILABLE_TOOLS][INST]a[/INST]"
)
_SHARED_TOOLS_MULTITURN_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}, '
    '{"type": "function", "function": {"name": "tool2", "description": "2", "parameters": {}}}]'
    "[/AVAILABLE_TOOLS][INST]a[/INST]b</s>[INST]c[/INST]d</s>"
)
_SHARED_SYSTEM_MESSAGE_TEXT = "<s>[SYSTEM_PROMPT]s[/SYSTEM_PROMPT][INST]a[/INST]"
_SHARED_SYSTEM_MESSAGE_MULTITURN_TEXT = "<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT][INST]a[/INST]b</s>[INST]c[/INST]d</s>"
_SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT = (
    "<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT][INST]a[/INST]b</s>[INST]c[/INST]d"
)
_SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT = (
    "<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT]"
    '[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}]'
    "[/AVAILABLE_TOOLS][INST]a[/INST]b</s>[INST]c[/INST]d</s>"
)
_SHARED_TOOL_MESSAGE_TEXT = "<s>[INST]a[/INST][TOOL_RESULTS]d[/TOOL_RESULTS]"
_SHARED_ASSISTANT_TOOL_CALLS_TEXT = "<s>[INST]a[/INST][TOOL_CALLS]b[ARGS]{}</s>"
_SHARED_ASSISTANT_LIST_CONTENT_TEXT = "<s>[INST]a[/INST]b</s>"
_SHARED_SYSTEM_AND_USER_WORDS_TEXT = "<s>[SYSTEM_PROMPT]hello world[/SYSTEM_PROMPT][INST]good day[/INST]"
_SHARED_TRUNCATION_LAST_USER_AT_LIMIT_TEXT = "<s>[INST]aaaaaaa[/INST]"
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
_ARBITRARY_TOOL_CALL_ID_TEXT = "<s>[INST]a[/INST][TOOL_CALLS]f[ARGS]{}</s>[TOOL_RESULTS]b[/TOOL_RESULTS]"
_SYSTEM_AUDIO_TEXT = (
    "<s>[SYSTEM_PROMPT]System with content[BEGIN_AUDIO][AUDIO][AUDIO][/SYSTEM_PROMPT][INST]Hello[/INST]"
)
_USER_AUDIO_URL_TEXT = "<s>[INST]Here is content[BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_USER_IMAGE_URL_TEXT = "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]Here is content[/INST]"
_SYSTEM_THEN_AUDIO_USER_TEXT = "<s>[SYSTEM_PROMPT]hello[/SYSTEM_PROMPT][INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]"


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


@pytest.mark.parametrize(
    argnames="config, request_builder, expected, expected_text",
    argvalues=[
        pytest.param(
            "no_think",
            partial(build_tool_result_text_request, settings=None),
            _TOOL_RESULT_TEXT_EXPECTED,
            _TOOL_RESULT_TEXT_TEXT,
            id="no-think-tool-result-text",
        ),
        pytest.param(
            "no_think",
            partial(build_tool_result_text_chunks_request, settings=None),
            _TOOL_RESULT_TEXT_CHUNKS_EXPECTED,
            _TOOL_RESULT_TEXT_CHUNKS_TEXT,
            id="no-think-tool-result-text-chunks",
        ),
        pytest.param(
            "with_think",
            partial(build_closed_think_request, settings=None),
            _CLOSED_THINK_EXPECTED,
            _CLOSED_THINK_TEXT,
            id="with-think-closed-think",
        ),
        pytest.param(
            "with_think",
            partial(build_open_think_request, settings=None),
            _OPEN_THINK_EXPECTED,
            _OPEN_THINK_TEXT,
            id="with-think-open-think",
        ),
        pytest.param(
            "with_think",
            partial(build_assistant_text_request, settings=None),
            _ASSISTANT_TEXT_EXPECTED,
            _ASSISTANT_TEXT_TEXT,
            id="with-think-assistant-text",
        ),
        pytest.param(
            "with_think",
            partial(build_prefixed_text_request, "A1", settings=None),
            _PREFIXED_ASSISTANT_TEXT_EXPECTED,
            _PREFIXED_ASSISTANT_TEXT_TEXT,
            id="with-think-prefixed-assistant-text",
        ),
        pytest.param(
            "with_think",
            partial(build_assistant_text_chunk_request, settings=None),
            _ASSISTANT_TEXT_EXPECTED,
            _ASSISTANT_TEXT_TEXT,
            id="with-think-assistant-text-chunk",
        ),
        pytest.param(
            "with_think",
            partial(build_closed_think_then_text_request, settings=None),
            _CLOSED_THINK_THEN_TEXT_EXPECTED,
            _CLOSED_THINK_THEN_TEXT_TEXT,
            id="with-think-closed-think-then-text",
        ),
        pytest.param(
            "with_think",
            partial(build_open_think_text_and_call_request, settings=None),
            _OPEN_THINK_TEXT_AND_CALL_EXPECTED,
            _OPEN_THINK_TEXT_AND_CALL_TEXT,
            id="with-think-open-think-text-and-call",
        ),
        pytest.param(
            "with_think",
            partial(build_system_text_request, settings=None),
            _SYSTEM_TEXT_EXPECTED,
            _SYSTEM_TEXT_TEXT,
            id="with-think-system-text",
        ),
        pytest.param(
            "with_think",
            partial(build_system_think_between_text_request, settings=None),
            _SYSTEM_THINK_BETWEEN_TEXT_EXPECTED,
            _SYSTEM_THINK_BETWEEN_TEXT_TEXT,
            id="with-think-system-think-between-text",
        ),
        pytest.param(
            "with_think",
            partial(build_system_two_closed_thinks_request, settings=None),
            _SYSTEM_TWO_CLOSED_THINKS_EXPECTED,
            _SYSTEM_TWO_CLOSED_THINKS_TEXT,
            id="with-think-system-two-closed-thinks",
        ),
        pytest.param(
            "with_think",
            partial(build_system_open_think_last_request, settings=None),
            _SYSTEM_OPEN_THINK_LAST_EXPECTED,
            _SYSTEM_OPEN_THINK_LAST_TEXT,
            id="with-think-system-open-think-last",
        ),
        pytest.param(
            "no_think",
            partial(build_non_ascii_call_arguments_request, "123456789", settings=None),
            _NON_ASCII_CALL_ARGUMENTS_EXPECTED,
            _NON_ASCII_CALL_ARGUMENTS_TEXT,
            id="no-think-non-ascii-call-arguments",
        ),
        pytest.param(
            "no_think",
            partial(build_tools_before_first_user_request, settings=None),
            _TOOLS_BEFORE_FIRST_USER_EXPECTED,
            _TOOLS_BEFORE_FIRST_USER_TEXT,
            id="no-think-tools-before-first-user",
        ),
        pytest.param(
            "audio",
            partial(build_system_then_audio_user_request, num_samples=1600, sampling_rate=16_000, settings=None),
            _SYSTEM_THEN_USER_AUDIO_EXPECTED,
            _SYSTEM_THEN_USER_AUDIO_TEXT,
            id="audio-system-then-user-audio",
        ),
        pytest.param(
            "image",
            partial(build_tool_result_image_request, settings=None),
            _TOOL_RESULT_IMAGE_EXPECTED,
            _TOOL_RESULT_IMAGE_TEXT,
            id="image-tool-result-image",
        ),
        pytest.param(
            "no_think",
            partial(build_normal_request, settings=None),
            _SHARED_NORMAL_EXPECTED,
            _SHARED_NORMAL_TEXT,
            id="no-think-shared-normal",
        ),
        pytest.param(
            "with_think",
            partial(build_normal_request, settings=None),
            _SHARED_NORMAL_EXPECTED,
            _SHARED_NORMAL_TEXT,
            id="with-think-shared-normal",
        ),
        pytest.param(
            "no_think",
            partial(build_tools_singleturn_request, settings=None),
            _SHARED_TOOLS_SINGLETURN_EXPECTED,
            _SHARED_TOOLS_SINGLETURN_TEXT,
            id="no-think-shared-tools-singleturn",
        ),
        pytest.param(
            "with_think",
            partial(build_tools_singleturn_request, settings=None),
            _SHARED_TOOLS_SINGLETURN_EXPECTED,
            _SHARED_TOOLS_SINGLETURN_TEXT,
            id="with-think-shared-tools-singleturn",
        ),
        pytest.param(
            "no_think",
            partial(build_tools_multiturn_request, settings=None),
            _SHARED_TOOLS_MULTITURN_EXPECTED,
            _SHARED_TOOLS_MULTITURN_TEXT,
            id="no-think-shared-tools-multiturn",
        ),
        pytest.param(
            "with_think",
            partial(build_tools_multiturn_request, settings=None),
            _SHARED_TOOLS_MULTITURN_EXPECTED,
            _SHARED_TOOLS_MULTITURN_TEXT,
            id="with-think-shared-tools-multiturn",
        ),
        pytest.param(
            "no_think",
            partial(build_system_message_request, settings=None),
            _SHARED_SYSTEM_MESSAGE_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_TEXT,
            id="no-think-shared-system-message",
        ),
        pytest.param(
            "no_think",
            partial(build_system_message_multiturn_request, settings=None),
            _SHARED_SYSTEM_MESSAGE_MULTITURN_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_MULTITURN_TEXT,
            id="no-think-shared-system-message-multiturn",
        ),
        pytest.param(
            "no_think",
            partial(build_system_message_prefixed_final_request, settings=None),
            _SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT,
            id="no-think-shared-system-message-prefixed-final",
        ),
        pytest.param(
            "no_think",
            partial(build_system_message_tools_multiturn_request, settings=None),
            _SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT,
            id="no-think-shared-system-message-tools-multiturn",
        ),
        pytest.param(
            "with_think",
            partial(build_system_message_request, settings=None),
            _SHARED_SYSTEM_MESSAGE_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_TEXT,
            id="with-think-shared-system-message",
        ),
        pytest.param(
            "with_think",
            partial(build_system_message_multiturn_request, settings=None),
            _SHARED_SYSTEM_MESSAGE_MULTITURN_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_MULTITURN_TEXT,
            id="with-think-shared-system-message-multiturn",
        ),
        pytest.param(
            "with_think",
            partial(build_system_message_prefixed_final_request, settings=None),
            _SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT,
            id="with-think-shared-system-message-prefixed-final",
        ),
        pytest.param(
            "with_think",
            partial(build_system_message_tools_multiturn_request, settings=None),
            _SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_EXPECTED,
            _SHARED_SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT,
            id="with-think-shared-system-message-tools-multiturn",
        ),
        pytest.param(
            "no_think",
            partial(build_tool_message_request, settings=None),
            _SHARED_TOOL_MESSAGE_EXPECTED,
            _SHARED_TOOL_MESSAGE_TEXT,
            id="no-think-shared-tool-message",
        ),
        pytest.param(
            "with_think",
            partial(build_tool_message_request, settings=None),
            _SHARED_TOOL_MESSAGE_EXPECTED,
            _SHARED_TOOL_MESSAGE_TEXT,
            id="with-think-shared-tool-message",
        ),
        pytest.param(
            "no_think",
            partial(build_assistant_tool_calls_request, settings=None),
            _SHARED_ASSISTANT_TOOL_CALLS_EXPECTED,
            _SHARED_ASSISTANT_TOOL_CALLS_TEXT,
            id="no-think-shared-assistant-tool-calls",
        ),
        pytest.param(
            "no_think",
            partial(build_assistant_list_content_request, settings=None),
            _SHARED_ASSISTANT_LIST_CONTENT_EXPECTED,
            _SHARED_ASSISTANT_LIST_CONTENT_TEXT,
            id="no-think-shared-assistant-list-content",
        ),
        pytest.param(
            "with_think",
            partial(build_assistant_tool_calls_request, settings=None),
            _SHARED_ASSISTANT_TOOL_CALLS_EXPECTED,
            _SHARED_ASSISTANT_TOOL_CALLS_TEXT,
            id="with-think-shared-assistant-tool-calls",
        ),
        pytest.param(
            "with_think",
            partial(build_assistant_list_content_request, settings=None),
            _SHARED_ASSISTANT_LIST_CONTENT_EXPECTED,
            _SHARED_ASSISTANT_LIST_CONTENT_TEXT,
            id="with-think-shared-assistant-list-content",
        ),
        pytest.param(
            "no_think",
            partial(build_system_and_user_words_request, settings=None),
            _SHARED_SYSTEM_AND_USER_WORDS_EXPECTED,
            _SHARED_SYSTEM_AND_USER_WORDS_TEXT,
            id="no-think-shared-system-and-user-words",
        ),
        pytest.param(
            "with_think",
            partial(build_system_and_user_words_request, settings=None),
            _SHARED_SYSTEM_AND_USER_WORDS_EXPECTED,
            _SHARED_SYSTEM_AND_USER_WORDS_TEXT,
            id="with-think-shared-system-and-user-words",
        ),
        pytest.param(
            "no_think",
            partial(build_truncation_last_user_at_limit_request, settings=None),
            _SHARED_TRUNCATION_LAST_USER_AT_LIMIT_EXPECTED,
            _SHARED_TRUNCATION_LAST_USER_AT_LIMIT_TEXT,
            id="no-think-shared-truncation-last-user-at-limit",
        ),
        pytest.param(
            "with_think",
            partial(build_truncation_last_user_at_limit_request, settings=None),
            _SHARED_TRUNCATION_LAST_USER_AT_LIMIT_EXPECTED,
            _SHARED_TRUNCATION_LAST_USER_AT_LIMIT_TEXT,
            id="with-think-shared-truncation-last-user-at-limit",
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
            "no_think",
            partial(build_arbitrary_tool_call_id_request, "x", settings=None),
            _ARBITRARY_TOOL_CALL_ID_EXPECTED,
            _ARBITRARY_TOOL_CALL_ID_TEXT,
            id="no-think-shared-arbitrary-tool-call-id",
        ),
        pytest.param(
            "with_think",
            partial(build_arbitrary_tool_call_id_request, "call/id-1", settings=None),
            _ARBITRARY_TOOL_CALL_ID_EXPECTED,
            _ARBITRARY_TOOL_CALL_ID_TEXT,
            id="with-think-shared-arbitrary-tool-call-id",
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
        pytest.param(
            "audio",
            partial(build_system_then_audio_user_request, settings=None),
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
            partial(build_empty_assistant_request, settings=None),
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="no-think-shared-empty-assistant",
        ),
        pytest.param(
            "with_think",
            partial(build_empty_assistant_request, settings=None),
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="with-think-shared-empty-assistant",
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

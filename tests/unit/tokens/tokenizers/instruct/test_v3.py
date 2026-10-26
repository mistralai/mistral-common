r"""Direct formatting tests for both bundled v3 instruct backends."""

from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.chunk import ContentChunk, ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import InstructTokenizer, Tokenized
from mistral_common.tokens.tokenizers.image import DATASET_MEAN, DATASET_STD
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.unit.tokens.tokenizers.instruct.request_builders import (
    build_empty_assistant_request,
    build_prefixed_final_message_request,
    build_system_message_request,
    build_system_tools_multiturn_request,
    build_tool_call_request,
    build_tools_multiturn_request,
    build_tools_singleturn_request,
)
from tests.utils import assert_tokenized_equal, decode_keep

# fmt: off
_SPM_TOOLS_SINGLETURN_TOKENS = [
    1, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113, 7230, 29508, 1316,
    1113, 7286, 2032, 1113, 29508, 1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3, 1032, 4
]
_TEKKEN_TOOLS_SINGLETURN_TOKENS = [
    1, 5, 1091, 19227, 4994, 2811, 1429, 5165, 1897, 1429, 5165, 2811, 16753, 2391, 2811, 1429, 71440, 1049, 1897,
    1429, 14653, 2811, 1429, 1049, 1897, 1429, 26204, 2811, 1445, 2821, 27028, 6, 3, 1097, 4
]
_SPM_TOOLS_MULTITURN_TOKENS = [
    1, 3, 1032, 4, 1055, 2, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032,
    1113, 7230, 29508, 1316, 1113, 7286, 2032, 1113, 29508, 1316, 1113, 12206, 2032, 1139, 1743, 1649, 10598,
    1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113, 7230, 29518, 1316, 1113, 7286,
    2032, 1113, 29518, 1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3, 1045, 4, 1049, 2
]
_TEKKEN_TOOLS_MULTITURN_TOKENS = [
    1, 3, 1097, 4, 1098, 2, 5, 1091, 19227, 4994, 2811, 1429, 5165, 1897, 1429, 5165, 2811, 16753, 2391, 2811,
    1429, 71440, 1049, 1897, 1429, 14653, 2811, 1429, 1049, 1897, 1429, 26204, 2811, 1445, 2821, 4179, 16753,
    4994, 2811, 1429, 5165, 1897, 1429, 5165, 2811, 16753, 2391, 2811, 1429, 71440, 1050, 1897, 1429, 14653,
    2811, 1429, 1050, 1897, 1429, 26204, 2811, 1445, 2821, 27028, 6, 3, 1099, 4, 1100, 2
]
_SPM_SYSTEM_TOOLS_MULTITURN_TOKENS = [
    1, 3, 1032, 4, 1055, 2, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032,
    1113, 7230, 29508, 1316, 1113, 7286, 2032, 1113, 29508, 1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3,
    17889, 23294, 781, 781, 29485, 4, 1049, 2
]
_TEKKEN_SYSTEM_TOOLS_MULTITURN_TOKENS = [
    1, 3, 1097, 4, 1098, 2, 5, 1091, 19227, 4994, 2811, 1429, 5165, 1897, 1429, 5165, 2811, 16753, 2391, 2811,
    1429, 71440, 1049, 1897, 1429, 14653, 2811, 1429, 1049, 1897, 1429, 26204, 2811, 1445, 2821, 27028, 6, 3,
    101289, 58343, 1267, 1099, 4, 1100, 2
]
_SPM_PREFIXED_FINAL_MESSAGE_TOKENS = [1, 3, 1032, 4, 1055, 2, 3, 17889, 23294, 781, 781, 29485, 4, 1049]
_TEKKEN_PREFIXED_FINAL_MESSAGE_TOKENS = [1, 3, 1097, 4, 1098, 2, 3, 101289, 58343, 1267, 1099, 4, 1100]
_SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 13450, 1113, 1081, 2032, 1113,
    29502, 29507, 10925, 2, 8, 10598, 4557, 2032, 1113, 29483, 1316, 1113, 3613, 29498, 1081, 2032, 1113, 29502,
    18163, 9, 1085, 2, 3, 1053, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 13450,
    1113, 1081, 2032, 1113, 29508, 29507, 10925, 2, 8, 10598, 4557, 2032, 1113, 29483, 1316, 1113, 3613, 29498,
    1081, 2032, 1113, 29508, 18163, 9
]
_TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TOKENS = [
    1, 3, 1097, 4, 9, 1091, 19227, 2391, 2811, 1429, 1098, 1897, 1429, 61906, 2811, 58208, 1429, 1327, 2811,
    1429, 1048, 1034, 27028, 2, 7, 19227, 5431, 2811, 1429, 1100, 1897, 1429, 19881, 3384, 2811, 1429, 1048,
    46005, 8, 1101, 2, 3, 1102, 4, 9, 1091, 19227, 2391, 2811, 1429, 1098, 1897, 1429, 61906, 2811, 58208,
    1429, 1327, 2811, 1429, 1049, 1034, 27028, 2, 7, 19227, 5431, 2811, 1429, 1100, 1897, 1429, 19881, 3384,
    2811, 1429, 1049, 46005, 8
]
_SPM_TOOL_MESSAGE_MULTIPLE_CALLS_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 13450, 1113, 1081, 2032, 1113,
    29502, 8474, 10598, 1629, 2032, 1113, 29543, 1316, 1113, 17452, 2032, 13450, 1113, 1081, 2032, 1113, 29508,
    29507, 10925, 2, 8, 10598, 4557, 2032, 1113, 29483, 1316, 1113, 3613, 29498, 1081, 2032, 1113, 29502, 18163,
    9, 8, 10598, 4557, 2032, 1113, 29483, 1316, 1113, 3613, 29498, 1081, 2032, 1113, 29508, 18163, 9, 1085,
    2, 3, 1053, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 13450, 1113, 1081, 2032,
    1113, 29518, 8474, 10598, 1629, 2032, 1113, 29543, 1316, 1113, 17452, 2032, 13450, 1113, 1081, 2032, 1113,
    29538, 29507, 10925, 2, 8, 10598, 4557, 2032, 1113, 29483, 1316, 1113, 3613, 29498, 1081, 2032, 1113, 29518,
    18163, 9, 8, 10598, 4557, 2032, 1113, 29483, 1316, 1113, 3613, 29498, 1081, 2032, 1113, 29538, 18163, 9
]
_TEKKEN_TOOL_MESSAGE_MULTIPLE_CALLS_TOKENS = [
    1, 3, 1097, 4, 9, 1091, 19227, 2391, 2811, 1429, 1098, 1897, 1429, 61906, 2811, 58208, 1429, 1327, 2811,
    1429, 1048, 50666, 16753, 2391, 2811, 1429, 1113, 1897, 1429, 61906, 2811, 58208, 1429, 1327, 2811, 1429,
    1049, 1034, 27028, 2, 7, 19227, 5431, 2811, 1429, 1100, 1897, 1429, 19881, 3384, 2811, 1429, 1048, 46005,
    8, 7, 19227, 5431, 2811, 1429, 1100, 1897, 1429, 19881, 3384, 2811, 1429, 1049, 46005, 8, 1101, 2, 3,
    1102, 4, 9, 1091, 19227, 2391, 2811, 1429, 1098, 1897, 1429, 61906, 2811, 58208, 1429, 1327, 2811, 1429,
    1050, 50666, 16753, 2391, 2811, 1429, 1113, 1897, 1429, 61906, 2811, 58208, 1429, 1327, 2811, 1429, 1051,
    1034, 27028, 2, 7, 19227, 5431, 2811, 1429, 1100, 1897, 1429, 19881, 3384, 2811, 1429, 1050, 46005, 8,
    7, 19227, 5431, 2811, 1429, 1100, 1897, 1429, 19881, 3384, 2811, 1429, 1051, 46005, 8
]
_SPM_TOOL_MESSAGE_PLAIN_STRING_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 13450, 1113, 1081, 2032, 1113,
    29508, 29518, 29538, 29549, 29550, 29552, 29555, 29551, 29542, 29507, 10925, 2, 8, 10598, 4557, 2032, 1113,
    29483, 1316, 1113, 3613, 29498, 1081, 2032, 1113, 29508, 29518, 29538, 29549, 29550, 29552, 29555, 29551, 29542,
    18163, 9
]
_TEKKEN_TOOL_MESSAGE_PLAIN_STRING_TOKENS = [
    1, 3, 1097, 4, 9, 1091, 19227, 2391, 2811, 1429, 1098, 1897, 1429, 61906, 2811, 58208, 1429, 1327, 2811,
    1429, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1034, 27028, 2, 7, 19227, 5431, 2811, 1429, 1100,
    1897, 1429, 19881, 3384, 2811, 1429, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 46005, 8
]
_SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 13450, 1113, 1081, 2032, 1113,
    29508, 29518, 29538, 29549, 29550, 29552, 29555, 29551, 29542, 29507, 10925, 2, 8, 10598, 4557, 2032, 10598,
    29476, 2032, 29473, 29508, 1649, 1113, 3613, 29498, 1081, 2032, 1113, 29508, 29518, 29538, 29549, 29550, 29552,
    29555, 29551, 29542, 18163, 9
]
_TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_TOKENS = [
    1, 3, 1097, 4, 9, 1091, 19227, 2391, 2811, 1429, 1098, 1897, 1429, 61906, 2811, 58208, 1429, 1327, 2811,
    1429, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1034, 27028, 2, 7, 19227, 5431, 2811, 16753,
    1097, 2811, 1032, 1049, 4179, 1429, 19881, 3384, 2811, 1429, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056,
    1057, 46005, 8
]
_SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 13450, 1113, 1081, 2032, 1113,
    29508, 29518, 29538, 29549, 29550, 29552, 29555, 29551, 29542, 29507, 10925, 2, 8, 10598, 4557, 2032, 1113,
    29483, 1426, 29507, 29476, 18125, 29473, 29508, 14162, 1113, 3613, 29498, 1081, 2032, 1113, 29508, 29518, 29538,
    29549, 29550, 29552, 29555, 29551, 29542, 18163, 9
]
_TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_TOKENS = [
    1, 3, 1097, 4, 9, 1091, 19227, 2391, 2811, 1429, 1098, 1897, 1429, 61906, 2811, 58208, 1429, 1327, 2811,
    1429, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1034, 27028, 2, 7, 19227, 5431, 2811, 1429, 1100,
    123085, 1097, 37297, 1032, 1049, 35697, 1429, 19881, 3384, 2811, 1429, 1049, 1050, 1051, 1052, 1053, 1054,
    1055, 1056, 1057, 46005, 8
]
_SPM_TOOL_CALL_WITHOUT_ID_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 1139, 1743, 29561, 2
]
_TEKKEN_TOOL_CALL_WITHOUT_ID_TOKENS = [
    1, 3, 1097, 4, 9, 1091, 19227, 2391, 2811, 1429, 1098, 1897, 1429, 61906, 2811, 1445, 2821, 1093, 2
]
_SPM_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TOKENS = [
    1, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113, 29490, 1316,
    1113, 7286, 2032, 1113, 29485, 2783, 29565, 1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3, 1032, 4, 5,
    1501, 7567, 1629, 2032, 1113, 29490, 1316, 1113, 17452, 2032, 10598, 29512, 2032, 1113, 29565, 8474, 1113,
    1081, 2032, 1113, 29508, 29518, 29538, 29549, 29550, 29552, 29555, 29551, 29542, 29507, 10925, 2, 8,
    10598, 4557, 2032, 1113, 29565, 1316, 1113, 3613, 29498, 1081, 2032, 1113, 29508, 29518, 29538, 29549,
    29550, 29552, 29555, 29551, 29542, 18163, 9
]
_TEKKEN_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TOKENS = [
    1, 5, 1091, 19227, 4994, 2811, 1429, 5165, 1897, 1429, 5165, 2811, 16753, 2391, 2811, 1429, 1102, 1897, 1429,
    14653, 2811, 1429, 3173, 1102, 1337, 1897, 1429, 26204, 2811, 1445, 2821, 27028, 6, 3, 1097, 4, 9, 1091, 19227,
    2391, 2811, 1429, 1102, 1897, 1429, 61906, 2811, 16753, 1120, 2811, 1429, 1337, 50666, 1429, 1327, 2811, 1429,
    1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1034, 27028, 2, 7, 19227, 5431, 2811, 1429, 1337, 1897,
    1429, 19881, 3384, 2811, 1429, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 46005, 8
]
_TEKKEN_USER_TEXT_CHUNKS_TOKENS = [1, 3, 1097, 1098, 4]
_TEKKEN_USER_TEXT_CHUNKS_SYSTEM_PROMPT_TOKENS = [1, 3, 1115, 1267, 1097, 1098, 4]
_TEKKEN_SYSTEM_PROMPT_LAST_USER_ONLY_TOKENS = [1, 3, 1097, 4, 1098, 2, 3, 1115, 1267, 1099, 4]
_TEKKEN_MM_TEXT_THEN_IMAGE_TOKENS = [1, 3, 10, 13, 1097, 4]
_TEKKEN_MM_TEXT_THEN_IMAGE_SYSTEM_PROMPT_TOKENS = [1, 3, 1115, 1267, 10, 13, 1097, 4]
# fmt: on

_RED_16X16_IMAGE = np.broadcast_to(
    array=(
        (
            np.array(object=(255, 0, 0), dtype=np.float32) / np.float32(255.0)
            - np.array(object=DATASET_MEAN, dtype=np.float32)
        )
        / np.array(object=DATASET_STD, dtype=np.float32)
    )[:, None, None],
    shape=(3, 16, 16),
)

_SPM_TOOLS_SINGLETURN_EXPECTED = Tokenized(tokens=_SPM_TOOLS_SINGLETURN_TOKENS)
_TEKKEN_TOOLS_SINGLETURN_EXPECTED = Tokenized(tokens=_TEKKEN_TOOLS_SINGLETURN_TOKENS)
_SPM_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_SPM_TOOLS_MULTITURN_TOKENS)
_TEKKEN_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_TEKKEN_TOOLS_MULTITURN_TOKENS)
_SPM_SYSTEM_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_SPM_SYSTEM_TOOLS_MULTITURN_TOKENS)
_TEKKEN_SYSTEM_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_TEKKEN_SYSTEM_TOOLS_MULTITURN_TOKENS)
_SPM_PREFIXED_FINAL_MESSAGE_EXPECTED = Tokenized(tokens=_SPM_PREFIXED_FINAL_MESSAGE_TOKENS, prefix_ids=[1049])
_TEKKEN_PREFIXED_FINAL_MESSAGE_EXPECTED = Tokenized(tokens=_TEKKEN_PREFIXED_FINAL_MESSAGE_TOKENS, prefix_ids=[1100])
_SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_EXPECTED = Tokenized(
    tokens=_SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TOKENS
)
_TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_EXPECTED = Tokenized(
    tokens=_TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TOKENS
)
_SPM_TOOL_MESSAGE_MULTIPLE_CALLS_EXPECTED = Tokenized(tokens=_SPM_TOOL_MESSAGE_MULTIPLE_CALLS_TOKENS)
_TEKKEN_TOOL_MESSAGE_MULTIPLE_CALLS_EXPECTED = Tokenized(tokens=_TEKKEN_TOOL_MESSAGE_MULTIPLE_CALLS_TOKENS)
_SPM_TOOL_MESSAGE_PLAIN_STRING_EXPECTED = Tokenized(tokens=_SPM_TOOL_MESSAGE_PLAIN_STRING_TOKENS)
_TEKKEN_TOOL_MESSAGE_PLAIN_STRING_EXPECTED = Tokenized(tokens=_TEKKEN_TOOL_MESSAGE_PLAIN_STRING_TOKENS)
_SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_EXPECTED = Tokenized(tokens=_SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_TOKENS)
_TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_EXPECTED = Tokenized(tokens=_TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_TOKENS)
_SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_EXPECTED = Tokenized(tokens=_SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_TOKENS)
_TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_EXPECTED = Tokenized(tokens=_TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_TOKENS)
_SPM_TOOL_CALL_WITHOUT_ID_EXPECTED = Tokenized(tokens=_SPM_TOOL_CALL_WITHOUT_ID_TOKENS)
_TEKKEN_TOOL_CALL_WITHOUT_ID_EXPECTED = Tokenized(tokens=_TEKKEN_TOOL_CALL_WITHOUT_ID_TOKENS)
_SPM_NON_ASCII_TOOLS_CALLS_AND_RESULTS_EXPECTED = Tokenized(tokens=_SPM_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TOKENS)
_TEKKEN_NON_ASCII_TOOLS_CALLS_AND_RESULTS_EXPECTED = Tokenized(tokens=_TEKKEN_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TOKENS)
_TEKKEN_USER_TEXT_CHUNKS_EXPECTED = Tokenized(tokens=_TEKKEN_USER_TEXT_CHUNKS_TOKENS)
_TEKKEN_USER_TEXT_CHUNKS_SYSTEM_PROMPT_EXPECTED = Tokenized(tokens=_TEKKEN_USER_TEXT_CHUNKS_SYSTEM_PROMPT_TOKENS)
_TEKKEN_SYSTEM_PROMPT_LAST_USER_ONLY_EXPECTED = Tokenized(tokens=_TEKKEN_SYSTEM_PROMPT_LAST_USER_ONLY_TOKENS)
_TEKKEN_MM_TEXT_THEN_IMAGE_EXPECTED = Tokenized(
    tokens=_TEKKEN_MM_TEXT_THEN_IMAGE_TOKENS,
    images=[_RED_16X16_IMAGE],
)
_TEKKEN_MM_TEXT_THEN_IMAGE_SYSTEM_PROMPT_EXPECTED = Tokenized(
    tokens=_TEKKEN_MM_TEXT_THEN_IMAGE_SYSTEM_PROMPT_TOKENS,
    images=[_RED_16X16_IMAGE],
)

_SPM_TOOLS_SINGLETURN_TEXT = (
    '<s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":'
    '▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}][/AVAILABLE_TOOLS][INST]▁a[/INST]'
)
_TEKKEN_TOOLS_SINGLETURN_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", '
    '"description": "1", "parameters": {}}}][/AVAILABLE_TOOLS][INST]a[/INST]'
)
_SPM_TOOLS_MULTITURN_TEXT = (
    "<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁["
    '{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}},'
    '▁{"type":▁"function",▁"function":▁{"name":▁"tool2",▁"description":▁"2",▁"parameters":▁{}}}]'
    "[/AVAILABLE_TOOLS][INST]▁c[/INST]▁d</s>"
)
_TEKKEN_TOOLS_MULTITURN_TEXT = (
    "<s>[INST]a[/INST]b</s>[AVAILABLE_TOOLS]["
    '{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}, '
    '{"type": "function", "function": {"name": "tool2", "description": "2", "parameters": {}}}]'
    "[/AVAILABLE_TOOLS][INST]c[/INST]d</s>"
)
_SPM_SYSTEM_TOOLS_MULTITURN_TEXT = (
    "<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁["
    '{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}'
    "][/AVAILABLE_TOOLS][INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d</s>"
)
_TEKKEN_SYSTEM_TOOLS_MULTITURN_TEXT = (
    "<s>[INST]a[/INST]b</s>[AVAILABLE_TOOLS]["
    '{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}'
    "][/AVAILABLE_TOOLS][INST]SYSTEM\n\nc[/INST]d</s>"
)
_SPM_PREFIXED_FINAL_MESSAGE_TEXT = "<s>[INST]▁a[/INST]▁b</s>[INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d"
_TEKKEN_PREFIXED_FINAL_MESSAGE_TEXT = "<s>[INST]a[/INST]b</s>[INST]SYSTEM\n\nc[/INST]d"
_SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"0"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"0"}[/TOOL_RESULTS]▁e</s>[INST]▁f[/INST]'
    '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"1"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"1"}[/TOOL_RESULTS]'
)
_TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TEXT = (
    '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "0"}]</s>'
    '[TOOL_RESULTS]{"content": "d", "call_id": "0"}[/TOOL_RESULTS]e</s>[INST]f[/INST]'
    '[TOOL_CALLS][{"name": "b", "arguments": {}, "id": "1"}]</s>'
    '[TOOL_RESULTS]{"content": "d", "call_id": "1"}[/TOOL_RESULTS]'
)
_SPM_TOOL_MESSAGE_MULTIPLE_CALLS_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"0"},▁'
    '{"name":▁"q",▁"arguments":▁{},▁"id":▁"1"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"0"}[/TOOL_RESULTS]'
    '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"1"}[/TOOL_RESULTS]▁e</s>[INST]▁f[/INST]'
    '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"2"},▁'
    '{"name":▁"q",▁"arguments":▁{},▁"id":▁"3"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"2"}[/TOOL_RESULTS]'
    '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"3"}[/TOOL_RESULTS]'
)
_TEKKEN_TOOL_MESSAGE_MULTIPLE_CALLS_TEXT = (
    '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "0"}, '
    '{"name": "q", "arguments": {}, "id": "1"}]</s>'
    '[TOOL_RESULTS]{"content": "d", "call_id": "0"}[/TOOL_RESULTS]'
    '[TOOL_RESULTS]{"content": "d", "call_id": "1"}[/TOOL_RESULTS]e</s>[INST]f[/INST]'
    '[TOOL_CALLS][{"name": "b", "arguments": {}, "id": "2"}, '
    '{"name": "q", "arguments": {}, "id": "3"}]</s>'
    '[TOOL_RESULTS]{"content": "d", "call_id": "2"}[/TOOL_RESULTS]'
    '[TOOL_RESULTS]{"content": "d", "call_id": "3"}[/TOOL_RESULTS]'
)
_SPM_TOOL_MESSAGE_PLAIN_STRING_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"123456789"}[/TOOL_RESULTS]'
)
_TEKKEN_TOOL_MESSAGE_PLAIN_STRING_TEXT = (
    '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "123456789"}]</s>'
    '[TOOL_RESULTS]{"content": "d", "call_id": "123456789"}[/TOOL_RESULTS]'
)
_SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁{"a":▁1},▁"call_id":▁"123456789"}[/TOOL_RESULTS]'
)
_TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_TEXT = (
    '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "123456789"}]</s>'
    '[TOOL_RESULTS]{"content": {"a": 1}, "call_id": "123456789"}[/TOOL_RESULTS]'
)
_SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_TEXT = (
    r'<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>'
    r'[TOOL_RESULTS]▁{"content":▁"d{\"a\":▁1}",▁"call_id":▁"123456789"}[/TOOL_RESULTS]'
)
_TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_TEXT = (
    r'<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "123456789"}]</s>'
    r'[TOOL_RESULTS]{"content": "d{\"a\": 1}", "call_id": "123456789"}[/TOOL_RESULTS]'
)
_SPM_TOOL_CALL_WITHOUT_ID_TEXT = '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>'
_TEKKEN_TOOL_CALL_WITHOUT_ID_TEXT = '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}}]</s>'
_SPM_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TEXT = (
    '<s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"f",▁"description":▁"café",▁"parameters":▁{}}}'
    '][/AVAILABLE_TOOLS][INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"f",▁"arguments":▁{"x":▁"é"},▁"id":▁"123456789"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁"é",▁"call_id":▁"123456789"}[/TOOL_RESULTS]'
)
_TEKKEN_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "f", "description": "café", '
    '"parameters": {}}}][/AVAILABLE_TOOLS][INST]a[/INST][TOOL_CALLS][{"name": "f", "arguments": '
    '{"x": "é"}, "id": "123456789"}]</s>[TOOL_RESULTS]{"content": "é", "call_id": "123456789"}[/TOOL_RESULTS]'
)
_TEKKEN_USER_TEXT_CHUNKS_TEXT = "<s>[INST]ab[/INST]"
_TEKKEN_USER_TEXT_CHUNKS_SYSTEM_PROMPT_TEXT = "<s>[INST]s\n\nab[/INST]"
_TEKKEN_SYSTEM_PROMPT_LAST_USER_ONLY_TEXT = "<s>[INST]a[/INST]b</s>[INST]s\n\nc[/INST]"
_TEKKEN_MM_TEXT_THEN_IMAGE_TEXT = "<s>[INST][IMG][IMG_END]a[/INST]"
_TEKKEN_MM_TEXT_THEN_IMAGE_SYSTEM_PROMPT_TEXT = "<s>[INST]s\n\n[IMG][IMG_END]a[/INST]"

_TOOL_CALL_AND_CONTENT_MESSAGE = r"^Cannot have tool calls and content defined in the same assistant message"
_INVALID_ASSISTANT_MESSAGE = r"^Invalid assistant message:"
_SYSTEM_MESSAGE_NOT_IMPLEMENTED_MESSAGE = r"^System message encoding not implemented for InstructTokenizerV3$"


@pytest.fixture(scope="module")
def spm_tokenizer() -> InstructTokenizer:
    r"""Provide the bundled v3 SentencePiece instruct tokenizer."""
    return MistralTokenizer.v3().instruct_tokenizer


@pytest.fixture(scope="module")
def tekken_tokenizer() -> InstructTokenizer:
    r"""Provide the bundled v3 Tekken instruct tokenizer."""
    return MistralTokenizer.v3(is_tekken=True).instruct_tokenizer


@pytest.fixture(scope="module")
def tekken_mm_tokenizer() -> InstructTokenizer:
    r"""Provide the bundled v3 Tekken tokenizer with image encoding."""
    return MistralTokenizer.v3(is_tekken=True, is_mm=True).instruct_tokenizer


def _build_tool_message_multiple_shots_with_history_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with tool-call history across multiple shots."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(id="0", function=FunctionCall(name="b", arguments="{}"))]),
            ToolMessage(name="b", content="d", tool_call_id="0"),
            AssistantMessage(content="e"),
            UserMessage(content="f"),
            AssistantMessage(tool_calls=[ToolCall(id="1", function=FunctionCall(name="b", arguments="{}"))]),
            ToolMessage(name="b", content="d", tool_call_id="1"),
        ]
    )


def _build_tool_message_multiple_calls_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with two calls in each of two assistant turns."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(
                tool_calls=[
                    ToolCall(id="0", function=FunctionCall(name="b", arguments="{}")),
                    ToolCall(id="1", function=FunctionCall(name="q", arguments="{}")),
                ]
            ),
            ToolMessage(name="b", content="d", tool_call_id="0"),
            ToolMessage(name="q", content="d", tool_call_id="1"),
            AssistantMessage(content="e"),
            UserMessage(content="f"),
            AssistantMessage(
                tool_calls=[
                    ToolCall(id="2", function=FunctionCall(name="b", arguments="{}")),
                    ToolCall(id="3", function=FunctionCall(name="q", arguments="{}")),
                ]
            ),
            ToolMessage(name="b", content="d", tool_call_id="2"),
            ToolMessage(name="q", content="d", tool_call_id="3"),
        ]
    )


def _build_tool_message_plain_string_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with a plain-string tool result."""
    return _build_tool_message_request(content="d")


def _build_tool_message_json_object_string_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with a JSON-object-string tool result."""
    return _build_tool_message_request(content='{"a": 1}')


def _build_tool_message_text_chunk_list_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with a text-chunk-list tool result."""
    return _build_tool_message_request(content=[TextChunk(text="d"), TextChunk(text='{"a": 1}')])


def _build_tool_message_request(content: str | list[ContentChunk]) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 one-call request with the supplied result content."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="b", arguments="{}"))]),
            ToolMessage(name="b", content=content, tool_call_id="123456789"),
        ]
    )


def _build_explicit_null_id_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with an explicit null-like tool-call ID."""
    return build_tool_call_request(tool_call=ToolCall(id="null", function=FunctionCall(name="b", arguments="{}")))


def _build_default_id_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with a default tool-call ID."""
    return build_tool_call_request(tool_call=ToolCall(function=FunctionCall(name="b", arguments="{}")))


def _build_non_ascii_tools_calls_and_results_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with non-ASCII tool metadata, arguments and result."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=[Tool(function=Function(name="f", description="café", parameters={}))],
        messages=[
            UserMessage(content="a"),
            AssistantMessage(
                tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="f", arguments='{"x": "é"}'))]
            ),
            ToolMessage(name="f", content="é", tool_call_id="123456789"),
        ],
    )


def _build_user_text_chunks_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with two user text chunks."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[TextChunk(text="a"), TextChunk(text="b")])]
    )


def _build_user_text_chunks_system_prompt_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with two user text chunks and a system prompt."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[TextChunk(text="a"), TextChunk(text="b")])],
        system_prompt="s",
    )


def _build_system_prompt_last_user_only_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with list content before a prompted final user."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content=[TextChunk(text="a")]),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
        ],
        system_prompt="s",
    )


def _build_text_then_image_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with text followed by a red image chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="a"),
                    ImageChunk(image=Image.new(mode="RGB", size=(16, 16), color="red")),
                ]
            )
        ]
    )


def _build_text_then_image_system_prompt_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with a prompt and text followed by a red image."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="a"),
                    ImageChunk(image=Image.new(mode="RGB", size=(16, 16), color="red")),
                ]
            )
        ],
        system_prompt="s",
    )


def _build_tool_call_and_content_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request combining assistant content and one tool call."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(
                content="b", tool_calls=[ToolCall(id="0", function=FunctionCall(name="b", arguments="{}"))]
            ),
        ]
    )


@pytest.mark.parametrize(
    argnames="backend, request_builder, expected, expected_text",
    argvalues=[
        pytest.param(
            "spm",
            build_tools_singleturn_request,
            _SPM_TOOLS_SINGLETURN_EXPECTED,
            _SPM_TOOLS_SINGLETURN_TEXT,
            id="spm-tools-singleturn",
        ),
        pytest.param(
            "tekken",
            build_tools_singleturn_request,
            _TEKKEN_TOOLS_SINGLETURN_EXPECTED,
            _TEKKEN_TOOLS_SINGLETURN_TEXT,
            id="tekken-tools-singleturn",
        ),
        pytest.param(
            "spm",
            build_tools_multiturn_request,
            _SPM_TOOLS_MULTITURN_EXPECTED,
            _SPM_TOOLS_MULTITURN_TEXT,
            id="spm-tools-multiturn",
        ),
        pytest.param(
            "tekken",
            build_tools_multiturn_request,
            _TEKKEN_TOOLS_MULTITURN_EXPECTED,
            _TEKKEN_TOOLS_MULTITURN_TEXT,
            id="tekken-tools-multiturn",
        ),
        pytest.param(
            "spm",
            build_system_tools_multiturn_request,
            _SPM_SYSTEM_TOOLS_MULTITURN_EXPECTED,
            _SPM_SYSTEM_TOOLS_MULTITURN_TEXT,
            id="spm-system-tools-multiturn",
        ),
        pytest.param(
            "tekken",
            build_system_tools_multiturn_request,
            _TEKKEN_SYSTEM_TOOLS_MULTITURN_EXPECTED,
            _TEKKEN_SYSTEM_TOOLS_MULTITURN_TEXT,
            id="tekken-system-tools-multiturn",
        ),
        pytest.param(
            "spm",
            build_prefixed_final_message_request,
            _SPM_PREFIXED_FINAL_MESSAGE_EXPECTED,
            _SPM_PREFIXED_FINAL_MESSAGE_TEXT,
            id="spm-prefixed-final-message",
        ),
        pytest.param(
            "tekken",
            build_prefixed_final_message_request,
            _TEKKEN_PREFIXED_FINAL_MESSAGE_EXPECTED,
            _TEKKEN_PREFIXED_FINAL_MESSAGE_TEXT,
            id="tekken-prefixed-final-message",
        ),
        pytest.param(
            "spm",
            _build_tool_message_multiple_shots_with_history_request,
            _SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_EXPECTED,
            _SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TEXT,
            id="spm-tool-message-multiple-shots-with-history",
        ),
        pytest.param(
            "tekken",
            _build_tool_message_multiple_shots_with_history_request,
            _TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_EXPECTED,
            _TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TEXT,
            id="tekken-tool-message-multiple-shots-with-history",
        ),
        pytest.param(
            "spm",
            _build_tool_message_multiple_calls_request,
            _SPM_TOOL_MESSAGE_MULTIPLE_CALLS_EXPECTED,
            _SPM_TOOL_MESSAGE_MULTIPLE_CALLS_TEXT,
            id="spm-tool-message-multiple-calls",
        ),
        pytest.param(
            "tekken",
            _build_tool_message_multiple_calls_request,
            _TEKKEN_TOOL_MESSAGE_MULTIPLE_CALLS_EXPECTED,
            _TEKKEN_TOOL_MESSAGE_MULTIPLE_CALLS_TEXT,
            id="tekken-tool-message-multiple-calls",
        ),
        pytest.param(
            "spm",
            _build_tool_message_plain_string_request,
            _SPM_TOOL_MESSAGE_PLAIN_STRING_EXPECTED,
            _SPM_TOOL_MESSAGE_PLAIN_STRING_TEXT,
            id="spm-tool-message-plain-string",
        ),
        pytest.param(
            "tekken",
            _build_tool_message_plain_string_request,
            _TEKKEN_TOOL_MESSAGE_PLAIN_STRING_EXPECTED,
            _TEKKEN_TOOL_MESSAGE_PLAIN_STRING_TEXT,
            id="tekken-tool-message-plain-string",
        ),
        pytest.param(
            "spm",
            _build_tool_message_json_object_string_request,
            _SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_EXPECTED,
            _SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_TEXT,
            id="spm-tool-message-json-object-string",
        ),
        pytest.param(
            "tekken",
            _build_tool_message_json_object_string_request,
            _TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_EXPECTED,
            _TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_TEXT,
            id="tekken-tool-message-json-object-string",
        ),
        pytest.param(
            "spm",
            _build_tool_message_text_chunk_list_request,
            _SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_EXPECTED,
            _SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_TEXT,
            id="spm-tool-message-text-chunk-list",
        ),
        pytest.param(
            "tekken",
            _build_tool_message_text_chunk_list_request,
            _TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_EXPECTED,
            _TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_TEXT,
            id="tekken-tool-message-text-chunk-list",
        ),
        pytest.param(
            "spm",
            _build_explicit_null_id_request,
            _SPM_TOOL_CALL_WITHOUT_ID_EXPECTED,
            _SPM_TOOL_CALL_WITHOUT_ID_TEXT,
            id="spm-tool-call-explicit-null-id",
        ),
        pytest.param(
            "tekken",
            _build_explicit_null_id_request,
            _TEKKEN_TOOL_CALL_WITHOUT_ID_EXPECTED,
            _TEKKEN_TOOL_CALL_WITHOUT_ID_TEXT,
            id="tekken-tool-call-explicit-null-id",
        ),
        pytest.param(
            "spm",
            _build_default_id_request,
            _SPM_TOOL_CALL_WITHOUT_ID_EXPECTED,
            _SPM_TOOL_CALL_WITHOUT_ID_TEXT,
            id="spm-tool-call-default-id",
        ),
        pytest.param(
            "tekken",
            _build_default_id_request,
            _TEKKEN_TOOL_CALL_WITHOUT_ID_EXPECTED,
            _TEKKEN_TOOL_CALL_WITHOUT_ID_TEXT,
            id="tekken-tool-call-default-id",
        ),
        pytest.param(
            "spm",
            _build_non_ascii_tools_calls_and_results_request,
            _SPM_NON_ASCII_TOOLS_CALLS_AND_RESULTS_EXPECTED,
            _SPM_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TEXT,
            id="spm-non-ascii-tools-calls-and-results",
        ),
        pytest.param(
            "tekken",
            _build_non_ascii_tools_calls_and_results_request,
            _TEKKEN_NON_ASCII_TOOLS_CALLS_AND_RESULTS_EXPECTED,
            _TEKKEN_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TEXT,
            id="tekken-non-ascii-tools-calls-and-results",
        ),
        pytest.param(
            "tekken",
            _build_user_text_chunks_request,
            _TEKKEN_USER_TEXT_CHUNKS_EXPECTED,
            _TEKKEN_USER_TEXT_CHUNKS_TEXT,
            id="tekken-user-text-chunks",
        ),
        pytest.param(
            "tekken",
            _build_user_text_chunks_system_prompt_request,
            _TEKKEN_USER_TEXT_CHUNKS_SYSTEM_PROMPT_EXPECTED,
            _TEKKEN_USER_TEXT_CHUNKS_SYSTEM_PROMPT_TEXT,
            id="tekken-user-text-chunks-system-prompt",
        ),
        pytest.param(
            "tekken",
            _build_system_prompt_last_user_only_request,
            _TEKKEN_SYSTEM_PROMPT_LAST_USER_ONLY_EXPECTED,
            _TEKKEN_SYSTEM_PROMPT_LAST_USER_ONLY_TEXT,
            id="tekken-system-prompt-last-user-only",
        ),
        pytest.param(
            "tekken_mm",
            _build_text_then_image_request,
            _TEKKEN_MM_TEXT_THEN_IMAGE_EXPECTED,
            _TEKKEN_MM_TEXT_THEN_IMAGE_TEXT,
            id="tekken-mm-text-then-image",
        ),
        pytest.param(
            "tekken_mm",
            _build_text_then_image_system_prompt_request,
            _TEKKEN_MM_TEXT_THEN_IMAGE_SYSTEM_PROMPT_EXPECTED,
            _TEKKEN_MM_TEXT_THEN_IMAGE_SYSTEM_PROMPT_TEXT,
            id="tekken-mm-text-then-image-system-prompt",
        ),
    ],
)
def test_encode_instruct(
    request: pytest.FixtureRequest,
    backend: str,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected: Tokenized,
    expected_text: str,
) -> None:
    r"""Preserve complete v3 instruct encodings for each backend and request."""
    tokenizer: InstructTokenizer = request.getfixturevalue(argname=f"{backend}_tokenizer")
    tokenized = tokenizer.encode_instruct(request=request_builder())
    assert_tokenized_equal(actual=tokenized, expected=expected)
    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text


@pytest.mark.parametrize(
    argnames="backend, request_builder, expected_exception, expected_message",
    argvalues=[
        pytest.param(
            "spm",
            _build_tool_call_and_content_request,
            ValueError,
            _TOOL_CALL_AND_CONTENT_MESSAGE,
            id="spm-tool-call-and-content",
        ),
        pytest.param(
            "spm",
            build_empty_assistant_request,
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="spm-empty-assistant",
        ),
        pytest.param(
            "tekken",
            _build_tool_call_and_content_request,
            ValueError,
            _TOOL_CALL_AND_CONTENT_MESSAGE,
            id="tekken-tool-call-and-content",
        ),
        pytest.param(
            "tekken",
            build_empty_assistant_request,
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="tekken-empty-assistant",
        ),
        pytest.param(
            "spm",
            build_system_message_request,
            NotImplementedError,
            _SYSTEM_MESSAGE_NOT_IMPLEMENTED_MESSAGE,
            id="spm-system-message",
        ),
        pytest.param(
            "tekken",
            build_system_message_request,
            NotImplementedError,
            _SYSTEM_MESSAGE_NOT_IMPLEMENTED_MESSAGE,
            id="tekken-system-message",
        ),
    ],
)
def test_encode_instruct_rejected(
    request: pytest.FixtureRequest,
    backend: str,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected_exception: type[Exception],
    expected_message: str | None,
) -> None:
    r"""Preserve direct v3 exception types and specified messages."""
    tokenizer: InstructTokenizer = request.getfixturevalue(argname=f"{backend}_tokenizer")
    instruct_request = request_builder()

    with pytest.raises(expected_exception=expected_exception, match=expected_message):
        tokenizer.encode_instruct(request=instruct_request)

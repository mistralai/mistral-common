r"""Direct formatting tests for the bundled v7 instruct tokenizer and its synthetic Tekken variants."""

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
from PIL import Image

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.chunk import ContentChunk, ImageChunk, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import InstructTokenizer, Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.image import DATASET_MEAN, DATASET_STD, ImageConfig, ImageEncoder
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV7
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import quick_vocab
from tests.test_tokenizer_v7_audio import get_tekkenizer_with_audio
from tests.unit.tokens.tokenizers.instruct.request_builders import (
    build_assistant_tool_calls_request,
    build_audio_user_request,
    build_empty_assistant_request,
    build_image_user_request,
    build_multi_audio_user_request,
    build_multi_image_user_request,
    build_multiturn_audio_request,
    build_multiturn_image_request,
    build_normal_request,
    build_system_message_multiturn_request,
    build_system_message_prefixed_final_request,
    build_system_message_request,
    build_system_message_tools_multiturn_request,
    build_text_then_audio_user_request,
    build_text_then_image_user_request,
    build_tool_message_request,
    build_tools_multiturn_request,
    build_tools_singleturn_request,
)
from tests.utils import assert_tokenized_equal, decode_keep

# fmt: off
_IMAGE_AND_TOOL_RESULT_TOKENS = [1, 3, 10, 10, 14, 10, 10, 15, 1032, 4, 1055, 2, 8, 1055, 18, 1053, 9]
_PREFIXED_ASSISTANT_AFTER_IMAGE_TOKENS = [1, 3, 10, 10, 14, 10, 10, 15, 1032, 4, 1055]
_PREFIXED_CALL_WITHOUT_CONTENT_TOKENS = [
    1, 5, 1501, 7567, 1629, 2032, 1113, 2381, 29498, 6410, 1316, 1113, 17452, 2032, 1139, 1743, 29561
]
_PREFIXED_CALL_WITHOUT_CONTENT_PREFIX_IDS = [
    5, 1501, 7567, 1629, 2032, 1113, 2381, 29498, 6410, 1316, 1113, 17452, 2032, 1139, 1743, 29561
]
_SYSTEM_TOOLS_AND_CALL_TOKENS = [
    1, 16, 1032, 17, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113, 29475,
    1316, 1113, 7286, 2032, 8657, 1113, 12206, 2032, 10598, 1891, 2032, 1113, 3582, 1316, 1113, 11491, 2032, 10598,
    29489, 2032, 10598, 1891, 2032, 1113, 2195, 8474, 1113, 29484, 2032, 10598, 1891, 2032, 1113, 2195, 29507, 1743,
    1743, 10925, 7, 3, 1032, 4, 1055, 5, 1501, 7567, 1629, 2032, 1113, 29475, 1316, 1113, 17452, 2032, 10598, 29489,
    2032, 1113, 29484, 29507, 1743, 29561, 2
]
_TOOL_RESULT_AFTER_HISTORY_TOKENS = [
    1, 16, 1032, 17, 3, 1032, 4, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113,
    29475, 1316, 1113, 7286, 2032, 8657, 1113, 12206, 2032, 10598, 1891, 2032, 1113, 3582, 1316, 1113, 11491, 2032,
    10598, 29489, 2032, 10598, 1891, 2032, 1113, 2195, 8474, 1113, 29484, 2032, 10598, 1891, 2032, 1113, 2195, 29507,
    1743, 1743, 10925, 7, 3, 1045, 4, 1055, 5, 1501, 7567, 1629, 2032, 1113, 29475, 1316, 1113, 17452, 2032, 10598,
    29489, 2032, 1113, 29484, 29507, 1743, 29561, 2, 8, 29473, 29508, 29518, 29538, 29549, 18, 1055, 9
]
_TOOL_CALL_AND_CONTENT_TOKENS = [
    1, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144, 132,
    134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216,
    149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134, 134, 144, 132,
    134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 144, 132, 223, 134, 216,
    221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144, 132, 134, 202, 217, 210, 199,
    216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 150, 134, 144, 132, 134,
    200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134, 134, 144, 132, 134, 212, 197, 214, 197,
    209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 193, 6, 3, 197, 4, 198, 149, 198, 150, 9, 191, 223,
    134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 149, 134, 144, 132, 134, 197, 214, 203, 217, 209, 201, 210, 216,
    215, 134, 158, 132, 223, 225, 144, 132, 134, 205, 200, 134, 158, 132, 134, 148, 148, 148, 148, 148, 148, 148, 148,
    148, 134, 225, 144, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 150, 134, 144, 132, 134, 197, 214,
    203, 217, 209, 201, 210, 216, 215, 134, 158, 132, 223, 225, 144, 132, 134, 205, 200, 134, 158, 132, 134, 149, 149,
    149, 149, 149, 149, 149, 149, 149, 134, 225, 193, 2
]
_PREFIXED_TOOL_CALL_AND_CONTENT_TOKENS = [
    1, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144, 132,
    134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216,
    149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134, 134, 144, 132,
    134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 144, 132, 223, 134, 216,
    221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144, 132, 134, 202, 217, 210, 199,
    216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 150, 134, 144, 132, 134,
    200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134, 134, 144, 132, 134, 212, 197, 214, 197,
    209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 193, 6, 3, 197, 4, 198, 149, 198, 150, 9, 191, 223,
    134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 149, 134, 144, 132, 134, 197, 214, 203, 217, 209, 201, 210, 216,
    215, 134, 158, 132, 223, 225, 144, 132, 134, 205, 200, 134, 158, 132, 134, 148, 148, 148, 148, 148, 148, 148, 148,
    148, 134, 225, 144, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 150, 134, 144, 132, 134, 197, 214,
    203, 217, 209, 201, 210, 216, 215, 134, 158, 132, 223, 225, 144, 132, 134, 205, 200, 134, 158, 132, 134, 149, 149,
    149, 149, 149, 149, 149, 149, 149, 134, 225, 193
]
_PREFIXED_TOOL_CALL_AND_CONTENT_PREFIX_IDS = [
    198, 149, 198, 150, 9, 191, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 149, 134, 144, 132, 134, 197,
    214, 203, 217, 209, 201, 210, 216, 215, 134, 158, 132, 223, 225, 144, 132, 134, 205, 200, 134, 158, 132, 134, 148,
    148, 148, 148, 148, 148, 148, 148, 148, 134, 225, 144, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216,
    150, 134, 144, 132, 134, 197, 214, 203, 217, 209, 201, 210, 216, 215, 134, 158, 132, 223, 225, 144, 132, 134, 205,
    200, 134, 158, 132, 134, 149, 149, 149, 149, 149, 149, 149, 149, 149, 134, 225, 193
]
_TOOL_CALLS_WITH_TWO_RESULTS_TOKENS = [
    1, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144, 132,
    134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216,
    149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134, 134, 144, 132,
    134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 144, 132, 223, 134, 216,
    221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144, 132, 134, 202, 217, 210, 199,
    216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 150, 134, 144, 132, 134,
    200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134, 134, 144, 132, 134, 212, 197, 214, 197,
    209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 193, 6, 3, 197, 4, 198, 149, 198, 150, 9, 191, 223,
    134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 149, 134, 144, 132, 134, 197, 214, 203, 217, 209, 201, 210, 216,
    215, 134, 158, 132, 223, 225, 144, 132, 134, 205, 200, 134, 158, 132, 134, 148, 148, 148, 148, 148, 148, 148, 148,
    148, 134, 225, 144, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 150, 134, 144, 132, 134, 197, 214,
    203, 217, 209, 201, 210, 216, 215, 134, 158, 132, 223, 225, 144, 132, 134, 205, 200, 134, 158, 132, 134, 149, 149,
    149, 149, 149, 149, 149, 149, 149, 134, 225, 193, 2, 7, 148, 148, 148, 148, 148, 148, 148, 148, 148, 19, 214, 149,
    8, 7, 149, 149, 149, 149, 149, 149, 149, 149, 149, 19, 214, 150, 8
]
_TRUNCATION_KEEP_SYS_AND_LAST_MESSAGE_TOKENS = [1, 17, 197, 18, 17, 197, 18, 3, 198, 198, 198, 198, 198, 198, 198, 4]
_TRUNCATION_DROP_FIRST_ASSISTANT_TOKENS = [1, 3, 198, 4, 3, 197, 4, 3, 197, 197, 197, 197, 197, 197, 197, 4]
_TRUNCATION_START_WITH_ASSISTANT_TOKENS = [1, 198, 2, 3, 197, 4, 3, 197, 197, 197, 197, 197, 197, 197, 4]
_TRUNCATION_DROP_BY_CHUNK_1_TOKENS = [1, 3, 197, 4, 198, 198, 198, 198, 198, 198, 198, 2]
_TRUNCATION_DROP_BY_CHUNK_2_TOKENS = [1, 3, 197, 197, 197, 197, 197, 197, 197, 4]
_TRUNCATION_FULL_CONVO_TOKENS = [1, 17, 197, 18, 3, 197, 4, 197, 2, 17, 198, 18, 3, 197, 4]
_TWO_IMAGES_TOKENS = [1, 3, 10, 10, 14, 10, 10, 15, 10, 10, 10, 14, 10, 10, 10, 15, 4]
_TEXT_THEN_TWO_IMAGES_TOKENS = [1, 3, 2086, 10, 10, 14, 10, 10, 15, 10, 10, 10, 14, 10, 10, 10, 15, 4]
_ONE_IMAGE_WITH_TEXT_TOKENS = [1, 3, 10, 10, 14, 10, 10, 15, 2086, 4]
_SYSTEM_LIST_CONTENT_TOKENS = [1, 16, 1036, 17, 3, 1032, 4]
_ASSISTANT_LIST_CONTENT_TOKENS = [1, 3, 1032, 4, 1055, 2]
_SYSTEM_AND_USER_WORDS_TOKENS = [1, 16, 7080, 29477, 2294, 17, 3, 1947, 2138, 4]
_TRUNCATION_LAST_USER_AT_LIMIT_TOKENS = [1, 3, 197, 197, 197, 197, 197, 197, 197, 4]
# fmt: on


def _build_solid_image(rgb: tuple[int, int, int], shape: tuple[int, int, int]) -> np.ndarray:
    r"""Build the normalized `(C, H, W)` array of a one-colour image whose size the encoder keeps."""
    vector = (
        np.array(object=rgb, dtype=np.float32) / np.float32(255.0) - np.array(object=DATASET_MEAN, dtype=np.float32)
    ) / np.array(object=DATASET_STD, dtype=np.float32)
    return np.broadcast_to(array=vector[:, None, None], shape=shape)


# Image rows below check token placement, shape and dtype in the token stream;
# normalization correctness is covered by the image encoder tests.
_RED_4X4_IMAGE = _build_solid_image(rgb=(255, 0, 0), shape=(3, 4, 4))
_BLUE_4X4_IMAGE = _build_solid_image(rgb=(0, 0, 255), shape=(3, 4, 4))
_BLUE_6X4_IMAGE = _build_solid_image(rgb=(0, 0, 255), shape=(3, 4, 6))

_IMAGE_AND_TOOL_RESULT_EXPECTED = Tokenized(tokens=_IMAGE_AND_TOOL_RESULT_TOKENS, images=[_RED_4X4_IMAGE])
_PREFIXED_ASSISTANT_AFTER_IMAGE_EXPECTED = Tokenized(
    tokens=_PREFIXED_ASSISTANT_AFTER_IMAGE_TOKENS, prefix_ids=[1055], images=[_RED_4X4_IMAGE]
)
_PREFIXED_CALL_WITHOUT_CONTENT_EXPECTED = Tokenized(
    tokens=_PREFIXED_CALL_WITHOUT_CONTENT_TOKENS, prefix_ids=_PREFIXED_CALL_WITHOUT_CONTENT_PREFIX_IDS
)
_SYSTEM_TOOLS_AND_CALL_EXPECTED = Tokenized(tokens=_SYSTEM_TOOLS_AND_CALL_TOKENS)
_TOOL_RESULT_AFTER_HISTORY_EXPECTED = Tokenized(tokens=_TOOL_RESULT_AFTER_HISTORY_TOKENS)
_TOOL_CALL_AND_CONTENT_EXPECTED = Tokenized(tokens=_TOOL_CALL_AND_CONTENT_TOKENS)
_PREFIXED_TOOL_CALL_AND_CONTENT_EXPECTED = Tokenized(
    tokens=_PREFIXED_TOOL_CALL_AND_CONTENT_TOKENS, prefix_ids=_PREFIXED_TOOL_CALL_AND_CONTENT_PREFIX_IDS
)
_TOOL_CALLS_WITH_TWO_RESULTS_EXPECTED = Tokenized(tokens=_TOOL_CALLS_WITH_TWO_RESULTS_TOKENS)
_TRUNCATION_KEEP_SYS_AND_LAST_MESSAGE_EXPECTED = Tokenized(tokens=_TRUNCATION_KEEP_SYS_AND_LAST_MESSAGE_TOKENS)
_TRUNCATION_DROP_FIRST_ASSISTANT_EXPECTED = Tokenized(tokens=_TRUNCATION_DROP_FIRST_ASSISTANT_TOKENS)
_TRUNCATION_START_WITH_ASSISTANT_EXPECTED = Tokenized(tokens=_TRUNCATION_START_WITH_ASSISTANT_TOKENS)
_TRUNCATION_DROP_BY_CHUNK_1_EXPECTED = Tokenized(tokens=_TRUNCATION_DROP_BY_CHUNK_1_TOKENS)
_TRUNCATION_DROP_BY_CHUNK_2_EXPECTED = Tokenized(tokens=_TRUNCATION_DROP_BY_CHUNK_2_TOKENS)
_TRUNCATION_FULL_CONVO_EXPECTED = Tokenized(tokens=_TRUNCATION_FULL_CONVO_TOKENS)
_TWO_IMAGES_EXPECTED = Tokenized(tokens=_TWO_IMAGES_TOKENS, images=[_RED_4X4_IMAGE, _BLUE_6X4_IMAGE])
_TEXT_THEN_TWO_IMAGES_EXPECTED = Tokenized(
    tokens=_TEXT_THEN_TWO_IMAGES_TOKENS, images=[_RED_4X4_IMAGE, _BLUE_6X4_IMAGE]
)
_ONE_IMAGE_WITH_TEXT_EXPECTED = Tokenized(tokens=_ONE_IMAGE_WITH_TEXT_TOKENS, images=[_RED_4X4_IMAGE])
_SYSTEM_LIST_CONTENT_EXPECTED = Tokenized(tokens=_SYSTEM_LIST_CONTENT_TOKENS)
_ASSISTANT_LIST_CONTENT_EXPECTED = Tokenized(tokens=_ASSISTANT_LIST_CONTENT_TOKENS)
_SYSTEM_AND_USER_WORDS_EXPECTED = Tokenized(tokens=_SYSTEM_AND_USER_WORDS_TOKENS)
_TRUNCATION_LAST_USER_AT_LIMIT_EXPECTED = Tokenized(tokens=_TRUNCATION_LAST_USER_AT_LIMIT_TOKENS)

_IMAGE_AND_TOOL_RESULT_TEXT = (
    "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]▁a[/INST]▁b</s>[TOOL_RESULTS]▁b[TOOL_CONTENT]▁f[/TOOL_RESULTS]"
)
_PREFIXED_ASSISTANT_AFTER_IMAGE_TEXT = "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]▁a[/INST]▁b"
_PREFIXED_CALL_WITHOUT_CONTENT_TEXT = '<s>[TOOL_CALLS]▁[{"name":▁"test_fn",▁"arguments":▁{}}]'
_SYSTEM_TOOLS_AND_CALL_TEXT = (
    '<s>[SYSTEM_PROMPT]▁a[/SYSTEM_PROMPT][AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"t",'
    '▁"description":▁"",▁"parameters":▁{"type":▁"object",▁"properties":▁{"g":▁{"type":▁"string"},'
    '▁"h":▁{"type":▁"string"}}}}}][/AVAILABLE_TOOLS][INST]▁a[/INST]▁b[TOOL_CALLS]'
    '▁[{"name":▁"t",▁"arguments":▁{"g":▁"h"}}]</s>'
)
_TOOL_RESULT_AFTER_HISTORY_TEXT = (
    '<s>[SYSTEM_PROMPT]▁a[/SYSTEM_PROMPT][INST]▁a[/INST][AVAILABLE_TOOLS]▁[{"type":▁"function",'
    '▁"function":▁{"name":▁"t",▁"description":▁"",▁"parameters":▁{"type":▁"object",'
    '▁"properties":▁{"g":▁{"type":▁"string"},▁"h":▁{"type":▁"string"}}}}}][/AVAILABLE_TOOLS]'
    '[INST]▁c[/INST]▁b[TOOL_CALLS]▁[{"name":▁"t",▁"arguments":▁{"g":▁"h"}}]</s>'
    "[TOOL_RESULTS]▁1234[TOOL_CONTENT]▁b[/TOOL_RESULTS]"
)
_TOOL_CALL_AND_CONTENT_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": '
    '{"name": "t1", "description": "", "parameters": {}}}, '
    '{"type": "function", "function": {"name": "t2", "description"'
    ': "", "parameters": {}}}][/AVAILABLE_TOOLS][INST]a[/INST]b1b2[TOOL_CALLS]'
    '[{"name": "t1", "arguments": {}, "id": "000000000"}, {"name": "t2", "arguments": {}'
    ', "id": "111111111"}]</s>'
)
_PREFIXED_TOOL_CALL_AND_CONTENT_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": '
    '{"name": "t1", "description": "", "parameters": {}}}, '
    '{"type": "function", "function": {"name": "t2", "description"'
    ': "", "parameters": {}}}][/AVAILABLE_TOOLS][INST]a[/INST]b1b2[TOOL_CALLS]'
    '[{"name": "t1", "arguments": {}, "id": "000000000"}, {"name": "t2", "arguments": {}'
    ', "id": "111111111"}]'
)
_TOOL_CALLS_WITH_TWO_RESULTS_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": '
    '{"name": "t1", "description": "", "parameters": {}}}, '
    '{"type": "function", "function": {"name": "t2", "description"'
    ': "", "parameters": {}}}][/AVAILABLE_TOOLS][INST]a[/INST]b1b2[TOOL_CALLS]'
    '[{"name": "t1", "arguments": {}, "id": "000000000"}, {"name": "t2", "arguments": {}'
    ', "id": "111111111"}]</s>'
    "[TOOL_RESULTS]000000000[TOOL_CONTENT]r1[/TOOL_RESULTS]"
    "[TOOL_RESULTS]111111111[TOOL_CONTENT]r2[/TOOL_RESULTS]"
)
_TRUNCATION_KEEP_SYS_AND_LAST_MESSAGE_TEXT = (
    "<s>[SYSTEM_PROMPT]a[/SYSTEM_PROMPT][SYSTEM_PROMPT]a[/SYSTEM_PROMPT][INST]bbbbbbb[/INST]"
)
_TRUNCATION_DROP_FIRST_ASSISTANT_TEXT = "<s>[INST]b[/INST][INST]a[/INST][INST]aaaaaaa[/INST]"
_TRUNCATION_START_WITH_ASSISTANT_TEXT = "<s>b</s>[INST]a[/INST][INST]aaaaaaa[/INST]"
_TRUNCATION_DROP_BY_CHUNK_1_TEXT = "<s>[INST]a[/INST]bbbbbbb</s>"
_TRUNCATION_DROP_BY_CHUNK_2_TEXT = "<s>[INST]aaaaaaa[/INST]"
_TRUNCATION_FULL_CONVO_TEXT = (
    "<s>[SYSTEM_PROMPT]a[/SYSTEM_PROMPT][INST]a[/INST]a</s>[SYSTEM_PROMPT]b[/SYSTEM_PROMPT][INST]a[/INST]"
)
_TWO_IMAGES_TEXT = (
    "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG][IMG_BREAK][IMG][IMG][IMG][IMG_END][/INST]"
)
_TEXT_THEN_TWO_IMAGES_TEXT = (
    "<s>[INST]▁x[IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG][IMG_BREAK][IMG][IMG][IMG][IMG_END][/INST]"
)
_ONE_IMAGE_WITH_TEXT_TEXT = "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]▁x[/INST]"
_SYSTEM_LIST_CONTENT_TEXT = "<s>[SYSTEM_PROMPT]▁s[/SYSTEM_PROMPT][INST]▁a[/INST]"
_ASSISTANT_LIST_CONTENT_TEXT = "<s>[INST]▁a[/INST]▁b</s>"
_SYSTEM_AND_USER_WORDS_TEXT = "<s>[SYSTEM_PROMPT]▁hello▁world[/SYSTEM_PROMPT][INST]▁good▁day[/INST]"
_TRUNCATION_LAST_USER_AT_LIMIT_TEXT = "<s>[INST]aaaaaaa[/INST]"

# fmt: off
_SPM_NORMAL_TOKENS = [1, 3, 1032, 4, 1055, 2, 3, 1045, 4, 1049, 2]
_SPM_TOOLS_SINGLETURN_TOKENS = [
    1, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113, 7230, 29508, 1316,
    1113, 7286, 2032, 1113, 29508, 1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3, 1032, 4]
_SPM_TOOLS_MULTITURN_TOKENS = [
    1, 3, 1032, 4, 1055, 2, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113,
    7230, 29508, 1316, 1113, 7286, 2032, 1113, 29508, 1316, 1113, 12206, 2032, 1139, 1743, 1649, 10598, 1891, 2032,
    1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113, 7230, 29518, 1316, 1113, 7286, 2032, 1113, 29518,
    1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3, 1045, 4, 1049, 2]
_SPM_TOOL_MESSAGE_TOKENS = [
    1, 3, 1032, 4, 8, 29473, 29508, 29518, 29538, 29549, 29550, 29552, 29555, 29551, 29542, 18, 1049, 9]
_SPM_ASSISTANT_TOOL_CALLS_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 13450, 1113, 1081, 2032, 1113,
    29508, 29518, 29538, 29549, 29550, 29552, 29555, 29551, 29542, 29507, 10925, 2
]
_SPM_SYSTEM_MESSAGE_TOKENS = [1, 16, 1036, 17, 3, 1032, 4]
_SPM_SYSTEM_MESSAGE_MULTITURN_TOKENS = [1, 16, 17889, 23294, 17, 3, 1032, 4, 1055, 2, 3, 1045, 4, 1049, 2]
_SPM_SYSTEM_MESSAGE_PREFIXED_FINAL_TOKENS = [1, 16, 17889, 23294, 17, 3, 1032, 4, 1055, 2, 3, 1045, 4, 1049]
_SPM_SYSTEM_MESSAGE_TOOLS_MULTITURN_TOKENS = [
    1, 16, 17889, 23294, 17, 3, 1032, 4, 1055, 2, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032,
    10598, 1629, 2032, 1113, 7230, 29508, 1316, 1113, 7286, 2032, 1113, 29508, 1316, 1113, 12206, 2032, 1139, 1743,
    10925, 7, 3, 1045, 4, 1049, 2
]
_NO_AUDIO_NORMAL_TOKENS = [1, 3, 197, 4, 198, 2, 3, 199, 4, 200, 2]
_NO_AUDIO_TOOLS_SINGLETURN_TOKENS = [
    1, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144,
    132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134,
    216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132,
    134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225,
    193, 6, 3, 197, 4]
_NO_AUDIO_TOOLS_MULTITURN_TOKENS = [
    1, 3, 197, 4, 198, 2, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211,
    210, 134, 144, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134,
    158, 132, 134, 216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210,
    134, 158, 132, 134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223,
    225, 225, 225, 144, 132, 223, 134, 216, 221, 212, 201, 134, 158, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210,
    134, 144, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158, 132, 223, 134, 210, 197, 209, 201, 134, 158,
    132, 134, 216, 211, 211, 208, 150, 134, 144, 132, 134, 200, 201, 215, 199, 214, 205, 212, 216, 205, 211, 210, 134,
    158, 132, 134, 150, 134, 144, 132, 134, 212, 197, 214, 197, 209, 201, 216, 201, 214, 215, 134, 158, 132, 223, 225,
    225, 225, 193, 6, 3, 199, 4, 200, 2]
_NO_AUDIO_TOOL_MESSAGE_TOKENS = [1, 3, 197, 4, 7, 149, 150, 151, 152, 153, 154, 155, 156, 157, 19, 200, 8]
_NO_AUDIO_ASSISTANT_TOOL_CALLS_TOKENS = [
    1, 3, 197, 4, 9, 191, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 198, 134, 144, 132, 134, 197, 214, 203,
    217, 209, 201, 210, 216, 215, 134, 158, 132, 223, 225, 144, 132, 134, 205, 200, 134, 158, 132, 134, 149, 150, 151,
    152, 153, 154, 155, 156, 157, 134, 225, 193, 2
]
_NO_AUDIO_SYSTEM_MESSAGE_TOKENS = [1, 17, 215, 18, 3, 197, 4]
_NO_AUDIO_SYSTEM_MESSAGE_MULTITURN_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 3, 197, 4, 198, 2, 3, 199, 4, 200, 2
]
_NO_AUDIO_SYSTEM_MESSAGE_PREFIXED_FINAL_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 3, 197, 4, 198, 2, 3, 199, 4, 200
]
_NO_AUDIO_SYSTEM_MESSAGE_TOOLS_MULTITURN_TOKENS = [
    1, 17, 183, 189, 183, 184, 169, 177, 18, 3, 197, 4, 198, 2, 5, 191, 223, 134, 216, 221, 212, 201, 134, 158, 132,
    134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 144, 132, 134, 202, 217, 210, 199, 216, 205, 211, 210, 134, 158,
    132, 223, 134, 210, 197, 209, 201, 134, 158, 132, 134, 216, 211, 211, 208, 149, 134, 144, 132, 134, 200, 201,
    215, 199, 214, 205, 212, 216, 205, 211, 210, 134, 158, 132, 134, 149, 134, 144, 132, 134, 212, 197, 214, 197, 209,
    201, 216, 201, 214, 215, 134, 158, 132, 223, 225, 225, 225, 193, 6, 3, 199, 4, 200, 2
]
_SPM_PATCH2_IMAGE_USER_TOKENS = [1, 3, 10, 10, 14, 10, 10, 15, 1032, 4]
_WITH_AUDIO_AUDIO_USER_TOKENS = [1, 3, 25, 24, 24, 4]
_WITH_AUDIO_TEXT_THEN_AUDIO_USER_TOKENS = [1, 3, 197, 25, 24, 24, 4]
_SPM_PATCH2_MULTI_IMAGE_USER_TOKENS = [1, 3, 10, 10, 14, 10, 10, 15, 10, 10, 14, 10, 10, 15, 4]
_SPM_PATCH2_MULTITURN_IMAGE_TOKENS = [1, 3, 10, 10, 14, 10, 10, 15, 4, 1055, 2, 3, 10, 10, 14, 10, 10, 15, 4, 1049, 2]
_WITH_AUDIO_MULTI_AUDIO_USER_TOKENS = [1, 3, 25, 24, 24, 25, 24, 24, 4]
_WITH_AUDIO_MULTITURN_AUDIO_TOKENS = [1, 3, 25, 24, 24, 4, 198, 2, 3, 25, 24, 24, 4, 200, 2]
# fmt: on

_SPM_NORMAL_EXPECTED = Tokenized(tokens=_SPM_NORMAL_TOKENS)
_SPM_TOOLS_SINGLETURN_EXPECTED = Tokenized(tokens=_SPM_TOOLS_SINGLETURN_TOKENS)
_SPM_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_SPM_TOOLS_MULTITURN_TOKENS)
_SPM_TOOL_MESSAGE_EXPECTED = Tokenized(tokens=_SPM_TOOL_MESSAGE_TOKENS)
_SPM_ASSISTANT_TOOL_CALLS_EXPECTED = Tokenized(tokens=_SPM_ASSISTANT_TOOL_CALLS_TOKENS)
_SPM_SYSTEM_MESSAGE_EXPECTED = Tokenized(tokens=_SPM_SYSTEM_MESSAGE_TOKENS)
_SPM_SYSTEM_MESSAGE_MULTITURN_EXPECTED = Tokenized(tokens=_SPM_SYSTEM_MESSAGE_MULTITURN_TOKENS)
_SPM_SYSTEM_MESSAGE_PREFIXED_FINAL_EXPECTED = Tokenized(
    tokens=_SPM_SYSTEM_MESSAGE_PREFIXED_FINAL_TOKENS, prefix_ids=[1049]
)
_SPM_SYSTEM_MESSAGE_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_SPM_SYSTEM_MESSAGE_TOOLS_MULTITURN_TOKENS)
_NO_AUDIO_NORMAL_EXPECTED = Tokenized(tokens=_NO_AUDIO_NORMAL_TOKENS)
_NO_AUDIO_TOOLS_SINGLETURN_EXPECTED = Tokenized(tokens=_NO_AUDIO_TOOLS_SINGLETURN_TOKENS)
_NO_AUDIO_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_NO_AUDIO_TOOLS_MULTITURN_TOKENS)
_NO_AUDIO_TOOL_MESSAGE_EXPECTED = Tokenized(tokens=_NO_AUDIO_TOOL_MESSAGE_TOKENS)
_NO_AUDIO_ASSISTANT_TOOL_CALLS_EXPECTED = Tokenized(tokens=_NO_AUDIO_ASSISTANT_TOOL_CALLS_TOKENS)
_NO_AUDIO_SYSTEM_MESSAGE_EXPECTED = Tokenized(tokens=_NO_AUDIO_SYSTEM_MESSAGE_TOKENS)
_NO_AUDIO_SYSTEM_MESSAGE_MULTITURN_EXPECTED = Tokenized(tokens=_NO_AUDIO_SYSTEM_MESSAGE_MULTITURN_TOKENS)
_NO_AUDIO_SYSTEM_MESSAGE_PREFIXED_FINAL_EXPECTED = Tokenized(
    tokens=_NO_AUDIO_SYSTEM_MESSAGE_PREFIXED_FINAL_TOKENS, prefix_ids=[200]
)
_NO_AUDIO_SYSTEM_MESSAGE_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_NO_AUDIO_SYSTEM_MESSAGE_TOOLS_MULTITURN_TOKENS)
_SILENT_AUDIO = Audio(audio_array=np.zeros(shape=2400, dtype=np.float32), sampling_rate=24_000, format="wav")
_SPM_PATCH2_IMAGE_USER_EXPECTED = Tokenized(tokens=_SPM_PATCH2_IMAGE_USER_TOKENS, images=[_RED_4X4_IMAGE])
_WITH_AUDIO_AUDIO_USER_EXPECTED = Tokenized(tokens=_WITH_AUDIO_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO])
_WITH_AUDIO_TEXT_THEN_AUDIO_USER_EXPECTED = Tokenized(
    tokens=_WITH_AUDIO_TEXT_THEN_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO]
)
_SPM_PATCH2_MULTI_IMAGE_USER_EXPECTED = Tokenized(
    tokens=_SPM_PATCH2_MULTI_IMAGE_USER_TOKENS, images=[_RED_4X4_IMAGE, _BLUE_4X4_IMAGE]
)
_SPM_PATCH2_MULTITURN_IMAGE_EXPECTED = Tokenized(
    tokens=_SPM_PATCH2_MULTITURN_IMAGE_TOKENS, images=[_RED_4X4_IMAGE, _RED_4X4_IMAGE]
)
_WITH_AUDIO_MULTI_AUDIO_USER_EXPECTED = Tokenized(
    tokens=_WITH_AUDIO_MULTI_AUDIO_USER_TOKENS, audios=[_SILENT_AUDIO, _SILENT_AUDIO]
)
_WITH_AUDIO_MULTITURN_AUDIO_EXPECTED = Tokenized(
    tokens=_WITH_AUDIO_MULTITURN_AUDIO_TOKENS, audios=[_SILENT_AUDIO, _SILENT_AUDIO]
)

_SPM_NORMAL_TEXT = "<s>[INST]▁a[/INST]▁b</s>[INST]▁c[/INST]▁d</s>"
_SPM_TOOLS_SINGLETURN_TEXT = (
    '<s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}]'
    "[/AVAILABLE_TOOLS][INST]▁a[/INST]"
)
_SPM_TOOLS_MULTITURN_TEXT = (
    '<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",'
    '▁"parameters":▁{}}},▁{"type":▁"function",▁"function":▁{"name":▁"tool2",▁"description":▁"2",▁"parameters":▁{}}}]'
    "[/AVAILABLE_TOOLS][INST]▁c[/INST]▁d</s>"
)
_SPM_TOOL_MESSAGE_TEXT = "<s>[INST]▁a[/INST][TOOL_RESULTS]▁123456789[TOOL_CONTENT]▁d[/TOOL_RESULTS]"
_SPM_ASSISTANT_TOOL_CALLS_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>'
)
_SPM_SYSTEM_MESSAGE_TEXT = "<s>[SYSTEM_PROMPT]▁s[/SYSTEM_PROMPT][INST]▁a[/INST]"
_SPM_SYSTEM_MESSAGE_MULTITURN_TEXT = (
    "<s>[SYSTEM_PROMPT]▁SYSTEM[/SYSTEM_PROMPT][INST]▁a[/INST]▁b</s>[INST]▁c[/INST]▁d</s>"
)
_SPM_SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT = (
    "<s>[SYSTEM_PROMPT]▁SYSTEM[/SYSTEM_PROMPT][INST]▁a[/INST]▁b</s>[INST]▁c[/INST]▁d"
)
_SPM_SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT = (
    '<s>[SYSTEM_PROMPT]▁SYSTEM[/SYSTEM_PROMPT][INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":'
    '▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}][/AVAILABLE_TOOLS][INST]▁c[/INST]▁d</s>'
)
_NO_AUDIO_NORMAL_TEXT = "<s>[INST]a[/INST]b</s>[INST]c[/INST]d</s>"
_NO_AUDIO_TOOLS_SINGLETURN_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}]'
    "[/AVAILABLE_TOOLS][INST]a[/INST]"
)
_NO_AUDIO_TOOLS_MULTITURN_TEXT = (
    '<s>[INST]a[/INST]b</s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", '
    '"parameters": {}}}, {"type": "function", "function": {"name": "tool2", "description": "2", "parameters": {}}}]'
    "[/AVAILABLE_TOOLS][INST]c[/INST]d</s>"
)
_NO_AUDIO_TOOL_MESSAGE_TEXT = "<s>[INST]a[/INST][TOOL_RESULTS]123456789[TOOL_CONTENT]d[/TOOL_RESULTS]"
_NO_AUDIO_ASSISTANT_TOOL_CALLS_TEXT = (
    '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "123456789"}]</s>'
)
_NO_AUDIO_SYSTEM_MESSAGE_TEXT = "<s>[SYSTEM_PROMPT]s[/SYSTEM_PROMPT][INST]a[/INST]"
_NO_AUDIO_SYSTEM_MESSAGE_MULTITURN_TEXT = (
    "<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT][INST]a[/INST]b</s>[INST]c[/INST]d</s>"
)
_NO_AUDIO_SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT = (
    "<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT][INST]a[/INST]b</s>[INST]c[/INST]d"
)
_NO_AUDIO_SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT = (
    '<s>[SYSTEM_PROMPT]SYSTEM[/SYSTEM_PROMPT][INST]a[/INST]b</s>[AVAILABLE_TOOLS][{"type": "function", "function": '
    '{"name": "tool1", "description": "1", "parameters": {}}}][/AVAILABLE_TOOLS][INST]c[/INST]d</s>'
)
_SPM_PATCH2_IMAGE_USER_TEXT = "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]▁a[/INST]"
_WITH_AUDIO_AUDIO_USER_TEXT = "<s>[INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_WITH_AUDIO_TEXT_THEN_AUDIO_USER_TEXT = "<s>[INST]a[BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_SPM_PATCH2_MULTI_IMAGE_USER_TEXT = (
    "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][/INST]"
)
_SPM_PATCH2_MULTITURN_IMAGE_TEXT = (
    "<s>[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][/INST]▁b</s>"
    "[INST][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][/INST]▁d</s>"
)
_WITH_AUDIO_MULTI_AUDIO_USER_TEXT = "<s>[INST][BEGIN_AUDIO][AUDIO][AUDIO][BEGIN_AUDIO][AUDIO][AUDIO][/INST]"
_WITH_AUDIO_MULTITURN_AUDIO_TEXT = (
    "<s>[INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]b</s>[INST][BEGIN_AUDIO][AUDIO][AUDIO][/INST]d</s>"
)

_INVALID_ASSISTANT_MESSAGE = r"^Invalid assistant message:"
_TRUNCATION_FAILED_MESSAGE = r"^Input couldn't fit in truncate_at_max_token$"
_THINK_NOT_IMPLEMENTED_MESSAGE = r"^Think not implemented for tokenizer < V13\.$"


@pytest.fixture(scope="module")
def spm_tokenizer() -> InstructTokenizer:
    r"""Provide the bundled v7 multimodal SentencePiece instruct tokenizer."""
    return MistralTokenizer.v7(is_mm=True).instruct_tokenizer


@pytest.fixture(scope="module")
def spm_patch2_tokenizer() -> InstructTokenizer:
    r"""Provide a separate bundled v7 multimodal instruct tokenizer with an image patch size of 2."""
    base = MistralTokenizer.v7(is_mm=True).instruct_tokenizer
    assert isinstance(base.image_encoder, ImageEncoder)
    patch2_encoder = ImageEncoder(
        image_config=ImageConfig(image_patch_size=2, max_image_size=1024),
        special_ids=base.image_encoder.special_ids,
    )
    return InstructTokenizerV7(tokenizer=base.tokenizer, image_encoder=patch2_encoder)


@pytest.fixture(scope="module")
def no_audio_tokenizer() -> InstructTokenizer:
    r"""Provide a v7 Tekken instruct tokenizer without audio.

    Synthetic formatting-only tokenizer; its IDs are not released-vocabulary evidence.
    """
    tokenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v7,
    )
    return InstructTokenizerV7(tokenizer=tokenizer)


@pytest.fixture(scope="module")
def with_audio_tokenizer() -> InstructTokenizer:
    r"""Provide a v7 Tekken instruct tokenizer with an audio encoder.

    Synthetic formatting-only tokenizer; its IDs are not released-vocabulary evidence.
    """
    return get_tekkenizer_with_audio()


def _build_red_image_chunk() -> ImageChunk:
    r"""Build an image chunk holding a red 4x4 image."""
    return ImageChunk(image=Image.new(mode="RGB", size=(4, 4), color="red"))


def _build_blue_image_chunk() -> ImageChunk:
    r"""Build an image chunk holding a blue 6x4 image."""
    return ImageChunk(image=Image.new(mode="RGB", size=(6, 4), color="blue"))


def _build_red_image_user_message() -> UserMessage:
    r"""Build a user message with the text `a` followed by a red 4x4 image."""
    return UserMessage(content=[TextChunk(text="a"), _build_red_image_chunk()])


def _build_image_user_request(content: list[ContentChunk]) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request holding a lone user message with the supplied chunks."""
    return InstructRequest[ChatMessage, Tool](messages=[UserMessage(content=content)])


def _build_string_properties_tool() -> Tool:
    r"""Build the tool `t` whose parameters are the string properties `g` and `h`."""
    return Tool(
        function=Function(
            name="t",
            parameters={
                "type": "object",
                "properties": {"g": {"type": "string"}, "h": {"type": "string"}},
            },
        )
    )


def _build_tool_call_with_argument() -> ToolCall:
    r"""Build a call to the tool `t` with the argument `g="h"`."""
    return ToolCall(function=FunctionCall(name="t", arguments='{"g": "h"}'))


def _build_two_tools() -> list[Tool]:
    r"""Build the tools `t1` and `t2`, both without parameters."""
    return [Tool(function=Function(name="t1", parameters={})), Tool(function=Function(name="t2", parameters={}))]


def _build_two_calls_assistant_message(prefix: bool) -> AssistantMessage:
    r"""Build an assistant message with content `b1b2` and calls to `t1` and `t2`."""
    return AssistantMessage(
        content="b1b2",
        prefix=prefix,
        tool_calls=[
            ToolCall(id="000000000", function=FunctionCall(name="t1", arguments="{}")),
            ToolCall(id="111111111", function=FunctionCall(name="t2", arguments="{}")),
        ],
    )


def _build_prefixed_assistant_request(
    content: str | None, tool_calls: list[ToolCall] | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message holding the supplied content and calls."""
    return InstructRequest[ChatMessage, Tool](
        messages=[AssistantMessage(content=content, tool_calls=tool_calls, prefix=True)]
    )


def _build_image_and_tool_result_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with an image user turn, a plain assistant turn and a tool result."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            _build_red_image_user_message(),
            AssistantMessage(content="b"),
            ToolMessage(tool_call_id="b", content="f"),
        ]
    )


def _build_prefixed_assistant_after_image_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose final assistant message is a prefix after an image user turn."""
    return InstructRequest[ChatMessage, Tool](
        messages=[_build_red_image_user_message(), AssistantMessage(content="b", prefix=True)]
    )


def _build_none_content_one_call_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of `None` content and one call."""
    return _build_prefixed_assistant_request(
        content=None, tool_calls=[ToolCall(function=FunctionCall(name="test_fn", arguments="{}"))]
    )


def _build_empty_content_one_call_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of empty content and one call."""
    return _build_prefixed_assistant_request(
        content="", tool_calls=[ToolCall(function=FunctionCall(name="test_fn", arguments="{}"))]
    )


def _build_system_tools_and_call_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a system message, a tool and an assistant turn calling it."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=[_build_string_properties_tool()],
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="a"),
            AssistantMessage(content="b", tool_calls=[_build_tool_call_with_argument()]),
        ],
    )


def _build_tool_result_after_history_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a tool result after consecutive user turns and a tool call."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=[_build_string_properties_tool()],
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="a"),
            UserMessage(content="c"),
            AssistantMessage(content="b", tool_calls=[_build_tool_call_with_argument()]),
            ToolMessage(content="b", tool_call_id="1234"),
        ],
    )


def _build_tool_call_and_content_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with an assistant turn combining content and two tool calls."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=_build_two_tools(),
        messages=[UserMessage(content="a"), _build_two_calls_assistant_message(prefix=False)],
    )


def _build_prefixed_tool_call_and_content_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose final prefixed assistant turn combines content and two tool calls."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=_build_two_tools(),
        messages=[UserMessage(content="a"), _build_two_calls_assistant_message(prefix=True)],
    )


def _build_tool_calls_with_two_results_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with an assistant turn of two tool calls followed by one result per call."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=_build_two_tools(),
        messages=[
            UserMessage(content="a"),
            _build_two_calls_assistant_message(prefix=False),
            ToolMessage(content="r1", tool_call_id="000000000"),
            ToolMessage(content="r2", tool_call_id="111111111"),
        ],
    )


def _build_truncation_keep_sys_and_last_message_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request where, next to the system prompts, only the last user message fits."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="c"),
            UserMessage(content="c"),
            SystemMessage(content="a"),
            UserMessage(content="bbbbbbb"),
        ],
        truncate_at_max_tokens=15,
    )


def _build_truncation_drop_first_assistant_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request where dropping the first assistant message makes everything else fit."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            AssistantMessage(content="c"),
            UserMessage(content="b"),
            UserMessage(content="a"),
            UserMessage(content="aaaaaaa"),
        ],
        truncate_at_max_tokens=15,
    )


def _build_truncation_start_with_assistant_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request whose truncated result may start with an assistant message because the input did."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            AssistantMessage(content="c"),
            AssistantMessage(content="b"),
            UserMessage(content="a"),
            UserMessage(content="aaaaaaa"),
        ],
        truncate_at_max_tokens=15,
    )


def _build_truncation_drop_by_chunk_1_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request where the first assistant message, then the user and tool pair, are dropped."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            AssistantMessage(content="c"),
            UserMessage(content="c"),
            ToolMessage(content="c", tool_call_id="1234"),
            UserMessage(content="a"),
            AssistantMessage(content="bbbbbbb"),
        ],
        truncate_at_max_tokens=15,
    )


def _build_truncation_drop_by_chunk_2_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request where only the last message is kept because the first chunk of three is too big."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="c"),
            AssistantMessage(content="c"),
            AssistantMessage(content="c"),
            UserMessage(content="aaaaaaa"),
        ],
        truncate_at_max_tokens=15,
    )


def _build_truncation_full_convo_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request over a conversation with several system messages, truncated at its start."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="c"),
            AssistantMessage(content="c"),
            UserMessage(content="a"),
            AssistantMessage(content="a"),
            SystemMessage(content="b"),
            UserMessage(content="a"),
        ],
        truncate_at_max_tokens=15,
    )


def _build_empty_text_then_two_images_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with an empty text chunk, a red 4x4 image and a blue 6x4 image."""
    return _build_image_user_request(content=[TextChunk(text=""), _build_red_image_chunk(), _build_blue_image_chunk()])


def _build_text_then_two_images_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a text chunk, a red 4x4 image and a blue 6x4 image."""
    return _build_image_user_request(content=[TextChunk(text="x"), _build_red_image_chunk(), _build_blue_image_chunk()])


def _build_two_images_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a red 4x4 image followed by a blue 6x4 image."""
    return _build_image_user_request(content=[_build_red_image_chunk(), _build_blue_image_chunk()])


def _build_trailing_image_moves_first_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single image follows a text chunk."""
    return _build_image_user_request(content=[TextChunk(text="x"), _build_red_image_chunk()])


def _build_leading_image_remains_first_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single image precedes a text chunk."""
    return _build_image_user_request(content=[_build_red_image_chunk(), TextChunk(text="x")])


def _build_system_list_content_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose system message holds a single text chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content=[TextChunk(text="s")]), UserMessage(content="a")]
    )


def _build_assistant_list_content_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose assistant message holds a single text chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), AssistantMessage(content=[TextChunk(text="b")])]
    )


def _build_system_and_user_words_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with multi-word system and user message content."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content="hello world"), UserMessage(content="good day")]
    )


def _build_truncation_last_user_at_limit_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a one-message request whose tokens, BOS excluded, equal the limit."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a" * 7)],
        truncate_at_max_tokens=9,
    )


def _build_none_content_no_calls_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of `None` content and no calls."""
    return _build_prefixed_assistant_request(content=None, tool_calls=None)


def _build_none_content_empty_calls_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of `None` content and an empty call list."""
    return _build_prefixed_assistant_request(content=None, tool_calls=[])


def _build_empty_content_no_calls_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of empty content and no calls."""
    return _build_prefixed_assistant_request(content="", tool_calls=None)


def _build_empty_content_empty_calls_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of empty content and an empty call list."""
    return _build_prefixed_assistant_request(content="", tool_calls=[])


def _build_truncation_system_prompt_too_long_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 9-token request whose only message is a system prompt that does not fit."""
    return InstructRequest[ChatMessage, Tool](messages=[SystemMessage(content="a" * 10)], truncate_at_max_tokens=9)


def _build_truncation_last_user_too_long_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 9-token request whose only message is a last user message that does not fit."""
    return InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="a" * 10)], truncate_at_max_tokens=9)


def _build_truncation_last_user_one_over_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a one-message request whose tokens, BOS excluded, exceed the limit by one."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a" * 8)],
        truncate_at_max_tokens=9,
    )


def _build_assistant_think_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose assistant message contains a think chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content=[ThinkChunk(thinking="t"), TextChunk(text="b")]),
        ]
    )


@pytest.mark.parametrize(
    argnames="config, request_builder, expected, expected_text",
    argvalues=[
        pytest.param(
            "spm_patch2",
            _build_image_and_tool_result_request,
            _IMAGE_AND_TOOL_RESULT_EXPECTED,
            _IMAGE_AND_TOOL_RESULT_TEXT,
            id="spm-patch2-image-and-tool-result",
        ),
        pytest.param(
            "spm_patch2",
            _build_prefixed_assistant_after_image_request,
            _PREFIXED_ASSISTANT_AFTER_IMAGE_EXPECTED,
            _PREFIXED_ASSISTANT_AFTER_IMAGE_TEXT,
            id="spm-patch2-prefixed-assistant-after-image",
        ),
        pytest.param(
            "spm",
            _build_none_content_one_call_request,
            _PREFIXED_CALL_WITHOUT_CONTENT_EXPECTED,
            _PREFIXED_CALL_WITHOUT_CONTENT_TEXT,
            id="spm-none-content-one-call",
        ),
        pytest.param(
            "spm",
            _build_empty_content_one_call_request,
            _PREFIXED_CALL_WITHOUT_CONTENT_EXPECTED,
            _PREFIXED_CALL_WITHOUT_CONTENT_TEXT,
            id="spm-empty-content-one-call",
        ),
        pytest.param(
            "spm",
            _build_system_tools_and_call_request,
            _SYSTEM_TOOLS_AND_CALL_EXPECTED,
            _SYSTEM_TOOLS_AND_CALL_TEXT,
            id="spm-system-tools-and-call",
        ),
        pytest.param(
            "spm",
            _build_tool_result_after_history_request,
            _TOOL_RESULT_AFTER_HISTORY_EXPECTED,
            _TOOL_RESULT_AFTER_HISTORY_TEXT,
            id="spm-tool-result-after-history",
        ),
        pytest.param(
            "no_audio",
            _build_tool_call_and_content_request,
            _TOOL_CALL_AND_CONTENT_EXPECTED,
            _TOOL_CALL_AND_CONTENT_TEXT,
            id="no-audio-tool-call-and-content",
        ),
        pytest.param(
            "with_audio",
            _build_tool_call_and_content_request,
            _TOOL_CALL_AND_CONTENT_EXPECTED,
            _TOOL_CALL_AND_CONTENT_TEXT,
            id="with-audio-tool-call-and-content",
        ),
        pytest.param(
            "no_audio",
            _build_prefixed_tool_call_and_content_request,
            _PREFIXED_TOOL_CALL_AND_CONTENT_EXPECTED,
            _PREFIXED_TOOL_CALL_AND_CONTENT_TEXT,
            id="no-audio-prefixed-tool-call-and-content",
        ),
        pytest.param(
            "with_audio",
            _build_prefixed_tool_call_and_content_request,
            _PREFIXED_TOOL_CALL_AND_CONTENT_EXPECTED,
            _PREFIXED_TOOL_CALL_AND_CONTENT_TEXT,
            id="with-audio-prefixed-tool-call-and-content",
        ),
        pytest.param(
            "no_audio",
            _build_tool_calls_with_two_results_request,
            _TOOL_CALLS_WITH_TWO_RESULTS_EXPECTED,
            _TOOL_CALLS_WITH_TWO_RESULTS_TEXT,
            id="no-audio-tool-calls-with-two-results",
        ),
        pytest.param(
            "with_audio",
            _build_tool_calls_with_two_results_request,
            _TOOL_CALLS_WITH_TWO_RESULTS_EXPECTED,
            _TOOL_CALLS_WITH_TWO_RESULTS_TEXT,
            id="with-audio-tool-calls-with-two-results",
        ),
        pytest.param(
            "no_audio",
            _build_truncation_keep_sys_and_last_message_request,
            _TRUNCATION_KEEP_SYS_AND_LAST_MESSAGE_EXPECTED,
            _TRUNCATION_KEEP_SYS_AND_LAST_MESSAGE_TEXT,
            id="no-audio-truncation-keep-sys-and-last-message",
        ),
        pytest.param(
            "with_audio",
            _build_truncation_keep_sys_and_last_message_request,
            _TRUNCATION_KEEP_SYS_AND_LAST_MESSAGE_EXPECTED,
            _TRUNCATION_KEEP_SYS_AND_LAST_MESSAGE_TEXT,
            id="with-audio-truncation-keep-sys-and-last-message",
        ),
        pytest.param(
            "no_audio",
            _build_truncation_drop_first_assistant_request,
            _TRUNCATION_DROP_FIRST_ASSISTANT_EXPECTED,
            _TRUNCATION_DROP_FIRST_ASSISTANT_TEXT,
            id="no-audio-truncation-drop-first-assistant",
        ),
        pytest.param(
            "with_audio",
            _build_truncation_drop_first_assistant_request,
            _TRUNCATION_DROP_FIRST_ASSISTANT_EXPECTED,
            _TRUNCATION_DROP_FIRST_ASSISTANT_TEXT,
            id="with-audio-truncation-drop-first-assistant",
        ),
        pytest.param(
            "no_audio",
            _build_truncation_start_with_assistant_request,
            _TRUNCATION_START_WITH_ASSISTANT_EXPECTED,
            _TRUNCATION_START_WITH_ASSISTANT_TEXT,
            id="no-audio-truncation-start-with-assistant",
        ),
        pytest.param(
            "with_audio",
            _build_truncation_start_with_assistant_request,
            _TRUNCATION_START_WITH_ASSISTANT_EXPECTED,
            _TRUNCATION_START_WITH_ASSISTANT_TEXT,
            id="with-audio-truncation-start-with-assistant",
        ),
        pytest.param(
            "no_audio",
            _build_truncation_drop_by_chunk_1_request,
            _TRUNCATION_DROP_BY_CHUNK_1_EXPECTED,
            _TRUNCATION_DROP_BY_CHUNK_1_TEXT,
            id="no-audio-truncation-drop-by-chunk-1",
        ),
        pytest.param(
            "with_audio",
            _build_truncation_drop_by_chunk_1_request,
            _TRUNCATION_DROP_BY_CHUNK_1_EXPECTED,
            _TRUNCATION_DROP_BY_CHUNK_1_TEXT,
            id="with-audio-truncation-drop-by-chunk-1",
        ),
        pytest.param(
            "no_audio",
            _build_truncation_drop_by_chunk_2_request,
            _TRUNCATION_DROP_BY_CHUNK_2_EXPECTED,
            _TRUNCATION_DROP_BY_CHUNK_2_TEXT,
            id="no-audio-truncation-drop-by-chunk-2",
        ),
        pytest.param(
            "with_audio",
            _build_truncation_drop_by_chunk_2_request,
            _TRUNCATION_DROP_BY_CHUNK_2_EXPECTED,
            _TRUNCATION_DROP_BY_CHUNK_2_TEXT,
            id="with-audio-truncation-drop-by-chunk-2",
        ),
        pytest.param(
            "no_audio",
            _build_truncation_full_convo_request,
            _TRUNCATION_FULL_CONVO_EXPECTED,
            _TRUNCATION_FULL_CONVO_TEXT,
            id="no-audio-truncation-full-convo",
        ),
        pytest.param(
            "with_audio",
            _build_truncation_full_convo_request,
            _TRUNCATION_FULL_CONVO_EXPECTED,
            _TRUNCATION_FULL_CONVO_TEXT,
            id="with-audio-truncation-full-convo",
        ),
        pytest.param(
            "spm_patch2",
            _build_empty_text_then_two_images_request,
            _TWO_IMAGES_EXPECTED,
            _TWO_IMAGES_TEXT,
            id="spm-patch2-empty-text-then-two-images",
        ),
        pytest.param(
            "spm_patch2",
            _build_text_then_two_images_request,
            _TEXT_THEN_TWO_IMAGES_EXPECTED,
            _TEXT_THEN_TWO_IMAGES_TEXT,
            id="spm-patch2-text-then-two-images",
        ),
        pytest.param(
            "spm_patch2",
            _build_two_images_request,
            _TWO_IMAGES_EXPECTED,
            _TWO_IMAGES_TEXT,
            id="spm-patch2-two-images",
        ),
        pytest.param(
            "spm_patch2",
            _build_trailing_image_moves_first_request,
            _ONE_IMAGE_WITH_TEXT_EXPECTED,
            _ONE_IMAGE_WITH_TEXT_TEXT,
            id="spm-patch2-trailing-image-moves-first",
        ),
        pytest.param(
            "spm_patch2",
            _build_leading_image_remains_first_request,
            _ONE_IMAGE_WITH_TEXT_EXPECTED,
            _ONE_IMAGE_WITH_TEXT_TEXT,
            id="spm-patch2-leading-image-remains-first",
        ),
        pytest.param(
            "spm",
            _build_system_list_content_request,
            _SYSTEM_LIST_CONTENT_EXPECTED,
            _SYSTEM_LIST_CONTENT_TEXT,
            id="spm-system-list-content",
        ),
        pytest.param(
            "spm",
            _build_assistant_list_content_request,
            _ASSISTANT_LIST_CONTENT_EXPECTED,
            _ASSISTANT_LIST_CONTENT_TEXT,
            id="spm-assistant-list-content",
        ),
        pytest.param(
            "spm",
            _build_system_and_user_words_request,
            _SYSTEM_AND_USER_WORDS_EXPECTED,
            _SYSTEM_AND_USER_WORDS_TEXT,
            id="spm-system-and-user-words",
        ),
        pytest.param(
            "no_audio",
            _build_truncation_last_user_at_limit_request,
            _TRUNCATION_LAST_USER_AT_LIMIT_EXPECTED,
            _TRUNCATION_LAST_USER_AT_LIMIT_TEXT,
            id="no-audio-truncation-last-user-at-limit",
        ),
        pytest.param(
            "with_audio",
            _build_truncation_last_user_at_limit_request,
            _TRUNCATION_LAST_USER_AT_LIMIT_EXPECTED,
            _TRUNCATION_LAST_USER_AT_LIMIT_TEXT,
            id="with-audio-truncation-last-user-at-limit",
        ),
        pytest.param(
            "spm",
            partial(build_normal_request, settings=None),
            _SPM_NORMAL_EXPECTED,
            _SPM_NORMAL_TEXT,
            id="spm-shared-normal",
        ),
        pytest.param(
            "spm",
            partial(build_tools_singleturn_request, settings=None),
            _SPM_TOOLS_SINGLETURN_EXPECTED,
            _SPM_TOOLS_SINGLETURN_TEXT,
            id="spm-shared-tools-singleturn",
        ),
        pytest.param(
            "spm",
            partial(build_tools_multiturn_request, settings=None),
            _SPM_TOOLS_MULTITURN_EXPECTED,
            _SPM_TOOLS_MULTITURN_TEXT,
            id="spm-shared-tools-multiturn",
        ),
        pytest.param(
            "spm",
            partial(build_tool_message_request, settings=None),
            _SPM_TOOL_MESSAGE_EXPECTED,
            _SPM_TOOL_MESSAGE_TEXT,
            id="spm-shared-tool-message",
        ),
        pytest.param(
            "spm",
            partial(build_assistant_tool_calls_request, settings=None),
            _SPM_ASSISTANT_TOOL_CALLS_EXPECTED,
            _SPM_ASSISTANT_TOOL_CALLS_TEXT,
            id="spm-shared-assistant-tool-calls",
        ),
        pytest.param(
            "spm",
            partial(build_system_message_request, settings=None),
            _SPM_SYSTEM_MESSAGE_EXPECTED,
            _SPM_SYSTEM_MESSAGE_TEXT,
            id="spm-shared-system-message",
        ),
        pytest.param(
            "spm",
            partial(build_system_message_multiturn_request, settings=None),
            _SPM_SYSTEM_MESSAGE_MULTITURN_EXPECTED,
            _SPM_SYSTEM_MESSAGE_MULTITURN_TEXT,
            id="spm-shared-system-message-multiturn",
        ),
        pytest.param(
            "spm",
            partial(build_system_message_prefixed_final_request, settings=None),
            _SPM_SYSTEM_MESSAGE_PREFIXED_FINAL_EXPECTED,
            _SPM_SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT,
            id="spm-shared-system-message-prefixed-final",
        ),
        pytest.param(
            "spm",
            partial(build_system_message_tools_multiturn_request, settings=None),
            _SPM_SYSTEM_MESSAGE_TOOLS_MULTITURN_EXPECTED,
            _SPM_SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT,
            id="spm-shared-system-message-tools-multiturn",
        ),
        pytest.param(
            "no_audio",
            partial(build_normal_request, settings=None),
            _NO_AUDIO_NORMAL_EXPECTED,
            _NO_AUDIO_NORMAL_TEXT,
            id="no-audio-shared-normal",
        ),
        pytest.param(
            "no_audio",
            partial(build_tools_singleturn_request, settings=None),
            _NO_AUDIO_TOOLS_SINGLETURN_EXPECTED,
            _NO_AUDIO_TOOLS_SINGLETURN_TEXT,
            id="no-audio-shared-tools-singleturn",
        ),
        pytest.param(
            "no_audio",
            partial(build_tools_multiturn_request, settings=None),
            _NO_AUDIO_TOOLS_MULTITURN_EXPECTED,
            _NO_AUDIO_TOOLS_MULTITURN_TEXT,
            id="no-audio-shared-tools-multiturn",
        ),
        pytest.param(
            "no_audio",
            partial(build_tool_message_request, settings=None),
            _NO_AUDIO_TOOL_MESSAGE_EXPECTED,
            _NO_AUDIO_TOOL_MESSAGE_TEXT,
            id="no-audio-shared-tool-message",
        ),
        pytest.param(
            "no_audio",
            partial(build_assistant_tool_calls_request, settings=None),
            _NO_AUDIO_ASSISTANT_TOOL_CALLS_EXPECTED,
            _NO_AUDIO_ASSISTANT_TOOL_CALLS_TEXT,
            id="no-audio-shared-assistant-tool-calls",
        ),
        pytest.param(
            "no_audio",
            partial(build_system_message_request, settings=None),
            _NO_AUDIO_SYSTEM_MESSAGE_EXPECTED,
            _NO_AUDIO_SYSTEM_MESSAGE_TEXT,
            id="no-audio-shared-system-message",
        ),
        pytest.param(
            "no_audio",
            partial(build_system_message_multiturn_request, settings=None),
            _NO_AUDIO_SYSTEM_MESSAGE_MULTITURN_EXPECTED,
            _NO_AUDIO_SYSTEM_MESSAGE_MULTITURN_TEXT,
            id="no-audio-shared-system-message-multiturn",
        ),
        pytest.param(
            "no_audio",
            partial(build_system_message_prefixed_final_request, settings=None),
            _NO_AUDIO_SYSTEM_MESSAGE_PREFIXED_FINAL_EXPECTED,
            _NO_AUDIO_SYSTEM_MESSAGE_PREFIXED_FINAL_TEXT,
            id="no-audio-shared-system-message-prefixed-final",
        ),
        pytest.param(
            "no_audio",
            partial(build_system_message_tools_multiturn_request, settings=None),
            _NO_AUDIO_SYSTEM_MESSAGE_TOOLS_MULTITURN_EXPECTED,
            _NO_AUDIO_SYSTEM_MESSAGE_TOOLS_MULTITURN_TEXT,
            id="no-audio-shared-system-message-tools-multiturn",
        ),
        pytest.param(
            "spm_patch2",
            partial(build_image_user_request, settings=None),
            _SPM_PATCH2_IMAGE_USER_EXPECTED,
            _SPM_PATCH2_IMAGE_USER_TEXT,
            id="spm-patch2-shared-image-user",
        ),
        pytest.param(
            "spm_patch2",
            partial(build_text_then_image_user_request, settings=None),
            _SPM_PATCH2_IMAGE_USER_EXPECTED,
            _SPM_PATCH2_IMAGE_USER_TEXT,
            id="spm-patch2-shared-text-then-image-user",
        ),
        pytest.param(
            "with_audio",
            partial(build_audio_user_request, settings=None),
            _WITH_AUDIO_AUDIO_USER_EXPECTED,
            _WITH_AUDIO_AUDIO_USER_TEXT,
            id="with-audio-shared-audio-user",
        ),
        pytest.param(
            "with_audio",
            partial(build_text_then_audio_user_request, settings=None),
            _WITH_AUDIO_TEXT_THEN_AUDIO_USER_EXPECTED,
            _WITH_AUDIO_TEXT_THEN_AUDIO_USER_TEXT,
            id="with-audio-shared-text-then-audio-user",
        ),
        pytest.param(
            "spm_patch2",
            partial(build_multi_image_user_request, settings=None),
            _SPM_PATCH2_MULTI_IMAGE_USER_EXPECTED,
            _SPM_PATCH2_MULTI_IMAGE_USER_TEXT,
            id="spm-patch2-shared-multi-image-user",
        ),
        pytest.param(
            "spm_patch2",
            partial(build_multiturn_image_request, settings=None),
            _SPM_PATCH2_MULTITURN_IMAGE_EXPECTED,
            _SPM_PATCH2_MULTITURN_IMAGE_TEXT,
            id="spm-patch2-shared-multiturn-image",
        ),
        pytest.param(
            "with_audio",
            partial(build_multi_audio_user_request, settings=None),
            _WITH_AUDIO_MULTI_AUDIO_USER_EXPECTED,
            _WITH_AUDIO_MULTI_AUDIO_USER_TEXT,
            id="with-audio-shared-multi-audio-user",
        ),
        pytest.param(
            "with_audio",
            partial(build_multiturn_audio_request, settings=None),
            _WITH_AUDIO_MULTITURN_AUDIO_EXPECTED,
            _WITH_AUDIO_MULTITURN_AUDIO_TEXT,
            id="with-audio-shared-multiturn-audio",
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
    r"""Preserve complete v7 instruct encodings for each tokenizer configuration and request."""
    tokenizer: InstructTokenizer = request.getfixturevalue(argname=f"{config}_tokenizer")
    tokenized = tokenizer.encode_instruct(request=request_builder())
    assert_tokenized_equal(actual=tokenized, expected=expected)
    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text


@pytest.mark.parametrize(
    argnames="config, request_builder, expected_exception, expected_message",
    argvalues=[
        pytest.param(
            "spm",
            _build_none_content_no_calls_request,
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="spm-none-content-no-calls",
        ),
        pytest.param(
            "spm",
            _build_none_content_empty_calls_request,
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="spm-none-content-empty-calls",
        ),
        pytest.param(
            "spm",
            _build_empty_content_no_calls_request,
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="spm-empty-content-no-calls",
        ),
        pytest.param(
            "spm",
            _build_empty_content_empty_calls_request,
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="spm-empty-content-empty-calls",
        ),
        pytest.param(
            "no_audio",
            _build_truncation_system_prompt_too_long_request,
            TokenizerException,
            _TRUNCATION_FAILED_MESSAGE,
            id="no-audio-truncation-system-prompt-too-long",
        ),
        pytest.param(
            "with_audio",
            _build_truncation_system_prompt_too_long_request,
            TokenizerException,
            _TRUNCATION_FAILED_MESSAGE,
            id="with-audio-truncation-system-prompt-too-long",
        ),
        pytest.param(
            "no_audio",
            _build_truncation_last_user_too_long_request,
            TokenizerException,
            _TRUNCATION_FAILED_MESSAGE,
            id="no-audio-truncation-last-user-too-long",
        ),
        pytest.param(
            "with_audio",
            _build_truncation_last_user_too_long_request,
            TokenizerException,
            _TRUNCATION_FAILED_MESSAGE,
            id="with-audio-truncation-last-user-too-long",
        ),
        pytest.param(
            "no_audio",
            _build_truncation_last_user_one_over_request,
            TokenizerException,
            _TRUNCATION_FAILED_MESSAGE,
            id="no-audio-truncation-last-user-one-over",
        ),
        pytest.param(
            "with_audio",
            _build_truncation_last_user_one_over_request,
            TokenizerException,
            _TRUNCATION_FAILED_MESSAGE,
            id="with-audio-truncation-last-user-one-over",
        ),
        pytest.param(
            "spm",
            _build_assistant_think_request,
            TokenizerException,
            _THINK_NOT_IMPLEMENTED_MESSAGE,
            id="spm-assistant-think",
        ),
        pytest.param(
            "spm",
            partial(build_empty_assistant_request, settings=None),
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="spm-shared-empty-assistant",
        ),
        pytest.param(
            "no_audio",
            partial(build_empty_assistant_request, settings=None),
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="no-audio-shared-empty-assistant",
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
    r"""Preserve direct v7 exception types and specified messages."""
    tokenizer: InstructTokenizer = request.getfixturevalue(argname=f"{config}_tokenizer")
    instruct_request = request_builder()

    with pytest.raises(expected_exception=expected_exception, match=expected_message):
        tokenizer.encode_instruct(request=instruct_request)

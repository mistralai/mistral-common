"""Direct formatting tests for both bundled v3 instruct backends."""

from collections.abc import Callable

import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.chunk import ContentChunk, TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import InstructTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils import decode_keep

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
# The begin-instruction token (ID 3) directly follows the end-tool token.
_SPM_SYSTEM_TOOLS_MULTITURN_TOKENS = [
    1, 3, 1032, 4, 1055, 2, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032,
    1113, 7230, 29508, 1316, 1113, 7286, 2032, 1113, 29508, 1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3,
    17889, 23294, 781, 781, 29485, 4, 1049, 2
]
# The begin-instruction token (ID 3) directly follows the end-tool token.
_TEKKEN_SYSTEM_TOOLS_MULTITURN_TOKENS = [
    1, 3, 1097, 4, 1098, 2, 5, 1091, 19227, 4994, 2811, 1429, 5165, 1897, 1429, 5165, 2811, 16753, 2391, 2811,
    1429, 71440, 1049, 1897, 1429, 14653, 2811, 1429, 1049, 1897, 1429, 26204, 2811, 1445, 2821, 27028, 6, 3,
    101289, 58343, 1267, 1099, 4, 1100, 2
]
_SPM_PREFIXED_FINAL_MESSAGE_TOKENS = [1, 3, 1032, 4, 1055, 2, 3, 17889, 23294, 781, 781, 29485, 4, 1049]
_SPM_PREFIXED_FINAL_MESSAGE_PREFIX_IDS = [1049]
_TEKKEN_PREFIXED_FINAL_MESSAGE_TOKENS = [1, 3, 1097, 4, 1098, 2, 3, 101289, 58343, 1267, 1099, 4, 1100]
_TEKKEN_PREFIXED_FINAL_MESSAGE_PREFIX_IDS = [1100]
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
# Fine-tuning accepts a tool call as the last message; an absent ID must not serialize as "null".
_SPM_TOOL_CALL_WITHOUT_ID_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 1139, 1743, 29561, 2
]
# Fine-tuning accepts a tool call as the last message; an absent ID must not serialize as "null".
_TEKKEN_TOOL_CALL_WITHOUT_ID_TOKENS = [
    1, 3, 1097, 4, 9, 1091, 19227, 2391, 2811, 1429, 1098, 1897, 1429, 61906, 2811, 1445, 2821, 1093, 2
]
# fmt: on

_TOOL_CALL_AND_CONTENT_MESSAGE = "Cannot have tool calls and content defined in the same assistant message"
_INVALID_ASSISTANT_MESSAGE = r"^Invalid assistant message:"

_SPM_TOOLS_SINGLETURN_TEXT = (
    '<s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":'
    '▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}][/AVAILABLE_TOOLS][INST]▁a[/INST]'
)
_SPM_TOOLS_SINGLETURN_RAW_DECODE = (
    '[{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}] a'
)
_TEKKEN_TOOLS_SINGLETURN_TEXT = (
    '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", '
    '"description": "1", "parameters": {}}}][/AVAILABLE_TOOLS][INST]a[/INST]'
)
_TEKKEN_TOOLS_SINGLETURN_RAW_DECODE = (
    '[{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}]a'
)
_SPM_TOOLS_MULTITURN_TEXT = (
    "<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁["
    '{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}},'
    '▁{"type":▁"function",▁"function":▁{"name":▁"tool2",▁"description":▁"2",▁"parameters":▁{}}}]'
    "[/AVAILABLE_TOOLS][INST]▁c[/INST]▁d</s>"
)
_SPM_TOOLS_MULTITURN_RAW_DECODE = (
    'a b [{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}, '
    '{"type": "function", "function": {"name": "tool2", "description": "2", "parameters": {}}}] c d'
)
_TEKKEN_TOOLS_MULTITURN_TEXT = (
    "<s>[INST]a[/INST]b</s>[AVAILABLE_TOOLS]["
    '{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}, '
    '{"type": "function", "function": {"name": "tool2", "description": "2", "parameters": {}}}]'
    "[/AVAILABLE_TOOLS][INST]c[/INST]d</s>"
)
_TEKKEN_TOOLS_MULTITURN_RAW_DECODE = (
    'ab[{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}, '
    '{"type": "function", "function": {"name": "tool2", "description": "2", "parameters": {}}}]cd'
)
_SPM_SYSTEM_TOOLS_MULTITURN_TEXT = (
    "<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁["
    '{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}'
    "][/AVAILABLE_TOOLS][INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d</s>"
)
_SPM_SYSTEM_TOOLS_MULTITURN_RAW_DECODE = (
    'a b [{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}] SYSTEM\n\nc d'
)
_TEKKEN_SYSTEM_TOOLS_MULTITURN_TEXT = (
    "<s>[INST]a[/INST]b</s>[AVAILABLE_TOOLS]["
    '{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}'
    "][/AVAILABLE_TOOLS][INST]SYSTEM\n\nc[/INST]d</s>"
)
_TEKKEN_SYSTEM_TOOLS_MULTITURN_RAW_DECODE = (
    'ab[{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}]SYSTEM\n\ncd'
)
_SPM_PREFIXED_FINAL_MESSAGE_TEXT = "<s>[INST]▁a[/INST]▁b</s>[INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d"
_SPM_PREFIXED_FINAL_MESSAGE_RAW_DECODE = "a b SYSTEM\n\nc d"
_TEKKEN_PREFIXED_FINAL_MESSAGE_TEXT = "<s>[INST]a[/INST]b</s>[INST]SYSTEM\n\nc[/INST]d"
_TEKKEN_PREFIXED_FINAL_MESSAGE_RAW_DECODE = "abSYSTEM\n\ncd"
_SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"0"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"0"}[/TOOL_RESULTS]▁e</s>[INST]▁f[/INST]'
    '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"1"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"1"}[/TOOL_RESULTS]'
)
_SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_RAW_DECODE = (
    'a [{"name": "b", "arguments": {}, "id": "0"}] '
    '{"content": "d", "call_id": "0"} e f [{"name": "b", "arguments": {}, "id": "1"}] '
    '{"content": "d", "call_id": "1"}'
)
_TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TEXT = (
    '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "0"}]</s>'
    '[TOOL_RESULTS]{"content": "d", "call_id": "0"}[/TOOL_RESULTS]e</s>[INST]f[/INST]'
    '[TOOL_CALLS][{"name": "b", "arguments": {}, "id": "1"}]</s>'
    '[TOOL_RESULTS]{"content": "d", "call_id": "1"}[/TOOL_RESULTS]'
)
_TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_RAW_DECODE = (
    'a[{"name": "b", "arguments": {}, "id": "0"}]'
    '{"content": "d", "call_id": "0"}ef[{"name": "b", "arguments": {}, "id": "1"}]'
    '{"content": "d", "call_id": "1"}'
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
_SPM_TOOL_MESSAGE_MULTIPLE_CALLS_RAW_DECODE = (
    'a [{"name": "b", "arguments": {}, "id": "0"}, {"name": "q", "arguments": {}, "id": "1"}] '
    '{"content": "d", "call_id": "0"} {"content": "d", "call_id": "1"} e f '
    '[{"name": "b", "arguments": {}, "id": "2"}, {"name": "q", "arguments": {}, "id": "3"}] '
    '{"content": "d", "call_id": "2"} {"content": "d", "call_id": "3"}'
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
_TEKKEN_TOOL_MESSAGE_MULTIPLE_CALLS_RAW_DECODE = (
    'a[{"name": "b", "arguments": {}, "id": "0"}, {"name": "q", "arguments": {}, "id": "1"}]'
    '{"content": "d", "call_id": "0"}{"content": "d", "call_id": "1"}ef'
    '[{"name": "b", "arguments": {}, "id": "2"}, {"name": "q", "arguments": {}, "id": "3"}]'
    '{"content": "d", "call_id": "2"}{"content": "d", "call_id": "3"}'
)
_SPM_TOOL_MESSAGE_PLAIN_STRING_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"123456789"}[/TOOL_RESULTS]'
)
_SPM_TOOL_MESSAGE_PLAIN_STRING_RAW_DECODE = (
    'a [{"name": "b", "arguments": {}, "id": "123456789"}] {"content": "d", "call_id": "123456789"}'
)
_TEKKEN_TOOL_MESSAGE_PLAIN_STRING_TEXT = (
    '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "123456789"}]</s>'
    '[TOOL_RESULTS]{"content": "d", "call_id": "123456789"}[/TOOL_RESULTS]'
)
_TEKKEN_TOOL_MESSAGE_PLAIN_STRING_RAW_DECODE = (
    'a[{"name": "b", "arguments": {}, "id": "123456789"}]{"content": "d", "call_id": "123456789"}'
)
_SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>'
    '[TOOL_RESULTS]▁{"content":▁{"a":▁1},▁"call_id":▁"123456789"}[/TOOL_RESULTS]'
)
_SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_RAW_DECODE = (
    'a [{"name": "b", "arguments": {}, "id": "123456789"}] {"content": {"a": 1}, "call_id": "123456789"}'
)
_TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_TEXT = (
    '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "123456789"}]</s>'
    '[TOOL_RESULTS]{"content": {"a": 1}, "call_id": "123456789"}[/TOOL_RESULTS]'
)
_TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_RAW_DECODE = (
    'a[{"name": "b", "arguments": {}, "id": "123456789"}]{"content": {"a": 1}, "call_id": "123456789"}'
)
_SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_TEXT = (
    r'<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>'
    r'[TOOL_RESULTS]▁{"content":▁"d{\"a\":▁1}",▁"call_id":▁"123456789"}[/TOOL_RESULTS]'
)
_SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_RAW_DECODE = (
    'a [{"name": "b", "arguments": {}, "id": "123456789"}] {"content": "d{\\"a\\": 1}", "call_id": "123456789"}'
)
_TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_TEXT = (
    r'<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "123456789"}]</s>'
    r'[TOOL_RESULTS]{"content": "d{\"a\": 1}", "call_id": "123456789"}[/TOOL_RESULTS]'
)
_TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_RAW_DECODE = (
    'a[{"name": "b", "arguments": {}, "id": "123456789"}]{"content": "d{\\"a\\": 1}", "call_id": "123456789"}'
)
_SPM_TOOL_CALL_WITHOUT_ID_TEXT = '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>'
_SPM_TOOL_CALL_WITHOUT_ID_RAW_DECODE = 'a [{"name": "b", "arguments": {}}]'
_TEKKEN_TOOL_CALL_WITHOUT_ID_TEXT = '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}}]</s>'
_TEKKEN_TOOL_CALL_WITHOUT_ID_RAW_DECODE = 'a[{"name": "b", "arguments": {}}]'


@pytest.fixture(scope="module")
def spm_tokenizer() -> InstructTokenizer:
    r"""Provide the bundled v3 SentencePiece instruct tokenizer."""
    return MistralTokenizer.v3().instruct_tokenizer


@pytest.fixture(scope="module")
def tekken_tokenizer() -> InstructTokenizer:
    r"""Provide the bundled v3 Tekken instruct tokenizer."""
    return MistralTokenizer.v3(is_tekken=True).instruct_tokenizer


def _build_tools_singleturn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 single-turn request with one available tool."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a")],
        available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
    )


def _build_tools_multiturn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 multi-turn request with two available tools."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ],
        available_tools=[
            Tool(function=Function(name="tool1", description="1", parameters={})),
            Tool(function=Function(name="tool2", description="2", parameters={})),
        ],
    )


def _build_system_tools_multiturn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 multi-turn request with a system prompt and tool."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ],
        available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        system_prompt="SYSTEM",
    )


def _build_prefixed_final_message_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with a prefixed final assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d", prefix=True),
        ],
        system_prompt="SYSTEM",
    )


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
    return _build_tool_call_without_id_request(
        tool_call=ToolCall(id="null", function=FunctionCall(name="b", arguments="{}"))
    )


def _build_default_id_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request with a default tool-call ID."""
    return _build_tool_call_without_id_request(tool_call=ToolCall(function=FunctionCall(name="b", arguments="{}")))


def _build_tool_call_without_id_request(tool_call: ToolCall) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a one-call v3 request containing the supplied call."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), AssistantMessage(tool_calls=[tool_call])]
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


def _build_empty_assistant_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v3 request containing an empty assistant message."""
    return InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="a"), AssistantMessage(content="")])


@pytest.mark.parametrize(
    argnames="backend, request_builder, expected_text, expected_tokens, expected_prefix_ids, expected_raw_decode",
    argvalues=[
        pytest.param(
            "spm",
            _build_tools_singleturn_request,
            _SPM_TOOLS_SINGLETURN_TEXT,
            _SPM_TOOLS_SINGLETURN_TOKENS,
            None,
            _SPM_TOOLS_SINGLETURN_RAW_DECODE,
            id="spm-tools-singleturn",
        ),
        pytest.param(
            "tekken",
            _build_tools_singleturn_request,
            _TEKKEN_TOOLS_SINGLETURN_TEXT,
            _TEKKEN_TOOLS_SINGLETURN_TOKENS,
            None,
            _TEKKEN_TOOLS_SINGLETURN_RAW_DECODE,
            id="tekken-tools-singleturn",
        ),
        pytest.param(
            "spm",
            _build_tools_multiturn_request,
            _SPM_TOOLS_MULTITURN_TEXT,
            _SPM_TOOLS_MULTITURN_TOKENS,
            None,
            _SPM_TOOLS_MULTITURN_RAW_DECODE,
            id="spm-tools-multiturn",
        ),
        pytest.param(
            "tekken",
            _build_tools_multiturn_request,
            _TEKKEN_TOOLS_MULTITURN_TEXT,
            _TEKKEN_TOOLS_MULTITURN_TOKENS,
            None,
            _TEKKEN_TOOLS_MULTITURN_RAW_DECODE,
            id="tekken-tools-multiturn",
        ),
        pytest.param(
            "spm",
            _build_system_tools_multiturn_request,
            _SPM_SYSTEM_TOOLS_MULTITURN_TEXT,
            _SPM_SYSTEM_TOOLS_MULTITURN_TOKENS,
            None,
            _SPM_SYSTEM_TOOLS_MULTITURN_RAW_DECODE,
            id="spm-system-tools-multiturn",
        ),
        pytest.param(
            "tekken",
            _build_system_tools_multiturn_request,
            _TEKKEN_SYSTEM_TOOLS_MULTITURN_TEXT,
            _TEKKEN_SYSTEM_TOOLS_MULTITURN_TOKENS,
            None,
            _TEKKEN_SYSTEM_TOOLS_MULTITURN_RAW_DECODE,
            id="tekken-system-tools-multiturn",
        ),
        pytest.param(
            "spm",
            _build_prefixed_final_message_request,
            _SPM_PREFIXED_FINAL_MESSAGE_TEXT,
            _SPM_PREFIXED_FINAL_MESSAGE_TOKENS,
            _SPM_PREFIXED_FINAL_MESSAGE_PREFIX_IDS,
            _SPM_PREFIXED_FINAL_MESSAGE_RAW_DECODE,
            id="spm-prefixed-final-message",
        ),
        pytest.param(
            "tekken",
            _build_prefixed_final_message_request,
            _TEKKEN_PREFIXED_FINAL_MESSAGE_TEXT,
            _TEKKEN_PREFIXED_FINAL_MESSAGE_TOKENS,
            _TEKKEN_PREFIXED_FINAL_MESSAGE_PREFIX_IDS,
            _TEKKEN_PREFIXED_FINAL_MESSAGE_RAW_DECODE,
            id="tekken-prefixed-final-message",
        ),
        pytest.param(
            "spm",
            _build_tool_message_multiple_shots_with_history_request,
            _SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TEXT,
            _SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TOKENS,
            None,
            _SPM_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_RAW_DECODE,
            id="spm-tool-message-multiple-shots-with-history",
        ),
        pytest.param(
            "tekken",
            _build_tool_message_multiple_shots_with_history_request,
            _TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TEXT,
            _TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_TOKENS,
            None,
            _TEKKEN_TOOL_MESSAGE_MULTIPLE_SHOTS_WITH_HISTORY_RAW_DECODE,
            id="tekken-tool-message-multiple-shots-with-history",
        ),
        pytest.param(
            "spm",
            _build_tool_message_multiple_calls_request,
            _SPM_TOOL_MESSAGE_MULTIPLE_CALLS_TEXT,
            _SPM_TOOL_MESSAGE_MULTIPLE_CALLS_TOKENS,
            None,
            _SPM_TOOL_MESSAGE_MULTIPLE_CALLS_RAW_DECODE,
            id="spm-tool-message-multiple-calls",
        ),
        pytest.param(
            "tekken",
            _build_tool_message_multiple_calls_request,
            _TEKKEN_TOOL_MESSAGE_MULTIPLE_CALLS_TEXT,
            _TEKKEN_TOOL_MESSAGE_MULTIPLE_CALLS_TOKENS,
            None,
            _TEKKEN_TOOL_MESSAGE_MULTIPLE_CALLS_RAW_DECODE,
            id="tekken-tool-message-multiple-calls",
        ),
        pytest.param(
            "spm",
            _build_tool_message_plain_string_request,
            _SPM_TOOL_MESSAGE_PLAIN_STRING_TEXT,
            _SPM_TOOL_MESSAGE_PLAIN_STRING_TOKENS,
            None,
            _SPM_TOOL_MESSAGE_PLAIN_STRING_RAW_DECODE,
            id="spm-tool-message-plain-string",
        ),
        pytest.param(
            "tekken",
            _build_tool_message_plain_string_request,
            _TEKKEN_TOOL_MESSAGE_PLAIN_STRING_TEXT,
            _TEKKEN_TOOL_MESSAGE_PLAIN_STRING_TOKENS,
            None,
            _TEKKEN_TOOL_MESSAGE_PLAIN_STRING_RAW_DECODE,
            id="tekken-tool-message-plain-string",
        ),
        pytest.param(
            "spm",
            _build_tool_message_json_object_string_request,
            _SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_TEXT,
            _SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_TOKENS,
            None,
            _SPM_TOOL_MESSAGE_JSON_OBJECT_STRING_RAW_DECODE,
            id="spm-tool-message-json-object-string",
        ),
        pytest.param(
            "tekken",
            _build_tool_message_json_object_string_request,
            _TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_TEXT,
            _TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_TOKENS,
            None,
            _TEKKEN_TOOL_MESSAGE_JSON_OBJECT_STRING_RAW_DECODE,
            id="tekken-tool-message-json-object-string",
        ),
        pytest.param(
            "spm",
            _build_tool_message_text_chunk_list_request,
            _SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_TEXT,
            _SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_TOKENS,
            None,
            _SPM_TOOL_MESSAGE_TEXT_CHUNK_LIST_RAW_DECODE,
            id="spm-tool-message-text-chunk-list",
        ),
        pytest.param(
            "tekken",
            _build_tool_message_text_chunk_list_request,
            _TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_TEXT,
            _TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_TOKENS,
            None,
            _TEKKEN_TOOL_MESSAGE_TEXT_CHUNK_LIST_RAW_DECODE,
            id="tekken-tool-message-text-chunk-list",
        ),
        pytest.param(
            "spm",
            _build_explicit_null_id_request,
            _SPM_TOOL_CALL_WITHOUT_ID_TEXT,
            _SPM_TOOL_CALL_WITHOUT_ID_TOKENS,
            None,
            _SPM_TOOL_CALL_WITHOUT_ID_RAW_DECODE,
            id="spm-tool-call-explicit-null-id",
        ),
        pytest.param(
            "tekken",
            _build_explicit_null_id_request,
            _TEKKEN_TOOL_CALL_WITHOUT_ID_TEXT,
            _TEKKEN_TOOL_CALL_WITHOUT_ID_TOKENS,
            None,
            _TEKKEN_TOOL_CALL_WITHOUT_ID_RAW_DECODE,
            id="tekken-tool-call-explicit-null-id",
        ),
        pytest.param(
            "spm",
            _build_default_id_request,
            _SPM_TOOL_CALL_WITHOUT_ID_TEXT,
            _SPM_TOOL_CALL_WITHOUT_ID_TOKENS,
            None,
            _SPM_TOOL_CALL_WITHOUT_ID_RAW_DECODE,
            id="spm-tool-call-default-id",
        ),
        pytest.param(
            "tekken",
            _build_default_id_request,
            _TEKKEN_TOOL_CALL_WITHOUT_ID_TEXT,
            _TEKKEN_TOOL_CALL_WITHOUT_ID_TOKENS,
            None,
            _TEKKEN_TOOL_CALL_WITHOUT_ID_RAW_DECODE,
            id="tekken-tool-call-default-id",
        ),
    ],
)
def test_encode_instruct(
    request: pytest.FixtureRequest,
    backend: str,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected_text: str,
    expected_tokens: list[int],
    expected_prefix_ids: list[int] | None,
    expected_raw_decode: str,
) -> None:
    r"""Preserve complete v3 instruct encodings for each backend and request."""
    tokenizer: InstructTokenizer = request.getfixturevalue(argname=f"{backend}_tokenizer")
    tokenized = tokenizer.encode_instruct(request=request_builder())
    raw_decode = tokenizer.tokenizer.decode(tokens=tokenized.tokens)

    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text
    assert tokenized.tokens == expected_tokens
    assert tokenized.prefix_ids == expected_prefix_ids
    assert raw_decode == expected_raw_decode


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
            _build_empty_assistant_request,
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
            _build_empty_assistant_request,
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="tekken-empty-assistant",
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

r"""Direct formatting tests for the bundled v2 instruct tokenizer."""

from collections.abc import Callable
from functools import partial

import pytest

from mistral_common.exceptions import TokenizerException, UnsupportedTokenizerFeatureException
from mistral_common.protocol.instruct.chunk import ContentChunk, TextChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import InstructTokenizer, Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.unit.tokens.tokenizers.instruct.request_builders import (
    build_empty_assistant_request,
    build_normal_request,
    build_prefixed_final_message_request,
    build_system_message_multiturn_request,
    build_system_message_prefixed_final_request,
    build_system_message_request,
    build_system_message_tools_multiturn_request,
    build_system_multiturn_request,
    build_system_singleturn_request,
    build_system_tools_multiturn_request,
    build_tools_multiturn_request,
    build_tools_singleturn_request,
)
from tests.utils import assert_tokenized_equal, decode_keep

# fmt: off
_NORMAL_TOKENS = [1, 3, 1032, 4, 1055, 2, 3, 1045, 4, 1049, 2]
_SYSTEM_SINGLETURN_TOKENS = [1, 3, 17889, 23294, 781, 781, 29476, 4]
_SYSTEM_MULTITURN_TOKENS = [1, 3, 1032, 4, 1055, 2, 3, 17889, 23294, 781, 781, 29485, 4, 1049, 2]
_PREFIXED_FINAL_MESSAGE_TOKENS = [1, 3, 1032, 4, 1055, 2, 3, 17889, 23294, 781, 781, 29485, 4, 1049]
_TOOLS_SINGLETURN_TOKENS = [
    1, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113, 7230, 29508, 1316,
    1113, 7286, 2032, 1113, 29508, 1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3, 1032, 4
]
_TOOLS_MULTITURN_TOKENS = [
    1, 3, 1032, 4, 1055, 2, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032,
    1113, 7230, 29508, 1316, 1113, 7286, 2032, 1113, 29508, 1316, 1113, 12206, 2032, 1139, 1743, 1649, 10598,
    1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113, 7230, 29518, 1316, 1113, 7286,
    2032, 1113, 29518, 1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3, 1045, 4, 1049, 2
]
_SYSTEM_TOOLS_MULTITURN_TOKENS = [
    1, 3, 1032, 4, 1055, 2, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032,
    1113, 7230, 29508, 1316, 1113, 7286, 2032, 1113, 29508, 1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3,
    17889, 23294, 781, 781, 29485, 4, 1049, 2
]
_TOOL_MESSAGE_MULTIPLE_SHOTS_WITHOUT_HISTORY_TOKENS = [
    1, 3, 1032, 4, 1085, 2, 3, 1053, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 1139,
    1743, 29561, 2, 8, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 4557, 2032, 1113, 29483, 29507, 10925, 9
]
_TOOL_RESULT_PLAIN_STRING_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 1139, 1743, 29561, 2, 8,
    1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 4557, 2032, 1113, 29483, 29507, 10925, 9
]
_TOOL_RESULT_JSON_OBJECT_STRING_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 1139, 1743, 29561, 2, 8,
    1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 4557, 2032, 10598, 29476, 2032, 29473, 29508, 1743, 29561, 9
]
_TOOL_RESULT_TEXT_CHUNK_LIST_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 17452, 2032, 1139, 1743, 29561, 2, 8,
    1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 4557, 2032, 1113, 29483, 1426, 29507, 29476, 18125, 29473,
    29508, 8703, 10925, 9
]
_TOOL_RESULT_WITHOUT_CALL_TURN_TOKENS = [
    1, 3, 1032, 4, 1055, 2, 8, 1501, 7567, 1629, 2032, 1113, 29494, 1316, 1113, 4557, 2032, 1113, 29483, 29507,
    10925, 9
]
_MULTIPLE_RESULTS_IN_HISTORY_TOKENS = [1, 3, 1032, 4, 1053, 2, 3, 1087, 4]
_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TOKENS = [
    1, 6, 1501, 7567, 1891, 2032, 1113, 3396, 1316, 1113, 3396, 2032, 10598, 1629, 2032, 1113, 29490, 1316,
    1113, 7286, 2032, 1113, 29485, 2783, 29565, 1316, 1113, 12206, 2032, 1139, 1743, 10925, 7, 3, 1032, 4, 5,
    1501, 7567, 1629, 2032, 1113, 29490, 1316, 1113, 17452, 2032, 10598, 29512, 2032, 1113, 29565, 29507, 1743,
    29561, 2, 8, 1501, 7567, 1629, 2032, 1113, 29490, 1316, 1113, 4557, 2032, 1113, 29565, 29507, 10925, 9
]
_TRAILING_SPACES_STRIPPED_TOKENS = [1, 3, 1032, 4, 1055, 2]
_RESULTS_RESET_AT_CALL_TURN_TOKENS = [
    1, 3, 1032, 4, 5, 1501, 7567, 1629, 2032, 1113, 29490, 1316, 1113, 17452, 2032, 1139, 1743, 29561, 2, 8,
    1501, 7567, 1629, 2032, 1113, 29490, 1316, 1113, 4557, 2032, 1113, 29480, 29507, 10925, 9, 5, 1501, 7567,
    1629, 2032, 1113, 29490, 1316, 1113, 17452, 2032, 1139, 1743, 29561, 2, 8, 1501, 7567, 1629, 2032, 1113,
    29490, 1316, 1113, 4557, 2032, 1113, 29543, 29507, 10925, 9
]
_RESULTS_RESET_AT_USER_TOKENS = [
    1, 3, 1032, 4, 3, 1055, 4, 8, 1501, 7567, 1629, 2032, 1113, 29490, 1316, 1113, 4557, 2032, 1113, 29543,
    29507, 10925, 9, 8, 1501, 7567, 1629, 2032, 1113, 29490, 1316, 1113, 4557, 2032, 1113, 29532, 29507, 10925, 9
]
_RESULTS_WITHOUT_USER_TOKENS = [
    1, 8, 1501, 7567, 1629, 2032, 1113, 29490, 1316, 1113, 4557, 2032, 1113, 29480, 29507, 10925, 9, 8, 1501,
    7567, 1629, 2032, 1113, 29490, 1316, 1113, 4557, 2032, 1113, 29543, 29507, 10925, 9
]
# fmt: on

_NORMAL_EXPECTED = Tokenized(tokens=_NORMAL_TOKENS)
_SYSTEM_SINGLETURN_EXPECTED = Tokenized(tokens=_SYSTEM_SINGLETURN_TOKENS)
_SYSTEM_MULTITURN_EXPECTED = Tokenized(tokens=_SYSTEM_MULTITURN_TOKENS)
_PREFIXED_FINAL_MESSAGE_EXPECTED = Tokenized(tokens=_PREFIXED_FINAL_MESSAGE_TOKENS, prefix_ids=[1049])
_TOOLS_SINGLETURN_EXPECTED = Tokenized(tokens=_TOOLS_SINGLETURN_TOKENS)
_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_TOOLS_MULTITURN_TOKENS)
_SYSTEM_TOOLS_MULTITURN_EXPECTED = Tokenized(tokens=_SYSTEM_TOOLS_MULTITURN_TOKENS)
_TOOL_MESSAGE_MULTIPLE_SHOTS_WITHOUT_HISTORY_EXPECTED = Tokenized(
    tokens=_TOOL_MESSAGE_MULTIPLE_SHOTS_WITHOUT_HISTORY_TOKENS
)
_TOOL_RESULT_PLAIN_STRING_EXPECTED = Tokenized(tokens=_TOOL_RESULT_PLAIN_STRING_TOKENS)
_TOOL_RESULT_JSON_OBJECT_STRING_EXPECTED = Tokenized(tokens=_TOOL_RESULT_JSON_OBJECT_STRING_TOKENS)
_TOOL_RESULT_TEXT_CHUNK_LIST_EXPECTED = Tokenized(tokens=_TOOL_RESULT_TEXT_CHUNK_LIST_TOKENS)
_TOOL_RESULT_WITHOUT_CALL_TURN_EXPECTED = Tokenized(tokens=_TOOL_RESULT_WITHOUT_CALL_TURN_TOKENS)
_MULTIPLE_RESULTS_IN_HISTORY_EXPECTED = Tokenized(tokens=_MULTIPLE_RESULTS_IN_HISTORY_TOKENS)
_NON_ASCII_TOOLS_CALLS_AND_RESULTS_EXPECTED = Tokenized(tokens=_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TOKENS)
_TRAILING_SPACES_STRIPPED_EXPECTED = Tokenized(tokens=_TRAILING_SPACES_STRIPPED_TOKENS)
_RESULTS_RESET_AT_CALL_TURN_EXPECTED = Tokenized(tokens=_RESULTS_RESET_AT_CALL_TURN_TOKENS)
_RESULTS_RESET_AT_USER_EXPECTED = Tokenized(tokens=_RESULTS_RESET_AT_USER_TOKENS)
_RESULTS_WITHOUT_USER_EXPECTED = Tokenized(tokens=_RESULTS_WITHOUT_USER_TOKENS)

_NORMAL_TEXT = "<s>[INST]▁a[/INST]▁b</s>[INST]▁c[/INST]▁d</s>"
_SYSTEM_SINGLETURN_TEXT = "<s>[INST]▁SYSTEM<0x0A><0x0A>a[/INST]"
_SYSTEM_MULTITURN_TEXT = "<s>[INST]▁a[/INST]▁b</s>[INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d</s>"
_PREFIXED_FINAL_MESSAGE_TEXT = "<s>[INST]▁a[/INST]▁b</s>[INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d"
_TOOLS_SINGLETURN_TEXT = (
    '<s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":'
    '▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}][/AVAILABLE_TOOLS][INST]▁a[/INST]'
)
_TOOLS_MULTITURN_TEXT = (
    "<s>[INST]▁a[/INST]▁b</s>"
    '[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}'
    ',▁{"type":▁"function",▁"function":▁{"name":▁"tool2",▁"description":▁"2",▁"parameters":▁{}}}]'
    "[/AVAILABLE_TOOLS][INST]▁c[/INST]▁d</s>"
)
_SYSTEM_TOOLS_MULTITURN_TEXT = (
    '<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":'
    '▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}][/AVAILABLE_TOOLS]'
    "[INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d</s>"
)
_TOOL_MESSAGE_MULTIPLE_SHOTS_WITHOUT_HISTORY_TEXT = (
    '<s>[INST]▁a[/INST]▁e</s>[INST]▁f[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>'
    '[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁"d"}][/TOOL_RESULTS]'
)
_TOOL_RESULT_PLAIN_STRING_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>'
    '[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁"d"}][/TOOL_RESULTS]'
)
_TOOL_RESULT_JSON_OBJECT_STRING_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>'
    '[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁{"a":▁1}}][/TOOL_RESULTS]'
)
_TOOL_RESULT_TEXT_CHUNK_LIST_TEXT = (
    r'<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>'
    r'[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁"d{\"a\":▁1}"}][/TOOL_RESULTS]'
)
_TOOL_RESULT_WITHOUT_CALL_TURN_TEXT = (
    '<s>[INST]▁a[/INST]▁b</s>[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁"d"}][/TOOL_RESULTS]'
)
_MULTIPLE_RESULTS_IN_HISTORY_TEXT = "<s>[INST]▁a[/INST]▁f</s>[INST]▁g[/INST]"
_NON_ASCII_TOOLS_CALLS_AND_RESULTS_TEXT = (
    '<s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"f",▁"description":▁"café",▁"parameters":▁{}}}'
    '][/AVAILABLE_TOOLS][INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"f",▁"arguments":▁{"x":▁"é"}}]</s>'
    '[TOOL_RESULTS]▁[{"name":▁"f",▁"content":▁"é"}][/TOOL_RESULTS]'
)
_TRAILING_SPACES_STRIPPED_TEXT = "<s>[INST]▁a[/INST]▁b</s>"
_RESULTS_RESET_AT_CALL_TURN_TEXT = (
    '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"f",▁"arguments":▁{}}]</s>'
    '[TOOL_RESULTS]▁[{"name":▁"f",▁"content":▁"r"}][/TOOL_RESULTS]'
    '[TOOL_CALLS]▁[{"name":▁"f",▁"arguments":▁{}}]</s>'
    '[TOOL_RESULTS]▁[{"name":▁"f",▁"content":▁"q"}][/TOOL_RESULTS]'
)
_RESULTS_RESET_AT_USER_TEXT = (
    "<s>[INST]▁a[/INST][INST]▁b[/INST]"
    '[TOOL_RESULTS]▁[{"name":▁"f",▁"content":▁"q"}][/TOOL_RESULTS]'
    '[TOOL_RESULTS]▁[{"name":▁"f",▁"content":▁"z"}][/TOOL_RESULTS]'
)
_RESULTS_WITHOUT_USER_TEXT = (
    '<s>[TOOL_RESULTS]▁[{"name":▁"f",▁"content":▁"r"}][/TOOL_RESULTS]'
    '[TOOL_RESULTS]▁[{"name":▁"f",▁"content":▁"q"}][/TOOL_RESULTS]'
)

_PARALLEL_RESULTS_MESSAGE = r"v2.*multiple tool results.*assistant turn"
_INVALID_ASSISTANT_MESSAGE = r"^Invalid assistant message:"
_SYSTEM_MESSAGE_NOT_IMPLEMENTED_MESSAGE = r"^System message encoding not implemented for InstructTokenizerV2$"


@pytest.fixture(scope="module")
def tokenizer() -> InstructTokenizer:
    r"""Provide the bundled v2 instruct tokenizer for formatting cases."""
    return MistralTokenizer.v2().instruct_tokenizer


def _build_tool_message_multiple_shots_without_history_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the v2 request with multiple tool shots and no call history."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
            ToolMessage(name="b", content="d"),
            AssistantMessage(content="e"),
            UserMessage(content="f"),
            AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
            ToolMessage(name="b", content="d"),
        ]
    )


def _build_tool_result_plain_string_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the v2 request with a plain-string tool result."""
    return _build_tool_result_request(content="d")


def _build_tool_result_json_object_string_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the v2 request with a JSON-object-string tool result."""
    return _build_tool_result_request(content='{"a": 1}')


def _build_tool_result_text_chunk_list_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the v2 request with a text-chunk-list tool result."""
    return _build_tool_result_request(content=[TextChunk(text="d"), TextChunk(text='{"a": 1}')])


def _build_tool_result_request(content: str | list[ContentChunk]) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a one-call v2 request with the supplied result content."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
            ToolMessage(name="b", content=content),
        ]
    )


def _build_tool_result_without_call_turn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v2 request with a tool result after a turn without calls."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            ToolMessage(name="b", content="d"),
        ]
    )


def _build_multiple_results_in_history_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v2 request with two tool results before its final user message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(
                tool_calls=[
                    ToolCall(id="123456789", function=FunctionCall(name="b", arguments="{}")),
                    ToolCall(id="223456789", function=FunctionCall(name="c", arguments="{}")),
                ]
            ),
            ToolMessage(name="b", content="d"),
            ToolMessage(name="c", content="e"),
            AssistantMessage(content="f"),
            UserMessage(content="g"),
        ]
    )


def _build_non_ascii_tools_calls_and_results_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v2 request with non-ASCII tool metadata, arguments and result."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=[Tool(function=Function(name="f", description="café", parameters={}))],
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="f", arguments='{"x": "é"}'))]),
            ToolMessage(name="f", content="é"),
        ],
    )


def _build_trailing_spaces_stripped_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v2 request with trailing assistant-content spaces."""
    return InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="a"), AssistantMessage(content="b  ")])


def _build_results_reset_at_call_turn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v2 request with one result after each of two call turns."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="f", arguments="{}"))]),
            ToolMessage(name="f", content="r"),
            AssistantMessage(tool_calls=[ToolCall(id="223456789", function=FunctionCall(name="f", arguments="{}"))]),
            ToolMessage(name="f", content="q"),
        ]
    )


def _build_results_reset_at_user_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v2 request with results separated by a new user message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="f", arguments="{}"))]),
            ToolMessage(name="f", content="r"),
            UserMessage(content="b"),
            ToolMessage(name="f", content="q"),
            ToolMessage(name="f", content="z"),
        ]
    )


def _build_results_without_user_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v2 request with two results and no user or assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[ToolMessage(name="f", content="r"), ToolMessage(name="f", content="q")]
    )


def _build_non_final_prefixed_assistant_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build a v2 request with a non-final prefixed assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b", prefix=True),
            UserMessage(content="c"),
        ]
    )


def _build_parallel_results_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the v2 request with two results for one assistant turn."""
    tool_calls = [
        ToolCall(id=f"call0000{index}", function=FunctionCall(name=f"tool_{index}", arguments="{}")) for index in (1, 2)
    ]
    messages: list[ChatMessage] = [
        UserMessage(content="Run these tools."),
        AssistantMessage(content=None, tool_calls=tool_calls),
    ]
    messages.extend(
        ToolMessage(
            name=f"tool_{index}",
            content=f"result {index}",
            tool_call_id=f"call0000{index}",
        )
        for index in (1, 2)
    )
    return InstructRequest[ChatMessage, Tool](messages=messages)


@pytest.mark.parametrize(
    argnames="request_builder, expected, expected_text",
    argvalues=[
        pytest.param(partial(build_normal_request, settings=None), _NORMAL_EXPECTED, _NORMAL_TEXT, id="normal"),
        pytest.param(
            partial(build_system_singleturn_request, settings=None),
            _SYSTEM_SINGLETURN_EXPECTED,
            _SYSTEM_SINGLETURN_TEXT,
            id="system-singleturn",
        ),
        pytest.param(
            partial(build_system_multiturn_request, settings=None),
            _SYSTEM_MULTITURN_EXPECTED,
            _SYSTEM_MULTITURN_TEXT,
            id="system-multiturn",
        ),
        pytest.param(
            partial(build_prefixed_final_message_request, settings=None),
            _PREFIXED_FINAL_MESSAGE_EXPECTED,
            _PREFIXED_FINAL_MESSAGE_TEXT,
            id="prefixed-final-message",
        ),
        pytest.param(
            partial(build_tools_singleturn_request, settings=None),
            _TOOLS_SINGLETURN_EXPECTED,
            _TOOLS_SINGLETURN_TEXT,
            id="tools-singleturn",
        ),
        pytest.param(
            partial(build_tools_multiturn_request, settings=None),
            _TOOLS_MULTITURN_EXPECTED,
            _TOOLS_MULTITURN_TEXT,
            id="tools-multiturn",
        ),
        pytest.param(
            partial(build_system_tools_multiturn_request, settings=None),
            _SYSTEM_TOOLS_MULTITURN_EXPECTED,
            _SYSTEM_TOOLS_MULTITURN_TEXT,
            id="system-tools-multiturn",
        ),
        pytest.param(
            _build_tool_message_multiple_shots_without_history_request,
            _TOOL_MESSAGE_MULTIPLE_SHOTS_WITHOUT_HISTORY_EXPECTED,
            _TOOL_MESSAGE_MULTIPLE_SHOTS_WITHOUT_HISTORY_TEXT,
            id="tool-message-multiple-shots-without-history",
        ),
        pytest.param(
            _build_tool_result_plain_string_request,
            _TOOL_RESULT_PLAIN_STRING_EXPECTED,
            _TOOL_RESULT_PLAIN_STRING_TEXT,
            id="tool-result-plain-string",
        ),
        pytest.param(
            _build_tool_result_json_object_string_request,
            _TOOL_RESULT_JSON_OBJECT_STRING_EXPECTED,
            _TOOL_RESULT_JSON_OBJECT_STRING_TEXT,
            id="tool-result-json-object-string",
        ),
        pytest.param(
            _build_tool_result_text_chunk_list_request,
            _TOOL_RESULT_TEXT_CHUNK_LIST_EXPECTED,
            _TOOL_RESULT_TEXT_CHUNK_LIST_TEXT,
            id="tool-result-text-chunk-list",
        ),
        pytest.param(
            _build_tool_result_without_call_turn_request,
            _TOOL_RESULT_WITHOUT_CALL_TURN_EXPECTED,
            _TOOL_RESULT_WITHOUT_CALL_TURN_TEXT,
            id="tool-result-without-call-turn",
        ),
        pytest.param(
            _build_multiple_results_in_history_request,
            _MULTIPLE_RESULTS_IN_HISTORY_EXPECTED,
            _MULTIPLE_RESULTS_IN_HISTORY_TEXT,
            id="multiple-results-in-history",
        ),
        pytest.param(
            _build_non_ascii_tools_calls_and_results_request,
            _NON_ASCII_TOOLS_CALLS_AND_RESULTS_EXPECTED,
            _NON_ASCII_TOOLS_CALLS_AND_RESULTS_TEXT,
            id="non-ascii-tools-calls-and-results",
        ),
        pytest.param(
            _build_trailing_spaces_stripped_request,
            _TRAILING_SPACES_STRIPPED_EXPECTED,
            _TRAILING_SPACES_STRIPPED_TEXT,
            id="trailing-spaces-stripped",
        ),
        pytest.param(
            _build_results_reset_at_call_turn_request,
            _RESULTS_RESET_AT_CALL_TURN_EXPECTED,
            _RESULTS_RESET_AT_CALL_TURN_TEXT,
            id="results-reset-at-call-turn",
        ),
        pytest.param(
            _build_results_reset_at_user_request,
            _RESULTS_RESET_AT_USER_EXPECTED,
            _RESULTS_RESET_AT_USER_TEXT,
            id="results-reset-at-user",
        ),
        pytest.param(
            _build_results_without_user_request,
            _RESULTS_WITHOUT_USER_EXPECTED,
            _RESULTS_WITHOUT_USER_TEXT,
            id="results-without-user",
        ),
    ],
)
def test_encode_instruct(
    tokenizer: InstructTokenizer,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected: Tokenized,
    expected_text: str,
) -> None:
    r"""Preserve complete v2 instruct encodings for each request shape."""
    tokenized = tokenizer.encode_instruct(request=request_builder())
    assert_tokenized_equal(actual=tokenized, expected=expected)
    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text


@pytest.mark.parametrize(
    argnames="request_builder, expected_exception, expected_message",
    argvalues=[
        pytest.param(
            _build_non_final_prefixed_assistant_request, AssertionError, None, id="non-final-prefixed-assistant"
        ),
        pytest.param(
            partial(build_empty_assistant_request, settings=None),
            TokenizerException,
            _INVALID_ASSISTANT_MESSAGE,
            id="empty-assistant",
        ),
        pytest.param(
            partial(build_system_message_request, settings=None),
            NotImplementedError,
            _SYSTEM_MESSAGE_NOT_IMPLEMENTED_MESSAGE,
            id="system-message",
        ),
        pytest.param(
            partial(build_system_message_multiturn_request, settings=None),
            NotImplementedError,
            _SYSTEM_MESSAGE_NOT_IMPLEMENTED_MESSAGE,
            id="system-message-multiturn",
        ),
        pytest.param(
            partial(build_system_message_prefixed_final_request, settings=None),
            NotImplementedError,
            _SYSTEM_MESSAGE_NOT_IMPLEMENTED_MESSAGE,
            id="system-message-prefixed-final",
        ),
        pytest.param(
            partial(build_system_message_tools_multiturn_request, settings=None),
            NotImplementedError,
            _SYSTEM_MESSAGE_NOT_IMPLEMENTED_MESSAGE,
            id="system-message-tools-multiturn",
        ),
    ],
)
def test_encode_instruct_rejected(
    tokenizer: InstructTokenizer,
    request_builder: Callable[[], InstructRequest[ChatMessage, Tool]],
    expected_exception: type[Exception],
    expected_message: str | None,
) -> None:
    r"""Preserve direct v2 exception types and specified messages."""
    request = request_builder()

    with pytest.raises(expected_exception=expected_exception, match=expected_message):
        tokenizer.encode_instruct(request=request)


def test_rejects_multiple_results_for_one_v2_turn(tokenizer: InstructTokenizer) -> None:
    r"""Reject multiple tool results associated with one v2 assistant turn."""
    request = _build_parallel_results_request()

    with pytest.raises(
        expected_exception=UnsupportedTokenizerFeatureException,
        match=_PARALLEL_RESULTS_MESSAGE,
    ):
        tokenizer.encode_instruct(request=request)

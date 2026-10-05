import json

import pytest

from mistral_common.exceptions import UnsupportedTokenizerFeatureException
from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import InstructTokenizer, SpecialTokens, Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils import decode_keep

_PARALLEL_RESULTS_MESSAGE = r"v2.*multiple tool results.*assistant turn"


@pytest.fixture()
def tokenizer() -> InstructTokenizer:
    return MistralTokenizer.v2().instruct_tokenizer


def parallel_results_request() -> InstructRequest[ChatMessage, Tool]:
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


def tool_result_block_count(tokenizer: InstructTokenizer, tokenized: Tokenized) -> int:
    r"""Count encoded v2 or v3 tool-result blocks."""
    marker = tokenizer.tokenizer.get_special_token(SpecialTokens.begin_tool_results.value)
    return tokenized.tokens.count(marker)


def test_rejects_multiple_results_for_one_v2_turn(tokenizer: InstructTokenizer) -> None:
    request = parallel_results_request()

    with pytest.raises(UnsupportedTokenizerFeatureException, match=_PARALLEL_RESULTS_MESSAGE):
        tokenizer.encode_instruct(request)


def test_v3_still_encodes_multiple_tool_results() -> None:
    v3_tokenizer = MistralTokenizer.v3().instruct_tokenizer
    request = parallel_results_request()

    tokenized = v3_tokenizer.encode_instruct(request)

    assert tool_result_block_count(tokenizer=v3_tokenizer, tokenized=tokenized) == 2


def test_normal(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content="b"),
                UserMessage(content="c"),
                AssistantMessage(content="d"),
            ]
        )
    )
    tokens = tokenized.tokens
    text = decode_keep(tokenizer, tokenized)
    assert text == "<s>[INST]▁a[/INST]▁b</s>[INST]▁c[/INST]▁d</s>"
    assert tokens == [1, 3, 1032, 4, 1055, 2, 3, 1045, 4, 1049, 2]
    assert tokenized.prefix_ids is None


def test_non_final_prefixed_assistant_fails_prefix_invariant(tokenizer: InstructTokenizer) -> None:
    with pytest.raises(AssertionError):
        tokenizer.encode_instruct(
            InstructRequest(
                messages=[
                    UserMessage(content="a"),
                    AssistantMessage(content="b", prefix=True),
                    UserMessage(content="c"),
                ]
            )
        )


def test_tools_singleturn(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[UserMessage(content="a")],
            available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        )
    )
    tokens = tokenized.tokens
    text = decode_keep(tokenizer, tokenized)
    assert text == (
        '<s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}][/AVAILABLE_TOOLS][INST]▁a[/INST]'
    )  # NOTE THE SPACE
    begin_tool, end_tool = tokens.index(6), tokens.index(7)
    assert tokens[:begin_tool] + tokens[end_tool + 1 :] == [1, 3, 1032, 4]
    json.loads(tokenizer.tokenizer.decode(tokens[begin_tool : end_tool + 1]))


def test_tools_multiturn(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
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
    )
    tokens = tokenized.tokens
    text = decode_keep(tokenizer, tokenized)
    assert text == (
        "<s>[INST]▁a[/INST]▁b</s>"
        '[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}'
        ',▁{"type":▁"function",▁"function":▁{"name":▁"tool2",▁"description":▁"2",▁"parameters":▁{}}}]'
        "[/AVAILABLE_TOOLS][INST]▁c[/INST]▁d</s>"
    )
    begin_tool, end_tool = tokens.index(6), tokens.index(7)
    assert tokens[:begin_tool] + tokens[end_tool + 1 :] == [
        1,
        3,
        1032,
        4,
        1055,
        2,
        3,
        1045,
        4,
        1049,
        2,
    ]
    json.loads(tokenizer.tokenizer.decode(tokens[begin_tool : end_tool + 1]))


def test_system_singleturn(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(InstructRequest(messages=[UserMessage(content="a")], system_prompt="SYSTEM"))
    tokens = tokenized.tokens
    text = decode_keep(tokenizer, tokenized)
    assert text == "<s>[INST]▁SYSTEM<0x0A><0x0A>a[/INST]"  # NOTE THE SPACE
    assert tokens == [1, 3, 17889, 23294, 781, 781, 29476, 4]
    assert tokenizer.tokenizer.decode(tokens) == "SYSTEM\n\na"


def test_system_multiturn(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content="b"),
                UserMessage(content="c"),
                AssistantMessage(content="d"),
            ],
            system_prompt="SYSTEM",
        )
    )
    tokens = tokenized.tokens
    text = decode_keep(tokenizer, tokenized)
    assert text == "<s>[INST]▁a[/INST]▁b</s>[INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d</s>"
    assert tokens == [
        1,
        3,
        1032,
        4,
        1055,
        2,
        3,
        17889,
        23294,
        781,
        781,
        29485,
        4,
        1049,
        2,
    ]
    first_eos = tokens.index(2)
    assert tokenizer.tokenizer.decode(tokens[first_eos:]) == "SYSTEM\n\nc d"


def test_prefixed_final_message(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content="b"),
                UserMessage(content="c"),
                AssistantMessage(content="d", prefix=True),
            ],
            system_prompt="SYSTEM",
        )
    )
    tokens = tokenized.tokens
    text = decode_keep(tokenizer, tokenized)
    assert text == "<s>[INST]▁a[/INST]▁b</s>[INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d"
    assert tokens == [
        1,
        3,
        1032,
        4,
        1055,
        2,
        3,
        17889,
        23294,
        781,
        781,
        29485,
        4,
        1049,
    ]
    assert tokenized.prefix_ids == [1049]


def test_system_tools_multiturn(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content="b"),
                UserMessage(content="c"),
                AssistantMessage(content="d"),
            ],
            available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
            system_prompt="SYSTEM",
        )
    )
    tokens = tokenized.tokens
    text = decode_keep(tokenizer, tokenized)
    assert text == (
        '<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}][/AVAILABLE_TOOLS][INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d</s>'
    )

    begin_tool, end_tool = tokens.index(6), tokens.index(7)
    assert tokens[end_tool + 1 :].index(3) == 0  # begin_inst follows end_tool
    assert tokenizer.tokenizer.decode(tokens[:begin_tool]) == "a b"
    assert tokenizer.tokenizer.decode(tokens[end_tool + 1 :]) == "SYSTEM\n\nc d"


def test_tool_response(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
                ToolMessage(name="b", content="d"),
            ],
        )
    )
    text = decode_keep(tokenizer, tokenized)
    assert text == (
        '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁"d"}][/TOOL_RESULTS]'
    )

    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content=None, tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
                ToolMessage(name="b", content='{"a": 1}'),
            ],
        )
    )
    text = decode_keep(tokenizer, tokenized)
    assert text == (
        '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁{"a":▁1}}][/TOOL_RESULTS]'
    )

    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content=None, tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
                ToolMessage(name="b", content=[TextChunk(text="d"), TextChunk(text='{"a": 1}')]),
            ],
        )
    )
    text = decode_keep(tokenizer, tokenized)
    assert text == (
        '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁"d{\\"a\\":▁1}"}][/TOOL_RESULTS]'
    )


def test_tool_message_multiple_shots_without_history(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
                ToolMessage(name="b", content="d"),
                AssistantMessage(content="e"),
                UserMessage(content="f"),
                AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="b", arguments="{}"))]),
                ToolMessage(name="b", content="d"),
            ],
        )
    )
    text = decode_keep(tokenizer, tokenized)
    assert text == (
        '<s>[INST]▁a[/INST]▁e</s>[INST]▁f[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁"d"}][/TOOL_RESULTS]'
    )

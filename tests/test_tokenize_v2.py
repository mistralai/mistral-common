import json

import pytest

from mistral_common.exceptions import InvalidAssistantMessageException, InvalidMessageStructureException
from mistral_common.protocol.instruct.messages import AssistantMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import InstructTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


@pytest.fixture()
def tokenizer() -> InstructTokenizer:
    return MistralTokenizer.v2().instruct_tokenizer


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
    tokens, text = tokenized.tokens, tokenized.text
    assert text == "<s>[INST]▁a[/INST]▁b</s>[INST]▁c[/INST]▁d</s>"
    assert tokens == [1, 3, 1032, 4, 1055, 2, 3, 1045, 4, 1049, 2]


def test_tools_singleturn(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[UserMessage(content="a")],
            available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        )
    )
    tokens, text = tokenized.tokens, tokenized.text
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
    tokens, text = tokenized.tokens, tokenized.text
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
    tokens, text = tokenized.tokens, tokenized.text
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
    tokens, text = tokenized.tokens, tokenized.text
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


def test_continue_final_message(tokenizer: InstructTokenizer) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content="b"),
                UserMessage(content="c"),
                AssistantMessage(content="d"),
            ],
            system_prompt="SYSTEM",
            continue_final_message=True,
        )
    )
    tokens, text = tokenized.tokens, tokenized.text
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

    with pytest.raises(
        InvalidMessageStructureException, match="Cannot continue final message if it is not an assistant message"
    ):
        tokenizer.encode_instruct(
            InstructRequest(
                messages=[
                    UserMessage(content="a"),
                    AssistantMessage(content="b"),
                    UserMessage(content="c"),
                ],
                system_prompt="SYSTEM",
                continue_final_message=True,
            )
        )

    with pytest.raises(
        InvalidAssistantMessageException,
        match="`continue_message` is only supported for assistant messages that have `prefix=False`.",
    ):
        tokenizer.encode_assistant_message(  # type: ignore[attr-defined]
            AssistantMessage(
                content='"blabla"',
                prefix=True,
            ),
            is_before_last_user_message=False,
            continue_message=True,
        )


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
    tokens, text = tokenized.tokens, tokenized.text
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
    _, text = tokenized.tokens, tokenized.text
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
    _, text = tokenized.tokens, tokenized.text
    assert text == (
        '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁{"a":▁1}}][/TOOL_RESULTS]'
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
    _, text = tokenized.tokens, tokenized.text
    assert text == (
        '<s>[INST]▁a[/INST]▁e</s>[INST]▁f[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>[TOOL_RESULTS]▁[{"name":▁"b",▁"content":▁"d"}][/TOOL_RESULTS]'
    )

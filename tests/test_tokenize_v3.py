import json

import pytest
from mistral_common.protocol.instruct.messages import AssistantMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.instruct.normalize import InstructRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceInstructTokenizerV3


@pytest.fixture()
def tokenizer() -> SentencePieceInstructTokenizerV3:
    return MistralTokenizer.v3().instruct_tokenizer  # type: ignore


def test_tools_singleturn(tokenizer: SentencePieceInstructTokenizerV3) -> None:
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


def test_tools_multiturn(tokenizer: SentencePieceInstructTokenizerV3) -> None:
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
        "<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁["
        '{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}},'
        '▁{"type":▁"function",▁"function":▁{"name":▁"tool2",▁"description":▁"2",▁"parameters":▁{}}}]'
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


def test_system_tools_multiturn(tokenizer: SentencePieceInstructTokenizerV3) -> None:
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
        "<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁["
        '{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}]'
        "[/AVAILABLE_TOOLS][INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d</s>"
    )

    begin_tool, end_tool = tokens.index(6), tokens.index(7)
    assert tokens[end_tool + 1 :].index(3) == 0  # begin_inst follows end_tool
    assert tokenizer.tokenizer.decode(tokens[:begin_tool]) == "a b"
    assert tokenizer.tokenizer.decode(tokens[end_tool + 1 :]) == "SYSTEM\n\nc d"


def test_tool_message(tokenizer: SentencePieceInstructTokenizerV3) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(
                    tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="b", arguments="{}"))]
                ),
                ToolMessage(
                    name="b",
                    content="d",
                    tool_call_id="123456789",
                ),
            ],
        )
    )
    _, text = tokenized.tokens, tokenized.text
    assert text == (
        '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>[TOOL_RESULTS]▁{"call_id":▁"123456789",▁"content":▁"d"}[/TOOL_RESULTS]'
    )

    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(
                    content=None, tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="b", arguments="{}"))]
                ),
                ToolMessage(
                    name="b",
                    content='{"a": 1}',
                    tool_call_id="123456789",
                ),
            ],
        )
    )
    _, text = tokenized.tokens, tokenized.text
    assert text == (
        '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>[TOOL_RESULTS]▁{"call_id":▁"123456789",▁"content":▁{"a":▁1}}[/TOOL_RESULTS]'
    )


def test_tool_message_multiple_shots_with_history(tokenizer: SentencePieceInstructTokenizerV3) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(tool_calls=[ToolCall(id="0", function=FunctionCall(name="b", arguments="{}"))]),
                ToolMessage(name="b", content="d", tool_call_id="0"),
                AssistantMessage(content="e"),
                UserMessage(content="f"),
                AssistantMessage(tool_calls=[ToolCall(id="1", function=FunctionCall(name="b", arguments="{}"))]),
                ToolMessage(name="b", content="d", tool_call_id="1"),
            ],
        )
    )
    _, text = tokenized.tokens, tokenized.text
    assert text == (
        '<s>[INST]▁a[/INST]'
        '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"0"}]</s>[TOOL_RESULTS]▁{"call_id":▁"0",▁"content":▁"d"}[/TOOL_RESULTS]'
        '▁e</s>[INST]▁f[/INST]'
        '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"1"}]</s>[TOOL_RESULTS]▁{"call_id":▁"1",▁"content":▁"d"}[/TOOL_RESULTS]'
    )


def test_tool_message_multiple_calls(tokenizer: SentencePieceInstructTokenizerV3) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
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
            ],
        )
    )
    _, text = tokenized.tokens, tokenized.text
    assert text == (
        '<s>[INST]▁a[/INST]'
        '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"0"},▁{"name":▁"q",▁"arguments":▁{},▁"id":▁"1"}]</s>'
        '[TOOL_RESULTS]▁{"call_id":▁"0",▁"content":▁"d"}[/TOOL_RESULTS]'
        '[TOOL_RESULTS]▁{"call_id":▁"1",▁"content":▁"d"}[/TOOL_RESULTS]'
        '▁e</s>[INST]▁f[/INST]'
        '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"2"},▁{"name":▁"q",▁"arguments":▁{},▁"id":▁"3"}]</s>'
        '[TOOL_RESULTS]▁{"call_id":▁"2",▁"content":▁"d"}[/TOOL_RESULTS]'
        '[TOOL_RESULTS]▁{"call_id":▁"3",▁"content":▁"d"}[/TOOL_RESULTS]'
    )

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from mistral_common.protocol.instruct.messages import AssistantMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import InstructTokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer,
    is_sentencepiece,
)
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy


def tokenizer() -> InstructTokenizer:
    return MistralTokenizer.v3().instruct_tokenizer


def tekken_tokenizer() -> InstructTokenizer:
    tekken = MistralTokenizer.v3(is_tekken=True).instruct_tokenizer
    tekken.tokenizer._special_token_policy = SpecialTokenPolicy.IGNORE
    return tekken


def test_is_spm() -> None:
    # this is valid
    for suffix in list(TokenizerVersion.__members__):
        with NamedTemporaryFile(suffix=".model." + suffix) as f:
            assert is_sentencepiece(f.name)

    with NamedTemporaryFile(suffix=".model") as f:
        assert is_sentencepiece(f.name)

    # this is not valid
    with NamedTemporaryFile(suffix=".model.vx") as f:
        assert not is_sentencepiece(f.name)


def test_spm_version() -> None:
    directory = Path(__file__).parent / "data"

    for file in directory.iterdir():
        if not file.is_file() or str(file).endswith(".json"):
            continue
        suffix = file.suffix[1:]
        print(suffix)
        if suffix == "model":
            assert SentencePieceTokenizer(str(file)).version == TokenizerVersion.v1
        else:
            assert SentencePieceTokenizer(str(file)).version == TokenizerVersion(suffix)


@pytest.mark.parametrize(
    "tokenizer, expected_text, begin_tool_index, end_tool_index, expected_tokens_before_tool, expected_tokens_after_tool",
    [
        (tokenizer(),
         '<s>[AVAILABLE_TOOLS]▁[{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}][/AVAILABLE_TOOLS][INST]▁a[/INST]',
         6, 7,
         [1, 3, 1032, 4],
         []),
        (tekken_tokenizer(),
         '<s>[AVAILABLE_TOOLS][{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}}][/AVAILABLE_TOOLS][INST]a[/INST]',
         5, 6,
         [1, 3, 1097, 4],
         [])
    ])
def test_tools_singleturn(tokenizer: InstructTokenizer, expected_text: str, begin_tool_index: int, end_tool_index: int,
                          expected_tokens_before_tool: list, expected_tokens_after_tool: list) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[UserMessage(content="a")],
            available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        )
    )
    tokens, text = tokenized.tokens, tokenized.text
    assert text == expected_text

    begin_tool, end_tool = tokens.index(begin_tool_index), tokens.index(end_tool_index)
    assert tokens[:begin_tool] + tokens[end_tool + 1:] == expected_tokens_before_tool + expected_tokens_after_tool
    json.loads(tokenizer.tokenizer.decode(tokens[begin_tool: end_tool + 1]))


@pytest.mark.parametrize(
    "tokenizer, expected_text, begin_tool_index, end_tool_index, expected_tokens_before_tool, expected_tokens_after_tool",
    [
        (tokenizer(),
         "<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁["
         '{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}},'
         '▁{"type":▁"function",▁"function":▁{"name":▁"tool2",▁"description":▁"2",▁"parameters":▁{}}}]'
         "[/AVAILABLE_TOOLS][INST]▁c[/INST]▁d</s>",
         6, 7,
         [1, 3, 1032, 4, 1055],
         [2, 3, 1045, 4, 1049, 2]),
        (tekken_tokenizer(),
         "<s>[INST]a[/INST]b</s>[AVAILABLE_TOOLS]["
         '{"type": "function", "function": {"name": "tool1", "description": "1", "parameters": {}}},'
         ' {"type": "function", "function": {"name": "tool2", "description": "2", "parameters": {}}}]'
         "[/AVAILABLE_TOOLS][INST]c[/INST]d</s>",
         5, 6,
         [1, 3, 1097, 4, 1098],
         [2, 3, 1099, 4, 1100, 2])
    ])
def test_tools_multiturn(tokenizer: InstructTokenizer, expected_text: str, begin_tool_index: int, end_tool_index: int,
                         expected_tokens_before_tool: list, expected_tokens_after_tool: list) -> None:
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
    assert text == expected_text

    begin_tool, end_tool = tokens.index(begin_tool_index), tokens.index(end_tool_index)
    assert tokens[:begin_tool] + tokens[end_tool + 1:] == expected_tokens_before_tool + expected_tokens_after_tool
    json.loads(tokenizer.tokenizer.decode(tokens[begin_tool: end_tool + 1]))


@pytest.mark.parametrize(
    "tokenizer, expected_text, begin_tool_index, end_tool_index, decoded_before_tool, decoded_after_tool", [
        (tokenizer(),
         "<s>[INST]▁a[/INST]▁b</s>[AVAILABLE_TOOLS]▁["
         '{"type":▁"function",▁"function":▁{"name":▁"tool1",▁"description":▁"1",▁"parameters":▁{}}}]'
         "[/AVAILABLE_TOOLS][INST]▁SYSTEM<0x0A><0x0A>c[/INST]▁d</s>",
         6, 7, "a b", "SYSTEM\n\nc d"),
        (tekken_tokenizer(),
         '<s>[INST]a[/INST]b</s>[AVAILABLE_TOOLS][{"type": "function", "function": '
         '{"name": "tool1", "description": "1", "parameters": '
         '{}}}][/AVAILABLE_TOOLS][INST]SYSTEM\n'
         '\n'
         'c[/INST]d</s>',
         5, 6, "ab", "SYSTEM\n\ncd")
    ])
def test_system_tools_multiturn(tokenizer: InstructTokenizer, expected_text: str, begin_tool_index: int,
                                end_tool_index: int, decoded_before_tool: str, decoded_after_tool: str) -> None:
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
    assert text == expected_text

    begin_tool, end_tool = tokens.index(begin_tool_index), tokens.index(end_tool_index)
    assert tokens[end_tool + 1:].index(3) == 0  # begin_inst follows end_tool
    assert tokenizer.tokenizer.decode(tokens[:begin_tool]) == decoded_before_tool
    assert tokenizer.tokenizer.decode(tokens[end_tool + 1:]) == decoded_after_tool


@pytest.mark.parametrize("tokenizer, expected_text_1, expected_text_2", [
    (tokenizer(),
     '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"123456789"}[/TOOL_RESULTS]',
     '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"123456789"}]</s>[TOOL_RESULTS]▁{"content":▁{"a":▁1},▁"call_id":▁"123456789"}[/TOOL_RESULTS]'),
    (tekken_tokenizer(),
     '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "123456789"}]</s>[TOOL_RESULTS]{"content": "d", "call_id": "123456789"}[/TOOL_RESULTS]',
     '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}, "id": "123456789"}]</s>[TOOL_RESULTS]{"content": {"a": 1}, "call_id": "123456789"}[/TOOL_RESULTS]')
])
def test_tool_message(tokenizer: InstructTokenizer, expected_text_1: str, expected_text_2: str) -> None:
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
    assert text == expected_text_1

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
    assert text == expected_text_2


@pytest.mark.parametrize("tokenizer, expected_text", [
    (tokenizer(),
     '<s>[INST]▁a[/INST][TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{}}]</s>'),
    (tekken_tokenizer(),
     '<s>[INST]a[/INST][TOOL_CALLS][{"name": "b", "arguments": {}}]</s>')
])
def test_tool_message_no_id_fine_tuning_ok(tokenizer: InstructTokenizer, expected_text: str) -> None:
    # In fine-tuning we allow passing a tool call as the last message.
    # We need to make sure to not parse this empty id as "null"
    function = FunctionCall(name="b", arguments="{}")

    tool_calls = [ToolCall(id="null", function=function), ToolCall(function=function)]
    for tool_call in tool_calls:
        tokenized = tokenizer.encode_instruct(
            InstructRequest(
                messages=[
                    UserMessage(content="a"),
                    AssistantMessage(tool_calls=[tool_call]),
                ],
            )
        )
        _, text = tokenized.tokens, tokenized.text
        # make sure to "null" is in the output
        assert text == expected_text


@pytest.mark.parametrize("tokenizer, expected_text", [
    (tokenizer(),
     "<s>[INST]▁a[/INST]"
     '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"0"}]</s>[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"0"}[/TOOL_RESULTS]'
     "▁e</s>[INST]▁f[/INST]"
     '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"1"}]</s>[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"1"}[/TOOL_RESULTS]'),
    (tekken_tokenizer(),
     "<s>[INST]a[/INST]"
     '[TOOL_CALLS][{"name": "b", "arguments": {}, "id": "0"}]</s>[TOOL_RESULTS]{"content": "d", "call_id": "0"}[/TOOL_RESULTS]'
     "e</s>[INST]f[/INST]"
     '[TOOL_CALLS][{"name": "b", "arguments": {}, "id": "1"}]</s>[TOOL_RESULTS]{"content": "d", "call_id": "1"}[/TOOL_RESULTS]')
])
def test_tool_message_multiple_shots_with_history(tokenizer: InstructTokenizer, expected_text: str) -> None:
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
    assert text == expected_text


@pytest.mark.parametrize("tokenizer, expected_text", [
    (tokenizer(),
     "<s>[INST]▁a[/INST]"
     '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"0"},▁{"name":▁"q",▁"arguments":▁{},▁"id":▁"1"}]</s>'
     '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"0"}[/TOOL_RESULTS]'
     '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"1"}[/TOOL_RESULTS]'
     "▁e</s>[INST]▁f[/INST]"
     '[TOOL_CALLS]▁[{"name":▁"b",▁"arguments":▁{},▁"id":▁"2"},▁{"name":▁"q",▁"arguments":▁{},▁"id":▁"3"}]</s>'
     '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"2"}[/TOOL_RESULTS]'
     '[TOOL_RESULTS]▁{"content":▁"d",▁"call_id":▁"3"}[/TOOL_RESULTS]'),
    (tekken_tokenizer(),
     "<s>[INST]a[/INST]"
     '[TOOL_CALLS][{"name": "b", "arguments": {}, "id": "0"}, {"name": "q", "arguments": {}, "id": "1"}]</s>'
     '[TOOL_RESULTS]{"content": "d", "call_id": "0"}[/TOOL_RESULTS]'
     '[TOOL_RESULTS]{"content": "d", "call_id": "1"}[/TOOL_RESULTS]'
     "e</s>[INST]f[/INST]"
     '[TOOL_CALLS][{"name": "b", "arguments": {}, "id": "2"}, {"name": "q", "arguments": {}, "id": "3"}]</s>'
     '[TOOL_RESULTS]{"content": "d", "call_id": "2"}[/TOOL_RESULTS]'
     '[TOOL_RESULTS]{"content": "d", "call_id": "3"}[/TOOL_RESULTS]')
])
def test_tool_message_multiple_calls(tokenizer: InstructTokenizer, expected_text: str) -> None:
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
    assert text == expected_text

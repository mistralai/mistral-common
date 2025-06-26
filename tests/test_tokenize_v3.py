import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from mistral_common.exceptions import InvalidAssistantMessageException, InvalidMessageStructureException
from mistral_common.protocol.instruct.messages import AssistantMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import InstructTokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer,
    is_sentencepiece,
)
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy, Tekkenizer

TEKKEN_SPECIAL_WHITESPACE = ""
TEKKEN_WHITESPACE = " "
TEKKEN_BEGIN_TOOL_ID = 5
TEKKEN_END_TOOL_ID = 6

SPM_SPECIAL_WHITESPACE = "▁"
SPM_WHITESPACE = "▁"
SPM_BEGIN_TOOL_ID = 6
SPM_END_TOOL_ID = 7


def tokenizer() -> InstructTokenizer:
    return MistralTokenizer.v3().instruct_tokenizer


def tekken_tokenizer() -> InstructTokenizer:
    tekken = MistralTokenizer.v3(is_tekken=True).instruct_tokenizer
    tekken.tokenizer.special_token_policy = SpecialTokenPolicy.IGNORE  # type: ignore
    return tekken


def test_is_spm() -> None:
    # this is valid
    for suffix in list(TokenizerVersion.__members__) + ["v3m1"]:
        with NamedTemporaryFile(suffix=".model." + suffix) as f:
            assert is_sentencepiece(f.name)

    with NamedTemporaryFile(suffix=".model") as f:
        assert is_sentencepiece(f.name)

    # this is not valid
    with NamedTemporaryFile(suffix=".model.vx") as f:
        assert not is_sentencepiece(f.name)


def test_spm_version() -> None:
    directory = Path(__file__).parent.parent / "src" / "mistral_common" / "data"

    for file in directory.iterdir():
        if not file.is_file() or str(file).endswith(".json"):
            continue
        suffix = file.suffix[1:].split("m")[0]
        if suffix == "model":
            assert SentencePieceTokenizer(str(file)).version == TokenizerVersion.v1
        else:
            assert SentencePieceTokenizer(str(file)).version == TokenizerVersion(suffix)


@pytest.mark.parametrize(
    "tokenizer, special_ws, ws, begin_tool_index, end_tool_index, expected_tokens_before_tool",
    [
        (
            tokenizer(),
            SPM_WHITESPACE,
            SPM_WHITESPACE,
            SPM_BEGIN_TOOL_ID,
            SPM_END_TOOL_ID,
            [1, 3, 1032, 4],
        ),
        (
            tekken_tokenizer(),
            TEKKEN_SPECIAL_WHITESPACE,
            TEKKEN_WHITESPACE,
            TEKKEN_BEGIN_TOOL_ID,
            TEKKEN_END_TOOL_ID,
            [1, 3, 1097, 4],
        ),
    ],
)
def test_tools_singleturn(
    tokenizer: InstructTokenizer,
    special_ws: str,
    ws: str,
    begin_tool_index: int,
    end_tool_index: int,
    expected_tokens_before_tool: list,
) -> None:
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[UserMessage(content="a")],
            available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        )
    )
    tokens, text = tokenized.tokens, tokenized.text
    assert text == (
        f"<s>[AVAILABLE_TOOLS]{special_ws}["
        f'{{"type":{ws}"function",{ws}"function":{ws}{{"name":{ws}"tool1",{ws}"description":{ws}"1",{ws}"parameters":{ws}{{}}}}}}]'
        f"[/AVAILABLE_TOOLS][INST]{special_ws}a[/INST]"
    )

    begin_tool, end_tool = tokens.index(begin_tool_index), tokens.index(end_tool_index)
    assert tokens[:begin_tool] + tokens[end_tool + 1 :] == expected_tokens_before_tool + []
    json.loads(tokenizer.tokenizer.decode(tokens[begin_tool : end_tool + 1]))


@pytest.mark.parametrize(
    "tokenizer, special_ws, ws, begin_tool_index, end_tool_index, "
    "expected_tokens_before_tool, expected_tokens_after_tool",
    [
        (
            tokenizer(),
            SPM_SPECIAL_WHITESPACE,
            SPM_WHITESPACE,
            SPM_BEGIN_TOOL_ID,
            SPM_END_TOOL_ID,
            [1, 3, 1032, 4, 1055],
            [2, 3, 1045, 4, 1049, 2],
        ),
        (
            tekken_tokenizer(),
            TEKKEN_SPECIAL_WHITESPACE,
            TEKKEN_WHITESPACE,
            TEKKEN_BEGIN_TOOL_ID,
            TEKKEN_END_TOOL_ID,
            [1, 3, 1097, 4, 1098],
            [2, 3, 1099, 4, 1100, 2],
        ),
    ],
)
def test_tools_multiturn(
    tokenizer: InstructTokenizer,
    special_ws: str,
    ws: str,
    begin_tool_index: int,
    end_tool_index: int,
    expected_tokens_before_tool: list,
    expected_tokens_after_tool: list,
) -> None:
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
        f"<s>[INST]{special_ws}a[/INST]{special_ws}b</s>[AVAILABLE_TOOLS]{special_ws}["
        f'{{"type":{ws}"function",{ws}"function":{ws}{{"name":{ws}"tool1",{ws}"description":{ws}"1",{ws}"parameters":{ws}{{}}}}}},'
        f'{ws}{{"type":{ws}"function",{ws}"function":{ws}{{"name":{ws}"tool2",{ws}"description":{ws}"2",{ws}"parameters":{ws}{{}}}}}}]'
        f"[/AVAILABLE_TOOLS][INST]{special_ws}c[/INST]{special_ws}d</s>"
    )

    begin_tool, end_tool = tokens.index(begin_tool_index), tokens.index(end_tool_index)
    assert tokens[:begin_tool] + tokens[end_tool + 1 :] == expected_tokens_before_tool + expected_tokens_after_tool
    json.loads(tokenizer.tokenizer.decode(tokens[begin_tool : end_tool + 1]))


@pytest.mark.parametrize(
    "tokenizer, special_ws, ws, begin_tool_index, end_tool_index, new_line, decoded_before_tool, decoded_after_tool",
    [
        (
            tokenizer(),
            SPM_SPECIAL_WHITESPACE,
            SPM_WHITESPACE,
            SPM_BEGIN_TOOL_ID,
            SPM_END_TOOL_ID,
            "<0x0A><0x0A>",
            "a b",
            "SYSTEM\n\nc d",
        ),
        (
            tekken_tokenizer(),
            TEKKEN_SPECIAL_WHITESPACE,
            TEKKEN_WHITESPACE,
            TEKKEN_BEGIN_TOOL_ID,
            TEKKEN_END_TOOL_ID,
            "\n\n",
            "ab",
            "SYSTEM\n\ncd",
        ),
    ],
)
def test_system_tools_multiturn(
    tokenizer: InstructTokenizer,
    special_ws: str,
    ws: str,
    begin_tool_index: int,
    end_tool_index: int,
    new_line: str,
    decoded_before_tool: str,
    decoded_after_tool: str,
) -> None:
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
        f"<s>[INST]{special_ws}a[/INST]{special_ws}b</s>[AVAILABLE_TOOLS]{special_ws}["
        f'{{"type":{ws}"function",{ws}"function":{ws}{{"name":{ws}"tool1",{ws}"description":{ws}"1",{ws}"parameters":{ws}{{}}}}}}]'
        f"[/AVAILABLE_TOOLS][INST]{special_ws}SYSTEM{new_line}c[/INST]{special_ws}d</s>"
    )

    begin_tool, end_tool = tokens.index(begin_tool_index), tokens.index(end_tool_index)
    assert tokens[end_tool + 1 :].index(3) == 0  # begin_inst follows end_tool
    assert tokenizer.tokenizer.decode(tokens[:begin_tool]) == decoded_before_tool
    assert tokenizer.tokenizer.decode(tokens[end_tool + 1 :]) == decoded_after_tool


@pytest.mark.parametrize(
    "tokenizer, special_ws, new_line",
    [
        (
            tokenizer(),
            SPM_SPECIAL_WHITESPACE,
            "<0x0A><0x0A>",
        ),
        (
            tekken_tokenizer(),
            TEKKEN_SPECIAL_WHITESPACE,
            "\n\n",
        ),
    ],
)
def test_continue_final_message(
    tokenizer: InstructTokenizer,
    special_ws: str,
    new_line: str,
) -> None:
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
    assert text == (
        f"<s>[INST]{special_ws}a[/INST]{special_ws}b</s>[INST]{special_ws}SYSTEM{new_line}c[/INST]{special_ws}d"
    )
    if not isinstance(tokenizer.tokenizer, Tekkenizer):
        assert tokens == [1, 3, 1032, 4, 1055, 2, 3, 17889, 23294, 781, 781, 29485, 4, 1049]
    else:
        assert tokens == [1, 3, 1097, 4, 1098, 2, 3, 101289, 58343, 1267, 1099, 4, 1100]

    with pytest.raises(
        InvalidMessageStructureException, match="Cannot continue final message if it is not an assistant message"
    ):
        tokenized = tokenizer.encode_instruct(
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


@pytest.mark.parametrize(
    "tokenizer, special_ws, ws",
    [
        (
            tokenizer(),
            SPM_SPECIAL_WHITESPACE,
            SPM_WHITESPACE,
        ),
        (tekken_tokenizer(), TEKKEN_SPECIAL_WHITESPACE, TEKKEN_WHITESPACE),
    ],
)
def test_tool_message(tokenizer: InstructTokenizer, special_ws: str, ws: str) -> None:
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
        f'<s>[INST]{special_ws}a[/INST][TOOL_CALLS]{special_ws}[{{"name":{ws}"b",{ws}'
        f'"arguments":{ws}{{}},{ws}"id":{ws}"123456789"}}]</s>[TOOL_RESULTS]{special_ws}'
        f'{{"content":{ws}"d",{ws}"call_id":{ws}"123456789"}}[/TOOL_RESULTS]'
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
        f"<s>[INST]{special_ws}a[/INST][TOOL_CALLS]{special_ws}["
        f'{{"name":{ws}"b",{ws}"arguments":{ws}{{}},{ws}"id":{ws}"123456789"}}]</s>[TOOL_RESULTS]{special_ws}'
        f'{{"content":{ws}{{"a":{ws}1}},{ws}"call_id":{ws}"123456789"}}[/TOOL_RESULTS]'
    )


@pytest.mark.parametrize(
    "tokenizer, special_ws, ws",
    [
        (
            tokenizer(),
            SPM_SPECIAL_WHITESPACE,
            SPM_WHITESPACE,
        ),
        (tekken_tokenizer(), TEKKEN_SPECIAL_WHITESPACE, TEKKEN_WHITESPACE),
    ],
)
def test_tool_message_no_id_fine_tuning_ok(tokenizer: InstructTokenizer, special_ws: str, ws: str) -> None:
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
        assert (
            text
            == f'<s>[INST]{special_ws}a[/INST][TOOL_CALLS]{special_ws}[{{"name":{ws}"b",{ws}"arguments":{ws}{{}}}}]</s>'
        )


@pytest.mark.parametrize(
    "tokenizer, special_ws, ws",
    [
        (
            tokenizer(),
            SPM_SPECIAL_WHITESPACE,
            SPM_WHITESPACE,
        ),
        (
            tekken_tokenizer(),
            TEKKEN_SPECIAL_WHITESPACE,
            TEKKEN_WHITESPACE,
        ),
    ],
)
def test_tool_message_multiple_shots_with_history(tokenizer: InstructTokenizer, special_ws: str, ws: str) -> None:
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
        f"<s>[INST]{special_ws}a[/INST]"
        f'[TOOL_CALLS]{special_ws}[{{"name":{ws}"b",{ws}"arguments":{ws}{{}},{ws}"id":{ws}"0"}}]</s>[TOOL_RESULTS]{special_ws}{{"content":{ws}"d",{ws}"call_id":{ws}"0"}}[/TOOL_RESULTS]'  # noqa: E501
        f"{special_ws}e</s>[INST]{special_ws}f[/INST]"
        f'[TOOL_CALLS]{special_ws}[{{"name":{ws}"b",{ws}"arguments":{ws}{{}},{ws}"id":{ws}"1"}}]</s>[TOOL_RESULTS]{special_ws}{{"content":{ws}"d",{ws}"call_id":{ws}"1"}}[/TOOL_RESULTS]'
    )


@pytest.mark.parametrize(
    "tokenizer, special_ws, ws",
    [
        (tokenizer(), SPM_SPECIAL_WHITESPACE, SPM_WHITESPACE),
        (tekken_tokenizer(), TEKKEN_SPECIAL_WHITESPACE, TEKKEN_WHITESPACE),
    ],
)
def test_tool_message_multiple_calls(tokenizer: InstructTokenizer, special_ws: str, ws: str) -> None:
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
        f"<s>[INST]{special_ws}a[/INST]"
        f'[TOOL_CALLS]{special_ws}[{{"name":{ws}"b",{ws}"arguments":{ws}{{}},{ws}"id":{ws}"0"}},{ws}{{"name":{ws}"q",{ws}"arguments":{ws}{{}},{ws}"id":{ws}"1"}}]</s>'
        f'[TOOL_RESULTS]{special_ws}{{"content":{ws}"d",{ws}"call_id":{ws}"0"}}[/TOOL_RESULTS]'
        f'[TOOL_RESULTS]{special_ws}{{"content":{ws}"d",{ws}"call_id":{ws}"1"}}[/TOOL_RESULTS]'
        f"{special_ws}e</s>[INST]{special_ws}f[/INST]"
        f'[TOOL_CALLS]{special_ws}[{{"name":{ws}"b",{ws}"arguments":{ws}{{}},{ws}"id":{ws}"2"}},{ws}{{"name":{ws}"q",{ws}"arguments":{ws}{{}},{ws}"id":{ws}"3"}}]</s>'
        f'[TOOL_RESULTS]{special_ws}{{"content":{ws}"d",{ws}"call_id":{ws}"2"}}[/TOOL_RESULTS]'
        f'[TOOL_RESULTS]{special_ws}{{"content":{ws}"d",{ws}"call_id":{ws}"3"}}[/TOOL_RESULTS]'
    )


@pytest.mark.parametrize("tokenizer", [tokenizer(), tekken_tokenizer()])
def test_assistant_tool_call_and_content(tokenizer: InstructTokenizer) -> None:
    req: InstructRequest = InstructRequest(
        messages=[
            UserMessage(content="a"),
            AssistantMessage(
                content="b",
                tool_calls=[
                    ToolCall(id="0", function=FunctionCall(name="b", arguments="{}")),
                ],
            ),
        ],
    )

    with pytest.raises(ValueError, match="Cannot have tool calls and content defined in the same assistant message"):
        tokenizer.encode_instruct(req)

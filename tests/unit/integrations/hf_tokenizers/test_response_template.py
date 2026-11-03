import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mistral_common.experimental.think import _split_content_and_think_chunks
from mistral_common.experimental.tools import _decode_tool_calls, _split_content_and_tool_calls
from mistral_common.integrations.hf_tokenizers.response_template import (
    convert_tokenizer_to_response_template,
    generate_response_template,
)
from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import (
    PLAIN_BEGIN_THINK,
    PLAIN_END_THINK,
    SpecialTokenPolicy,
    SpecialTokens,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integrations.chat_templates.helpers import TestConfig, build_tekken_json
from tests.unit.integrations.hf_tokenizers.fixtures_response_template import (
    PR_V1,
    PR_V1_JSON,
    PR_V7,
    PR_V7_JSON,
    PR_V13,
    PR_V13_JSON,
    V11_NO_THINKING,
    V11_PLAIN_THINKING,
)


def test_top_level_keys_and_json_roundtrip() -> None:
    configurations = (
        (TokenizerVersion.v1, False, False),
        (TokenizerVersion.v2, False, False),
        (TokenizerVersion.v3, False, False),
        (TokenizerVersion.v7, False, False),
        (TokenizerVersion.v11, False, False),
        (TokenizerVersion.v13, False, False),
        (TokenizerVersion.v15, False, False),
        (TokenizerVersion.v11, False, True),
        (TokenizerVersion.v13, True, False),
    )
    for version, thinking_support, plain_thinking_support in configurations:
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=thinking_support,
            plain_thinking_support=plain_thinking_support,
        )

        assert json.loads(s=json.dumps(obj=template)) == template
        assert set(template) == {"defaults", "start_anchor", "fields"}
        assert template["defaults"] == {"role": "assistant"}


@pytest.mark.parametrize(
    ("version", "thinking_support", "plain_thinking_support", "expected", "expected_json"),
    [
        (TokenizerVersion.v1, False, False, PR_V1, PR_V1_JSON),
        (TokenizerVersion.v7, False, False, PR_V7, PR_V7_JSON),
        (TokenizerVersion.v13, True, False, PR_V13, PR_V13_JSON),
        (TokenizerVersion.v11, False, True, V11_PLAIN_THINKING, None),
        (TokenizerVersion.v11, False, False, V11_NO_THINKING, None),
    ],
    ids=["v1-pr", "v7-pr", "v13-pr", "v11-plain-thinking", "v11-no-thinking"],
)
def test_expected_fixtures(
    version: TokenizerVersion,
    thinking_support: bool,
    plain_thinking_support: bool,
    expected: Any,
    expected_json: str | None,
) -> None:
    template = generate_response_template(
        tokenizer_version=version,
        thinking_support=thinking_support,
        plain_thinking_support=plain_thinking_support,
    )
    assert template == expected
    if expected_json is not None:
        assert json.dumps(obj=template, indent=2) == expected_json


def test_fixtures_are_immutable() -> None:
    with pytest.raises(TypeError):
        PR_V13["defaults"] = {"role": "user"}  # type: ignore[index]
    with pytest.raises(TypeError):
        PR_V7["fields"]["content"]["close"] = []
    with pytest.raises(TypeError):
        V11_PLAIN_THINKING["fields"]["reasoning_content"]["open"] = "x"


def test_start_anchor_per_version() -> None:
    v1 = generate_response_template(
        tokenizer_version=TokenizerVersion.v1,
        thinking_support=False,
        plain_thinking_support=False,
    )
    assert v1["start_anchor"] == "[/INST]"

    for version in (
        TokenizerVersion.v2,
        TokenizerVersion.v3,
        TokenizerVersion.v7,
        TokenizerVersion.v11,
        TokenizerVersion.v13,
        TokenizerVersion.v15,
    ):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        assert template["start_anchor"] == ["[/INST]", "[/TOOL_RESULTS]"]


def test_v13_v15_tool_calls_pattern() -> None:
    expected_tool_calls = {
        "open_pattern": "\\[TOOL_CALLS\\](?P<name>.+?)\\[ARGS\\]",
        "close_pattern": "(?=\\[TOOL_CALLS\\])|</s>",
        "repeats": True,
        "content": "json",
        "transform": {
            "type": "function",
            "function": {"name": "{name}", "arguments": "{content}"},
        },
    }

    for version in (TokenizerVersion.v13, TokenizerVersion.v15):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=True,
            plain_thinking_support=False,
        )
        assert template["fields"]["tool_calls"] == expected_tool_calls


def test_v11_tool_calls_include_call_id() -> None:
    template = generate_response_template(
        tokenizer_version=TokenizerVersion.v11,
        thinking_support=False,
        plain_thinking_support=True,
    )

    assert template == V11_PLAIN_THINKING
    tool_calls = template["fields"]["tool_calls"]
    assert tool_calls == {
        "open_pattern": "\\[TOOL_CALLS\\](?P<name>.+?)\\[CALL_ID\\](?P<id>.+?)\\[ARGS\\]",
        "close_pattern": "(?=\\[TOOL_CALLS\\])|</s>",
        "repeats": True,
        "content": "json",
        "transform": {
            "type": "function",
            "id": "{id}",
            "function": {"name": "{name}", "arguments": "{content}"},
        },
    }
    assert template["fields"]["reasoning_content"] == {
        "open": PLAIN_BEGIN_THINK,
        "close": PLAIN_END_THINK,
        "content": "text",
    }

    assert (
        generate_response_template(
            tokenizer_version=TokenizerVersion.v11,
            thinking_support=False,
            plain_thinking_support=False,
        )
        == V11_NO_THINKING
    )
    assert "reasoning_content" not in V11_NO_THINKING["fields"]


def test_v3_v7_tool_calls_json_array() -> None:
    expected = {
        "open": "[TOOL_CALLS]",
        "close": "</s>",
        "content": "json",
        "transform_each": True,
        "transform": {
            "id": "{id}",
            "type": "function",
            "function": {"name": "{name}", "arguments": "{arguments}"},
        },
    }

    for version in (TokenizerVersion.v3, TokenizerVersion.v7):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        assert template["fields"]["tool_calls"] == expected


def test_v2_tool_calls_no_id() -> None:
    template = generate_response_template(
        tokenizer_version=TokenizerVersion.v2,
        thinking_support=False,
        plain_thinking_support=False,
    )

    assert template["fields"]["tool_calls"]["transform"] == {
        "type": "function",
        "function": {"name": "{name}", "arguments": "{arguments}"},
    }


def test_reasoning_field_rules() -> None:
    expected_reasoning = {"open": "[THINK]", "close": "[/THINK]", "content": "text"}
    for version in (TokenizerVersion.v13, TokenizerVersion.v15):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        assert template["fields"]["reasoning_content"] == expected_reasoning

    for version in (TokenizerVersion.v2, TokenizerVersion.v3, TokenizerVersion.v7):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        assert "reasoning_content" not in template["fields"]


def test_content_field() -> None:
    v1 = generate_response_template(
        tokenizer_version=TokenizerVersion.v1,
        thinking_support=False,
        plain_thinking_support=False,
    )
    assert v1["fields"]["content"] == {"close": "</s>", "content": "text"}

    for version in (
        TokenizerVersion.v2,
        TokenizerVersion.v3,
        TokenizerVersion.v7,
        TokenizerVersion.v11,
        TokenizerVersion.v13,
        TokenizerVersion.v15,
    ):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        content: dict[str, Any] = template["fields"]["content"]
        assert "open" not in content
        assert "open_pattern" not in content
        assert content["close"] == ["[TOOL_CALLS]", "</s>"]
        assert content["repeats"] is True
        assert content["join"] == ""
        assert content["content"] == "text"


def test_invalid_thinking_combos_raise() -> None:
    with pytest.raises(ValueError, match="Plain thinking support is only available"):
        generate_response_template(
            tokenizer_version=TokenizerVersion.v7,
            thinking_support=False,
            plain_thinking_support=True,
        )

    with pytest.raises(ValueError, match="Special-token thinking support requires"):
        generate_response_template(
            tokenizer_version=TokenizerVersion.v7,
            thinking_support=True,
            plain_thinking_support=False,
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_response_template(
            tokenizer_version=TokenizerVersion.v13,
            thinking_support=True,
            plain_thinking_support=True,
        )


def test_no_chat_templates_import() -> None:
    module_import = (
        "import sys; import mistral_common.integrations.hf_tokenizers.response_template; "
        "assert 'mistral_common.integrations.chat_templates' not in sys.modules"
    )
    result = subprocess.run(
        args=[sys.executable, "-c", module_import],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[4],
    )

    assert result.returncode == 0, result.stderr


def test_no_delimiter_string_literals() -> None:
    module_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "mistral_common"
        / "integrations"
        / "hf_tokenizers"
        / "response_template.py"
    )
    source = module_path.read_text(encoding="utf-8")

    for delimiter in (
        SpecialTokens.tool_calls.value,
        SpecialTokens.args.value,
        SpecialTokens.call_id.value,
        SpecialTokens.begin_think.value,
        SpecialTokens.end_think.value,
        SpecialTokens.end_inst.value,
        SpecialTokens.end_tool_results.value,
        SpecialTokens.eos.value,
    ):
        assert delimiter not in source
        escaped_delimiter = re.escape(delimiter)
        assert escaped_delimiter not in source
        assert escaped_delimiter.replace("\\", "\\\\") not in source
    assert "re.escape" in source


def test_convert_tokenizer_to_response_template_tekken_v13(tmp_path: Path) -> None:
    path = build_tekken_json(
        config=TestConfig(version=TokenizerVersion.v13, think=True),
        output_dir=tmp_path,
    )

    result = convert_tokenizer_to_response_template(tokenizer_file=path, plain_thinking_support=None)
    expected = generate_response_template(
        tokenizer_version=TokenizerVersion.v13,
        thinking_support=True,
        plain_thinking_support=False,
    )
    assert result == expected


def test_convert_v11_plain_thinking_heuristic(tmp_path: Path) -> None:
    path = build_tekken_json(
        config=TestConfig(version=TokenizerVersion.v11),
        output_dir=tmp_path,
    )

    heuristic_result = convert_tokenizer_to_response_template(
        tokenizer_file=path,
        plain_thinking_support=None,
    )
    assert heuristic_result["fields"]["reasoning_content"]["open"] == "<think>"

    override_result = convert_tokenizer_to_response_template(
        tokenizer_file=path,
        plain_thinking_support=False,
    )
    assert "reasoning_content" not in override_result["fields"]


def test_matches_experimental_parser_v13(tmp_path: Path) -> None:
    tokenizer_path = build_tekken_json(
        config=TestConfig(version=TokenizerVersion.v13, think=True),
        output_dir=tmp_path,
    )
    mistral_tokenizer = MistralTokenizer.from_file(
        tokenizer_filename=str(tokenizer_path),
        mode=ValidationMode.test,
    )
    messages: list[ChatMessage] = [
        UserMessage(content="hi"),
        AssistantMessage(
            content=[TextChunk(text="Sure."), ThinkChunk(thinking="let me check", closed=True)],
            tool_calls=[
                ToolCall(
                    id="abcdefghi",
                    function=FunctionCall(name="get_weather", arguments=json.dumps({"city": "Paris"})),
                ),
                ToolCall(
                    id="123456789",
                    function=FunctionCall(name="get_time", arguments=json.dumps({"city": "Paris"})),
                ),
            ],
            prefix=True,
        ),
    ]
    request = ChatCompletionRequest(messages=messages)
    tokens = mistral_tokenizer.encode_chat_completion(request).tokens
    tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer
    end_inst_id = tokenizer.get_special_token(SpecialTokens.end_inst.value)
    generation_start = max(index for index, token in enumerate(tokens) if token == end_inst_id) + 1
    generation = tokens[generation_start:]
    tool_calls_id = tokenizer.get_special_token(SpecialTokens.tool_calls.value)
    content_tokens, tool_call_tokens = _split_content_and_tool_calls(
        tokens=generation,
        tool_call_token_id=tool_calls_id,
    )
    decoded_tool_calls = _decode_tool_calls(tool_call_tokens=tool_call_tokens, tokenizer=tokenizer)
    assert [call.function.name for call in decoded_tool_calls] == ["get_weather", "get_time"]
    response_template = generate_response_template(
        tokenizer_version=TokenizerVersion.v13,
        thinking_support=True,
        plain_thinking_support=False,
    )
    tool_call_field = response_template["fields"]["tool_calls"]
    for call_tokens, decoded_tool_call in zip(tool_call_tokens, decoded_tool_calls, strict=True):
        decoded_call_region = tokenizer.decode(tokens=call_tokens, special_token_policy=SpecialTokenPolicy.KEEP)
        match = re.fullmatch(
            pattern=tool_call_field["open_pattern"] + ".*",
            string=decoded_call_region,
            flags=re.DOTALL,
        )
        assert match is not None
        assert match.group("name") == decoded_tool_call.function.name

    think_chunks = _split_content_and_think_chunks(
        tokens=content_tokens,
        begin_think_token_id=tokenizer.get_special_token(SpecialTokens.begin_think.value),
        end_think_token_id=tokenizer.get_special_token(SpecialTokens.end_think.value),
    )
    decoded_think_chunks = [
        tokenizer.decode(tokens=chunk_tokens, special_token_policy=SpecialTokenPolicy.KEEP)
        for chunk_tokens, is_think in think_chunks
        if is_think
    ]

    assert decoded_think_chunks == ["[THINK]let me check[/THINK]"]
    assert response_template["fields"]["reasoning_content"]["open"] == "[THINK]"
    assert response_template["fields"]["reasoning_content"]["close"] == "[/THINK]"
    assert response_template["fields"]["content"]["close"] == ["[TOOL_CALLS]", "</s>"]


def test_matches_experimental_parser_v7(tmp_path: Path) -> None:
    tokenizer_path = build_tekken_json(
        config=TestConfig(version=TokenizerVersion.v7),
        output_dir=tmp_path,
    )
    mistral_tokenizer = MistralTokenizer.from_file(
        tokenizer_filename=str(tokenizer_path),
        mode=ValidationMode.finetuning,
    )
    messages: list[ChatMessage] = [
        UserMessage(content="hi"),
        AssistantMessage(
            content="Sure.",
            tool_calls=[
                ToolCall(
                    id="abcdefghi",
                    function=FunctionCall(name="get_weather", arguments=json.dumps({"city": "Paris"})),
                )
            ],
        ),
    ]
    request = ChatCompletionRequest(messages=messages)
    tokens = mistral_tokenizer.encode_chat_completion(request).tokens
    tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer
    end_inst_id = tokenizer.get_special_token(SpecialTokens.end_inst.value)
    generation_start = max(index for index, token in enumerate(tokens) if token == end_inst_id) + 1
    generation = tokens[generation_start:]
    tool_calls_id = tokenizer.get_special_token(SpecialTokens.tool_calls.value)
    _, tool_call_tokens = _split_content_and_tool_calls(tokens=generation, tool_call_token_id=tool_calls_id)
    decoded_tool_calls = _decode_tool_calls(tool_call_tokens=tool_call_tokens, tokenizer=tokenizer)
    assert len(decoded_tool_calls) == 1
    assert decoded_tool_calls[0].function.name == "get_weather"
    response_template = generate_response_template(
        tokenizer_version=TokenizerVersion.v7,
        thinking_support=False,
        plain_thinking_support=False,
    )
    expected_transform = {
        "id": "{id}",
        "type": "function",
        "function": {"name": "{name}", "arguments": "{arguments}"},
    }
    assert response_template["fields"]["tool_calls"] == {
        "open": "[TOOL_CALLS]",
        "close": "</s>",
        "content": "json",
        "transform_each": True,
        "transform": expected_transform,
    }
    decoded_generation = tokenizer.decode(tokens=generation, special_token_policy=SpecialTokenPolicy.KEEP)
    call_start = decoded_generation.index(SpecialTokens.tool_calls.value) + len(SpecialTokens.tool_calls.value)
    call_end = decoded_generation.index(SpecialTokens.eos.value, call_start)
    call_array_region = decoded_generation[call_start:call_end]
    assert json.loads(call_array_region) == [{"id": "abcdefghi", "name": "get_weather", "arguments": {"city": "Paris"}}]


def test_matches_experimental_parser_v11_tool_calls(tmp_path: Path) -> None:
    tokenizer_path = build_tekken_json(
        config=TestConfig(version=TokenizerVersion.v11),
        output_dir=tmp_path,
    )
    mistral_tokenizer = MistralTokenizer.from_file(
        tokenizer_filename=str(tokenizer_path),
        mode=ValidationMode.test,
    )
    messages: list[ChatMessage] = [
        UserMessage(content="hi"),
        AssistantMessage(
            tool_calls=[
                ToolCall(
                    id="abcdefghi",
                    function=FunctionCall(name="get_weather", arguments=json.dumps({"city": "Paris"})),
                ),
                ToolCall(
                    id="123456789",
                    function=FunctionCall(name="get_time", arguments=json.dumps({"city": "Paris"})),
                ),
            ],
            prefix=True,
        ),
    ]
    request = ChatCompletionRequest(messages=messages)
    tokens = mistral_tokenizer.encode_chat_completion(request).tokens
    tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer
    end_inst_id = tokenizer.get_special_token(SpecialTokens.end_inst.value)
    generation_start = max(index for index, token in enumerate(tokens) if token == end_inst_id) + 1
    generation = tokens[generation_start:]
    tool_calls_id = tokenizer.get_special_token(SpecialTokens.tool_calls.value)
    _, tool_call_tokens = _split_content_and_tool_calls(tokens=generation, tool_call_token_id=tool_calls_id)
    decoded_tool_calls = _decode_tool_calls(tool_call_tokens=tool_call_tokens, tokenizer=tokenizer)
    assert [call.function.name for call in decoded_tool_calls] == ["get_weather", "get_time"]
    tool_call_field = generate_response_template(
        tokenizer_version=TokenizerVersion.v11,
        thinking_support=False,
        plain_thinking_support=False,
    )["fields"]["tool_calls"]

    for call_tokens, decoded_tool_call in zip(tool_call_tokens, decoded_tool_calls, strict=True):
        decoded_call_region = tokenizer.decode(tokens=call_tokens, special_token_policy=SpecialTokenPolicy.KEEP)
        match = re.fullmatch(
            pattern=tool_call_field["open_pattern"] + ".*",
            string=decoded_call_region,
            flags=re.DOTALL,
        )
        assert match is not None
        assert match.group("name") == decoded_tool_call.function.name
        assert match.group("id") == decoded_tool_call.id

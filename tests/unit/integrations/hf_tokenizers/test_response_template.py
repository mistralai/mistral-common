import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import mistral_common.integrations.hf_tokenizers.response_template
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

_COMMON_CONTENT_FIELD: dict[str, Any] = {
    "close": [SpecialTokens.tool_calls.value, SpecialTokens.eos.value],
    "content": "text",
    "repeats": True,
    "join": "",
}

_V11_TOOL_CALLS: dict[str, Any] = {
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


@pytest.fixture(scope="module")
def expected_v1() -> dict[str, Any]:
    return {
        "defaults": {"role": "assistant"},
        "start_anchor": SpecialTokens.end_inst.value,
        "fields": {"content": {"close": SpecialTokens.eos.value, "content": "text"}},
    }


@pytest.fixture(scope="module")
def expected_v7() -> dict[str, Any]:
    return {
        "defaults": {"role": "assistant"},
        "start_anchor": [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value],
        "fields": {
            "tool_calls": {
                "open": SpecialTokens.tool_calls.value,
                "close": SpecialTokens.eos.value,
                "content": "json",
                "transform_each": True,
                "transform": {
                    "id": "{id}",
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{arguments}"},
                },
            },
            "content": _COMMON_CONTENT_FIELD,
        },
    }


@pytest.fixture(scope="module")
def expected_v13() -> dict[str, Any]:
    return {
        "defaults": {"role": "assistant"},
        "start_anchor": [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value],
        "fields": {
            "reasoning_content": {
                "open": SpecialTokens.begin_think.value,
                "close": SpecialTokens.end_think.value,
                "content": "text",
            },
            "tool_calls": {
                "open_pattern": "\\[TOOL_CALLS\\](?P<name>.+?)\\[ARGS\\]",
                "close_pattern": "(?=\\[TOOL_CALLS\\])|</s>",
                "repeats": True,
                "content": "json",
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"},
                },
            },
            "content": _COMMON_CONTENT_FIELD,
        },
    }


@pytest.fixture(scope="module")
def expected_v11_plain_thinking() -> dict[str, Any]:
    return {
        "defaults": {"role": "assistant"},
        "start_anchor": [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value],
        "fields": {
            "reasoning_content": {
                "open": PLAIN_BEGIN_THINK,
                "close": PLAIN_END_THINK,
                "content": "text",
            },
            "tool_calls": _V11_TOOL_CALLS,
            "content": _COMMON_CONTENT_FIELD,
        },
    }


@pytest.fixture(scope="module")
def expected_v11_no_thinking() -> dict[str, Any]:
    return {
        "defaults": {"role": "assistant"},
        "start_anchor": [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value],
        "fields": {
            "tool_calls": _V11_TOOL_CALLS,
            "content": _COMMON_CONTENT_FIELD,
        },
    }


def test_expected_v1(expected_v1: dict[str, Any]) -> None:
    assert (
        generate_response_template(
            tokenizer_version=TokenizerVersion.v1,
            thinking_support=False,
            plain_thinking_support=False,
        )
        == expected_v1
    )


def test_expected_v7(expected_v7: dict[str, Any]) -> None:
    assert (
        generate_response_template(
            tokenizer_version=TokenizerVersion.v7,
            thinking_support=False,
            plain_thinking_support=False,
        )
        == expected_v7
    )


def test_expected_v13(expected_v13: dict[str, Any]) -> None:
    assert (
        generate_response_template(
            tokenizer_version=TokenizerVersion.v13,
            thinking_support=True,
            plain_thinking_support=False,
        )
        == expected_v13
    )


def test_expected_v11_plain_thinking(expected_v11_plain_thinking: dict[str, Any]) -> None:
    assert (
        generate_response_template(
            tokenizer_version=TokenizerVersion.v11,
            thinking_support=False,
            plain_thinking_support=True,
        )
        == expected_v11_plain_thinking
    )


def test_expected_v11_no_thinking(expected_v11_no_thinking: dict[str, Any]) -> None:
    template = generate_response_template(
        tokenizer_version=TokenizerVersion.v11,
        thinking_support=False,
        plain_thinking_support=False,
    )
    assert template == expected_v11_no_thinking
    assert "reasoning_content" not in template["fields"]


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


def test_v11_tool_calls_include_call_id(
    expected_v11_plain_thinking: dict[str, Any], expected_v11_no_thinking: dict[str, Any]
) -> None:
    template = generate_response_template(
        tokenizer_version=TokenizerVersion.v11,
        thinking_support=False,
        plain_thinking_support=True,
    )

    assert template == expected_v11_plain_thinking
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
        == expected_v11_no_thinking
    )


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


def test_reasoning_field_rules(
    expected_v11_plain_thinking: dict[str, Any], expected_v11_no_thinking: dict[str, Any]
) -> None:
    for version in (TokenizerVersion.v13, TokenizerVersion.v15):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        assert template["fields"]["reasoning_content"] == {
            "open": "[THINK]",
            "close": "[/THINK]",
            "content": "text",
        }
    for version in (TokenizerVersion.v2, TokenizerVersion.v3, TokenizerVersion.v7):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        assert "reasoning_content" not in template["fields"]
    assert "reasoning_content" in expected_v11_plain_thinking["fields"]
    assert "reasoning_content" not in expected_v11_no_thinking["fields"]


def test_content_field(expected_v1: dict[str, Any]) -> None:
    v1 = generate_response_template(
        tokenizer_version=TokenizerVersion.v1,
        thinking_support=False,
        plain_thinking_support=False,
    )
    assert v1["fields"]["content"] == {"close": "</s>", "content": "text"}
    assert v1 == expected_v1

    for version in (TokenizerVersion.v2, TokenizerVersion.v3, TokenizerVersion.v7, TokenizerVersion.v11):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        content = template["fields"]["content"]
        assert "open" not in content and "open_pattern" not in content
        assert content["close"] == ["[TOOL_CALLS]", "</s>"]
        assert content["repeats"] is True
        assert content["join"] == ""
        assert content["content"] == "text"


def test_invalid_thinking_combos_raise() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_response_template(
            tokenizer_version=TokenizerVersion.v13,
            thinking_support=True,
            plain_thinking_support=True,
        )
    with pytest.raises(ValueError, match="only available for tokenizer version v11"):
        generate_response_template(
            tokenizer_version=TokenizerVersion.v7,
            thinking_support=False,
            plain_thinking_support=True,
        )
    with pytest.raises(ValueError, match="v13 or above"):
        generate_response_template(
            tokenizer_version=TokenizerVersion.v7,
            thinking_support=True,
            plain_thinking_support=False,
        )


def test_no_chat_templates_import() -> None:
    result = subprocess.run(
        args=[
            sys.executable,
            "-c",
            "import sys; import mistral_common.integrations.hf_tokenizers.response_template; "
            "assert 'mistral_common.integrations.chat_templates' not in sys.modules",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_no_delimiter_string_literals() -> None:
    source = Path(mistral_common.integrations.hf_tokenizers.response_template.__file__).read_text(encoding="utf-8")
    for token in (
        SpecialTokens.tool_calls.value,
        SpecialTokens.args.value,
        SpecialTokens.call_id.value,
        SpecialTokens.begin_think.value,
        SpecialTokens.end_think.value,
        SpecialTokens.end_inst.value,
        SpecialTokens.end_tool_results.value,
        SpecialTokens.eos.value,
    ):
        assert token not in source
        assert re.escape(token) not in source
        assert re.escape(re.escape(token)) not in source
    assert "re.escape" in source


def test_convert_tokenizer_to_response_template_tekken_v13(tmp_path: Path) -> None:
    path = build_tekken_json(TestConfig(version=TokenizerVersion.v13, think=True), output_dir=tmp_path)
    result = convert_tokenizer_to_response_template(tokenizer_file=path, plain_thinking_support=None)
    assert result == generate_response_template(
        tokenizer_version=TokenizerVersion.v13,
        thinking_support=True,
        plain_thinking_support=False,
    )


def test_convert_v11_plain_thinking_heuristic(
    tmp_path: Path, expected_v11_plain_thinking: dict[str, Any], expected_v11_no_thinking: dict[str, Any]
) -> None:
    path = build_tekken_json(TestConfig(version=TokenizerVersion.v11), output_dir=tmp_path)
    auto = convert_tokenizer_to_response_template(tokenizer_file=path, plain_thinking_support=None)
    assert auto == expected_v11_plain_thinking
    forced_off = convert_tokenizer_to_response_template(tokenizer_file=path, plain_thinking_support=False)
    assert forced_off == expected_v11_no_thinking


def _encode_assistant_generation(
    version: TokenizerVersion,
    think: bool,
    tmp_path: Path,
    assistant: AssistantMessage,
) -> tuple[MistralTokenizer, list[int], str]:
    r"""Encode [user, assistant] and return (tokenizer, generation tokens, decoded KEEP text)."""
    path = build_tekken_json(TestConfig(version=version, think=think), output_dir=tmp_path)
    tokenizer = MistralTokenizer.from_file(str(path), mode=ValidationMode.test)
    messages: list[ChatMessage] = [UserMessage(content="hi"), assistant]
    request = ChatCompletionRequest(messages=messages)
    tokens = tokenizer.encode_chat_completion(request).tokens
    inst_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.end_inst.value)
    last_inst = len(tokens) - 1 - tokens[::-1].index(inst_id)
    generation = tokens[last_inst + 1 :]
    decoded = tokenizer.decode(generation, special_token_policy=SpecialTokenPolicy.KEEP)
    return tokenizer, generation, decoded


def _sample_tool_calls() -> list[ToolCall]:
    return [
        ToolCall(id="abcdefghi", function=FunctionCall(name="get_weather", arguments=json.dumps({"city": "Paris"}))),
        ToolCall(id="abcdefghj", function=FunctionCall(name="get_time", arguments=json.dumps({"tz": "CET"}))),
    ]


def test_matches_experimental_parser_v13(tmp_path: Path) -> None:
    assistant = AssistantMessage(
        content=[TextChunk(text="Sure."), ThinkChunk(thinking="let me check")],
        tool_calls=_sample_tool_calls(),
        prefix=True,
    )
    tokenizer, generation, decoded = _encode_assistant_generation(TokenizerVersion.v13, True, tmp_path, assistant)
    template = generate_response_template(
        tokenizer_version=TokenizerVersion.v13, thinking_support=True, plain_thinking_support=False
    )

    tool_call_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.tool_calls.value)
    content_tokens, tool_call_tokens = _split_content_and_tool_calls(generation, tool_call_id)
    think_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.begin_think.value)
    end_think_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.end_think.value)
    chunks = _split_content_and_think_chunks(content_tokens, think_id, end_think_id)
    tool_calls = _decode_tool_calls(tool_call_tokens, tokenizer.instruct_tokenizer.tokenizer)

    open_pattern = template["fields"]["tool_calls"]["open_pattern"]
    for tool_call_tokens_i, tool_call in zip(tool_call_tokens, tool_calls):
        region = tokenizer.decode(tool_call_tokens_i, special_token_policy=SpecialTokenPolicy.KEEP)
        match = re.fullmatch(open_pattern + ".*", region)
        assert match is not None
        assert match.group("name") == tool_call.function.name
    assert template["fields"]["reasoning_content"]["open"] == SpecialTokens.begin_think.value
    assert template["fields"]["content"]["close"][0] == SpecialTokens.tool_calls.value
    assert any(is_think for _, is_think in chunks)


def test_matches_experimental_parser_v7(tmp_path: Path) -> None:
    assistant = AssistantMessage(content="Checking.", tool_calls=_sample_tool_calls()[:1], prefix=True)
    tokenizer, generation, decoded = _encode_assistant_generation(TokenizerVersion.v7, False, tmp_path, assistant)
    template = generate_response_template(
        tokenizer_version=TokenizerVersion.v7, thinking_support=False, plain_thinking_support=False
    )

    tool_call_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.tool_calls.value)
    _, tool_call_tokens = _split_content_and_tool_calls(generation, tool_call_id)
    tool_calls = _decode_tool_calls(tool_call_tokens, tokenizer.instruct_tokenizer.tokenizer)

    assert template["fields"]["tool_calls"]["open"] == SpecialTokens.tool_calls.value
    assert template["fields"]["tool_calls"]["transform_each"] is True
    region = tokenizer.decode(tool_call_tokens[0], special_token_policy=SpecialTokenPolicy.KEEP)
    payload = json.loads(region[len(SpecialTokens.tool_calls.value) :].strip().rstrip(SpecialTokens.eos.value))
    assert isinstance(payload, list) and len(payload) == len(tool_calls)
    for decoded_element, tool_call in zip(payload, tool_calls):
        assert set(decoded_element) == {"id", "name", "arguments"}
        assert decoded_element["name"] == tool_call.function.name
        assert decoded_element["id"] == tool_call.id


def test_matches_experimental_parser_v11_tool_calls(tmp_path: Path) -> None:
    assistant = AssistantMessage(content="Checking.", tool_calls=_sample_tool_calls(), prefix=True)
    tokenizer, generation, decoded = _encode_assistant_generation(TokenizerVersion.v11, False, tmp_path, assistant)
    template = generate_response_template(
        tokenizer_version=TokenizerVersion.v11, thinking_support=False, plain_thinking_support=False
    )

    tool_call_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.tool_calls.value)
    _, tool_call_tokens = _split_content_and_tool_calls(generation, tool_call_id)
    tool_calls = _decode_tool_calls(tool_call_tokens, tokenizer.instruct_tokenizer.tokenizer)

    open_pattern = template["fields"]["tool_calls"]["open_pattern"]
    for tool_call_tokens_i, tool_call in zip(tool_call_tokens, tool_calls):
        region = tokenizer.decode(tool_call_tokens_i, special_token_policy=SpecialTokenPolicy.KEEP)
        match = re.fullmatch(open_pattern + ".*", region)
        assert match is not None
        assert match.group("name") == tool_call.function.name
        assert match.group("id") == tool_call.id

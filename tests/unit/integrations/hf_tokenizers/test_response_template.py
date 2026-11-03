import json
import re
from pathlib import Path
from typing import Any

import pytest

from mistral_common.experimental.think import _split_content_and_think_chunks
from mistral_common.experimental.tools import _decode_tool_calls, _split_content_and_tool_calls
from mistral_common.integrations.hf_tokenizers.response_template import (
    ThinkingMode,
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

_TOOL_ANCHORS = [
    SpecialTokens.end_system.value,
    SpecialTokens.end_inst.value,
    SpecialTokens.end_tool_results.value,
]
_NO_TOOL_ANCHORS = [SpecialTokens.end_system.value, SpecialTokens.end_inst.value]


def _v11_tool_calls() -> dict[str, Any]:
    return {
        "open_pattern": re.escape(SpecialTokens.tool_calls.value)
        + "(?P<name>.+?)"
        + "(?:"
        + re.escape(SpecialTokens.call_id.value)
        + ")?"
        + "(?P<id>(?<="
        + re.escape(SpecialTokens.call_id.value)
        + ").+?|)"
        + re.escape(SpecialTokens.args.value),
        "close_pattern": "(?=" + re.escape(SpecialTokens.tool_calls.value) + ")|" + re.escape(SpecialTokens.eos.value),
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
        "start_anchor": _NO_TOOL_ANCHORS,
        "fields": {"content": {"close": SpecialTokens.eos.value, "content": "text"}},
    }


@pytest.fixture(scope="module")
def expected_v7() -> dict[str, Any]:
    return {
        "defaults": {"role": "assistant"},
        "start_anchor": _TOOL_ANCHORS,
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
        "start_anchor": _TOOL_ANCHORS,
        "fields": {
            "reasoning_content": {
                "open": SpecialTokens.begin_think.value,
                "close": SpecialTokens.end_think.value,
                "content": "text",
            },
            "tool_calls": {
                "open_pattern": re.escape(SpecialTokens.tool_calls.value)
                + "(?P<name>.+?)"
                + re.escape(SpecialTokens.args.value),
                "close_pattern": "(?="
                + re.escape(SpecialTokens.tool_calls.value)
                + ")|"
                + re.escape(SpecialTokens.eos.value),
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
        "start_anchor": _TOOL_ANCHORS,
        "fields": {
            "reasoning_content": {
                "open": PLAIN_BEGIN_THINK,
                "close": PLAIN_END_THINK,
                "content": "text",
            },
            "tool_calls": _v11_tool_calls(),
            "content": _COMMON_CONTENT_FIELD,
        },
    }


@pytest.fixture(scope="module")
def expected_v11_no_thinking() -> dict[str, Any]:
    return {
        "defaults": {"role": "assistant"},
        "start_anchor": _TOOL_ANCHORS,
        "fields": {
            "tool_calls": _v11_tool_calls(),
            "content": _COMMON_CONTENT_FIELD,
        },
    }


def _pre_v11_tool_calls(*, with_id: bool) -> dict[str, Any]:
    transform: dict[str, Any] = {"id": "{id}"} if with_id else {}
    transform["type"] = "function"
    transform["function"] = {"name": "{name}", "arguments": "{arguments}"}
    return {
        "open": SpecialTokens.tool_calls.value,
        "close": SpecialTokens.eos.value,
        "content": "json",
        "transform_each": True,
        "transform": transform,
    }


@pytest.fixture(scope="module")
def expected_v2() -> dict[str, Any]:
    return {
        "defaults": {"role": "assistant"},
        "start_anchor": _TOOL_ANCHORS,
        "fields": {
            "tool_calls": _pre_v11_tool_calls(with_id=False),
            "content": _COMMON_CONTENT_FIELD,
        },
    }


@pytest.fixture(scope="module")
def expected_v3() -> dict[str, Any]:
    return {
        "defaults": {"role": "assistant"},
        "start_anchor": _TOOL_ANCHORS,
        "fields": {
            "tool_calls": _pre_v11_tool_calls(with_id=True),
            "content": _COMMON_CONTENT_FIELD,
        },
    }


@pytest.fixture(scope="module")
def expected_v15() -> dict[str, Any]:
    return {
        "defaults": {"role": "assistant"},
        "start_anchor": _TOOL_ANCHORS,
        "fields": {
            "reasoning_content": {
                "open": SpecialTokens.begin_think.value,
                "close": SpecialTokens.end_think.value,
                "content": "text",
            },
            "tool_calls": {
                "open_pattern": re.escape(SpecialTokens.tool_calls.value)
                + "(?P<name>.+?)"
                + re.escape(SpecialTokens.args.value),
                "close_pattern": "(?="
                + re.escape(SpecialTokens.tool_calls.value)
                + ")|"
                + re.escape(SpecialTokens.eos.value),
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


@pytest.mark.parametrize(
    ("version", "thinking_mode", "fixture_name"),
    [
        (TokenizerVersion.v1, ThinkingMode.none, "expected_v1"),
        (TokenizerVersion.v2, ThinkingMode.none, "expected_v2"),
        (TokenizerVersion.v3, ThinkingMode.none, "expected_v3"),
        (TokenizerVersion.v7, ThinkingMode.none, "expected_v7"),
        (TokenizerVersion.v13, ThinkingMode.special, "expected_v13"),
        (TokenizerVersion.v15, ThinkingMode.special, "expected_v15"),
        (TokenizerVersion.v11, ThinkingMode.plain, "expected_v11_plain_thinking"),
        (TokenizerVersion.v11, ThinkingMode.none, "expected_v11_no_thinking"),
    ],
    ids=["v1", "v2", "v3", "v7", "v13", "v15", "v11-plain-thinking", "v11-no-thinking"],
)
def test_expected_templates(
    request: pytest.FixtureRequest,
    version: TokenizerVersion,
    thinking_mode: ThinkingMode,
    fixture_name: str,
) -> None:
    expected = request.getfixturevalue(fixture_name)
    template = generate_response_template(tokenizer_version=version, thinking_mode=thinking_mode)
    assert template == expected


@pytest.mark.parametrize(
    ("version", "thinking_mode"),
    [
        (TokenizerVersion.v1, ThinkingMode.none),
        (TokenizerVersion.v2, ThinkingMode.none),
        (TokenizerVersion.v3, ThinkingMode.none),
        (TokenizerVersion.v7, ThinkingMode.none),
        (TokenizerVersion.v11, ThinkingMode.none),
        (TokenizerVersion.v11, ThinkingMode.plain),
        (TokenizerVersion.v13, ThinkingMode.none),
        (TokenizerVersion.v13, ThinkingMode.special),
        (TokenizerVersion.v15, ThinkingMode.special),
    ],
    ids=lambda v: v.value if isinstance(v, (TokenizerVersion, ThinkingMode)) else None,
)
def test_top_level_keys_and_json_roundtrip(version: TokenizerVersion, thinking_mode: ThinkingMode) -> None:
    template = generate_response_template(tokenizer_version=version, thinking_mode=thinking_mode)

    assert json.loads(s=json.dumps(obj=template)) == template
    assert set(template) == {"defaults", "start_anchor", "fields"}
    assert template["defaults"] == {"role": "assistant"}


@pytest.mark.parametrize(
    ("version", "thinking_mode", "match"),
    [
        (TokenizerVersion.v7, ThinkingMode.plain, "not supported"),
        (TokenizerVersion.v7, ThinkingMode.special, "not supported"),
        (TokenizerVersion.v13, ThinkingMode.plain, "not supported"),
    ],
    ids=["plain-on-v7", "special-on-v7", "plain-on-v13"],
)
def test_invalid_thinking_modes_raise(
    version: TokenizerVersion,
    thinking_mode: ThinkingMode,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        generate_response_template(tokenizer_version=version, thinking_mode=thinking_mode)


def test_convert_tokenizer_to_response_template_tekken_v13(
    tmp_path: Path,
    expected_v13: dict[str, Any],
) -> None:
    path = build_tekken_json(config=TestConfig(version=TokenizerVersion.v13, think=True), output_dir=tmp_path)
    tokenizer = MistralTokenizer.from_file(tokenizer_filename=str(path), mode=ValidationMode.test)
    result = convert_tokenizer_to_response_template(tokenizer=tokenizer)
    assert result == expected_v13


def test_convert_v11_plain_thinking_heuristic(
    tmp_path: Path, expected_v11_plain_thinking: dict[str, Any], expected_v11_no_thinking: dict[str, Any]
) -> None:
    # Matches the chat-template converter: v11 without audio defaults to plain thinking on.
    path = build_tekken_json(config=TestConfig(version=TokenizerVersion.v11), output_dir=tmp_path)
    tokenizer = MistralTokenizer.from_file(tokenizer_filename=str(path), mode=ValidationMode.test)
    auto = convert_tokenizer_to_response_template(tokenizer=tokenizer)
    assert auto == expected_v11_plain_thinking
    forced_off = convert_tokenizer_to_response_template(tokenizer=tokenizer, plain_thinking_support=False)
    assert forced_off == expected_v11_no_thinking


def test_convert_plain_thinking_rejected_for_non_v11(tmp_path: Path) -> None:
    path = build_tekken_json(config=TestConfig(version=TokenizerVersion.v13, think=True), output_dir=tmp_path)
    tokenizer = MistralTokenizer.from_file(tokenizer_filename=str(path), mode=ValidationMode.test)
    with pytest.raises(ValueError, match="only available for tokenizer version v11"):
        convert_tokenizer_to_response_template(tokenizer=tokenizer, plain_thinking_support=True)


def _encode_assistant_generation(
    version: TokenizerVersion,
    think: bool,
    tmp_path: Path,
    assistant: AssistantMessage,
) -> tuple[MistralTokenizer, list[int]]:
    r"""Encode [user, assistant] and return (tokenizer, generation tokens)."""
    path = build_tekken_json(config=TestConfig(version=version, think=think), output_dir=tmp_path)
    tokenizer = MistralTokenizer.from_file(tokenizer_filename=str(path), mode=ValidationMode.test)
    messages: list[ChatMessage] = [UserMessage(content="hi"), assistant]
    request = ChatCompletionRequest(messages=messages)
    tokens = tokenizer.encode_chat_completion(request=request).tokens
    inst_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.end_inst.value)
    last_inst = len(tokens) - 1 - tokens[::-1].index(inst_id)
    generation = tokens[last_inst + 1 :]
    return tokenizer, generation


def _sample_tool_calls() -> list[ToolCall]:
    return [
        ToolCall(
            id="abcdefghi", function=FunctionCall(name="get_weather", arguments=json.dumps(obj={"city": "Paris"}))
        ),
        ToolCall(id="abcdefghj", function=FunctionCall(name="get_time", arguments=json.dumps(obj={"tz": "CET"}))),
    ]


def test_v11_open_pattern_matches_call_with_and_without_id() -> None:
    # v11 emits [CALL_ID] only for non-null ids; both forms must match, with the
    # id group empty for the id-less form.
    template = generate_response_template(tokenizer_version=TokenizerVersion.v11, thinking_mode=ThinkingMode.none)
    open_pattern = template["fields"]["tool_calls"]["open_pattern"]

    with_id = re.fullmatch(
        pattern=open_pattern + ".*",
        string=f"{SpecialTokens.tool_calls.value}f{SpecialTokens.call_id.value}abc123def{SpecialTokens.args.value}{{}}",
    )
    without_id = re.fullmatch(
        pattern=open_pattern + ".*",
        string=f"{SpecialTokens.tool_calls.value}f{SpecialTokens.args.value}{{}}",
    )

    assert with_id is not None and with_id.group("name") == "f" and with_id.group("id") == "abc123def"
    assert without_id is not None and without_id.group("name") == "f" and without_id.group("id") == ""


@pytest.mark.parametrize(
    ("version", "think", "n_tool_calls", "with_thinking", "format_kind", "thinking_mode"),
    [
        (TokenizerVersion.v13, True, 2, True, "pattern-no-id", ThinkingMode.special),
        (TokenizerVersion.v7, False, 1, False, "json-array", ThinkingMode.none),
        (TokenizerVersion.v11, False, 2, False, "pattern-with-id", ThinkingMode.none),
    ],
    ids=["v13", "v7", "v11"],
)
def test_matches_experimental_parser(
    version: TokenizerVersion,
    think: bool,
    n_tool_calls: int,
    with_thinking: bool,
    format_kind: str,
    thinking_mode: ThinkingMode,
    tmp_path: Path,
) -> None:
    content: Any = [ThinkChunk(thinking="let me check"), TextChunk(text="Sure.")] if with_thinking else "Checking."
    assistant = AssistantMessage(content=content, tool_calls=_sample_tool_calls()[:n_tool_calls], prefix=True)
    tokenizer, generation = _encode_assistant_generation(
        version=version, think=think, tmp_path=tmp_path, assistant=assistant
    )
    template = generate_response_template(tokenizer_version=version, thinking_mode=thinking_mode)

    tool_call_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.tool_calls.value)
    content_tokens, tool_call_tokens = _split_content_and_tool_calls(tokens=generation, tool_call_token_id=tool_call_id)
    tool_calls = _decode_tool_calls(tool_call_tokens=tool_call_tokens, tokenizer=tokenizer.instruct_tokenizer.tokenizer)
    assert len(tool_calls) == n_tool_calls

    tool_calls_field = template["fields"]["tool_calls"]
    if format_kind == "json-array":
        assert "open_pattern" not in tool_calls_field
        assert tool_calls_field["open"] == SpecialTokens.tool_calls.value
        assert tool_calls_field["transform_each"] is True
        region = tokenizer.decode(tokens=tool_call_tokens[0], special_token_policy=SpecialTokenPolicy.KEEP)
        payload = json.loads(
            s=region[len(SpecialTokens.tool_calls.value) :].strip().removesuffix(SpecialTokens.eos.value)
        )
        assert isinstance(payload, list) and len(payload) == len(tool_calls)
        for decoded_element, tool_call in zip(payload, tool_calls):
            assert set(decoded_element) == {"id", "name", "arguments"}
            assert decoded_element["name"] == tool_call.function.name
            assert decoded_element["id"] == tool_call.id
            assert decoded_element["arguments"] == json.loads(s=tool_call.function.arguments)
    else:
        assert "open" not in tool_calls_field
        open_pattern = tool_calls_field["open_pattern"]
        assert (re.escape(SpecialTokens.call_id.value) in open_pattern) == (format_kind == "pattern-with-id")
        for tool_call_tokens_i, tool_call in zip(tool_call_tokens, tool_calls):
            region = tokenizer.decode(tokens=tool_call_tokens_i, special_token_policy=SpecialTokenPolicy.KEEP)
            match = re.fullmatch(pattern=open_pattern + ".*", string=region)
            assert match is not None
            assert match.group("name") == tool_call.function.name
            if format_kind == "pattern-with-id":
                assert match.group("id") == tool_call.id

    if with_thinking:
        think_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.begin_think.value)
        end_think_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.end_think.value)
        chunks = _split_content_and_think_chunks(
            tokens=content_tokens, begin_think_token_id=think_id, end_think_token_id=end_think_id
        )
        assert any(is_think for _, is_think in chunks)
        assert template["fields"]["reasoning_content"]["open"] == SpecialTokens.begin_think.value

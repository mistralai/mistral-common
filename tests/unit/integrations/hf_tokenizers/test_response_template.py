import json
import re
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


def _v11_tool_calls() -> dict[str, Any]:
    return {
        "open_pattern": re.escape(SpecialTokens.tool_calls.value)
        + "(?P<name>.+?)"
        + re.escape(SpecialTokens.call_id.value)
        + "(?P<id>.+?)"
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
        "start_anchor": [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value],
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
        "start_anchor": [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value],
        "fields": {
            "tool_calls": _v11_tool_calls(),
            "content": _COMMON_CONTENT_FIELD,
        },
    }


@pytest.mark.parametrize(
    ("version", "thinking_support", "plain_thinking_support", "fixture_name"),
    [
        (TokenizerVersion.v1, False, False, "expected_v1"),
        (TokenizerVersion.v7, False, False, "expected_v7"),
        (TokenizerVersion.v13, True, False, "expected_v13"),
        (TokenizerVersion.v11, False, True, "expected_v11_plain_thinking"),
        (TokenizerVersion.v11, False, False, "expected_v11_no_thinking"),
    ],
    ids=["v1", "v7", "v13", "v11-plain-thinking", "v11-no-thinking"],
)
def test_expected_templates(
    request: pytest.FixtureRequest,
    version: TokenizerVersion,
    thinking_support: bool,
    plain_thinking_support: bool,
    fixture_name: str,
) -> None:
    expected = request.getfixturevalue(fixture_name)
    template = generate_response_template(
        tokenizer_version=version,
        thinking_support=thinking_support,
        plain_thinking_support=plain_thinking_support,
    )
    assert template == expected


@pytest.mark.parametrize(
    ("version", "thinking_support", "plain_thinking_support"),
    [
        (TokenizerVersion.v1, False, False),
        (TokenizerVersion.v2, False, False),
        (TokenizerVersion.v3, False, False),
        (TokenizerVersion.v7, False, False),
        (TokenizerVersion.v11, False, False),
        (TokenizerVersion.v13, False, False),
        (TokenizerVersion.v15, False, False),
        (TokenizerVersion.v11, False, True),
        (TokenizerVersion.v13, True, False),
    ],
    ids=lambda v: v.value if isinstance(v, TokenizerVersion) else None,
)
def test_top_level_keys_and_json_roundtrip(
    version: TokenizerVersion, thinking_support: bool, plain_thinking_support: bool
) -> None:
    template = generate_response_template(
        tokenizer_version=version,
        thinking_support=thinking_support,
        plain_thinking_support=plain_thinking_support,
    )

    assert json.loads(s=json.dumps(obj=template)) == template
    assert set(template) == {"defaults", "start_anchor", "fields"}
    assert template["defaults"] == {"role": "assistant"}


@pytest.mark.parametrize(
    ("version", "expected_anchor"),
    [
        (TokenizerVersion.v1, SpecialTokens.end_inst.value),
        (TokenizerVersion.v2, [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value]),
        (TokenizerVersion.v3, [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value]),
        (TokenizerVersion.v7, [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value]),
        (TokenizerVersion.v11, [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value]),
        (TokenizerVersion.v13, [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value]),
        (TokenizerVersion.v15, [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value]),
    ],
    ids=lambda v: v.value if isinstance(v, TokenizerVersion) else None,
)
def test_start_anchor_per_version(version: TokenizerVersion, expected_anchor: Any) -> None:
    template = generate_response_template(
        tokenizer_version=version,
        thinking_support=False,
        plain_thinking_support=False,
    )
    assert template["start_anchor"] == expected_anchor


def _open_pattern(*middle: str) -> str:
    r"""Build a tool-call open_pattern from re.escape'd special tokens."""
    return re.escape(SpecialTokens.tool_calls.value) + "".join(middle) + re.escape(SpecialTokens.args.value)


_CLOSE_PATTERN = "(?=" + re.escape(SpecialTokens.tool_calls.value) + ")|" + re.escape(SpecialTokens.eos.value)


@pytest.mark.parametrize(
    ("version", "expected_tool_calls"),
    [
        (
            TokenizerVersion.v13,
            {
                "open_pattern": _open_pattern("(?P<name>.+?)"),
                "close_pattern": _CLOSE_PATTERN,
                "repeats": True,
                "content": "json",
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"},
                },
            },
        ),
        (
            TokenizerVersion.v15,
            {
                "open_pattern": _open_pattern("(?P<name>.+?)"),
                "close_pattern": _CLOSE_PATTERN,
                "repeats": True,
                "content": "json",
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"},
                },
            },
        ),
        (
            TokenizerVersion.v11,
            {
                "open_pattern": _open_pattern("(?P<name>.+?)", re.escape(SpecialTokens.call_id.value) + "(?P<id>.+?)"),
                "close_pattern": _CLOSE_PATTERN,
                "repeats": True,
                "content": "json",
                "transform": {
                    "type": "function",
                    "id": "{id}",
                    "function": {"name": "{name}", "arguments": "{content}"},
                },
            },
        ),
        (
            TokenizerVersion.v7,
            {
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
        ),
        (
            TokenizerVersion.v3,
            {
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
        ),
        (
            TokenizerVersion.v2,
            {
                "open": SpecialTokens.tool_calls.value,
                "close": SpecialTokens.eos.value,
                "content": "json",
                "transform_each": True,
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{arguments}"},
                },
            },
        ),
    ],
    ids=["v13", "v15", "v11", "v7", "v3", "v2"],
)
def test_tool_calls_field_per_version(version: TokenizerVersion, expected_tool_calls: dict[str, Any]) -> None:
    template = generate_response_template(
        tokenizer_version=version,
        thinking_support=version.uses_special_thinking,
        plain_thinking_support=False,
    )
    assert template["fields"]["tool_calls"] == expected_tool_calls


@pytest.mark.parametrize(
    ("version", "plain_thinking_support", "expected_reasoning"),
    [
        (TokenizerVersion.v13, False, {"open": "[THINK]", "close": "[/THINK]", "content": "text"}),
        (TokenizerVersion.v15, False, {"open": "[THINK]", "close": "[/THINK]", "content": "text"}),
        (TokenizerVersion.v11, True, {"open": PLAIN_BEGIN_THINK, "close": PLAIN_END_THINK, "content": "text"}),
        (TokenizerVersion.v11, False, None),
        (TokenizerVersion.v7, False, None),
        (TokenizerVersion.v3, False, None),
        (TokenizerVersion.v2, False, None),
    ],
    ids=["v13", "v15", "v11-plain", "v11-none", "v7", "v3", "v2"],
)
def test_reasoning_field_per_version(
    version: TokenizerVersion, plain_thinking_support: bool, expected_reasoning: dict[str, Any] | None
) -> None:
    template = generate_response_template(
        tokenizer_version=version,
        thinking_support=version.uses_special_thinking,
        plain_thinking_support=plain_thinking_support,
    )
    if expected_reasoning is None:
        assert "reasoning_content" not in template["fields"]
    else:
        assert template["fields"]["reasoning_content"] == expected_reasoning


@pytest.mark.parametrize(
    ("version", "expected_content"),
    [
        (TokenizerVersion.v1, {"close": SpecialTokens.eos.value, "content": "text"}),
        *[
            (
                version,
                {
                    "close": [SpecialTokens.tool_calls.value, SpecialTokens.eos.value],
                    "content": "text",
                    "repeats": True,
                    "join": "",
                },
            )
            for version in (
                TokenizerVersion.v2,
                TokenizerVersion.v3,
                TokenizerVersion.v7,
                TokenizerVersion.v11,
                TokenizerVersion.v13,
                TokenizerVersion.v15,
            )
        ],
    ],
    ids=lambda v: v.value if isinstance(v, TokenizerVersion) else None,
)
def test_content_field(version: TokenizerVersion, expected_content: dict[str, Any]) -> None:
    template = generate_response_template(
        tokenizer_version=version,
        thinking_support=False,
        plain_thinking_support=False,
    )
    content = template["fields"]["content"]
    assert "open" not in content and "open_pattern" not in content
    assert content == expected_content


@pytest.mark.parametrize(
    ("version", "thinking_support", "plain_thinking_support", "match"),
    [
        (TokenizerVersion.v13, True, True, "mutually exclusive"),
        (TokenizerVersion.v7, False, True, "only available for tokenizer version v11"),
        (TokenizerVersion.v7, True, False, "v13 or above"),
    ],
    ids=["both-thinking-modes", "plain-on-v7", "special-on-v7"],
)
def test_invalid_thinking_combos_raise(
    version: TokenizerVersion,
    thinking_support: bool,
    plain_thinking_support: bool,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        generate_response_template(
            tokenizer_version=version,
            thinking_support=thinking_support,
            plain_thinking_support=plain_thinking_support,
        )


_GUARDED_TOKENS = (
    SpecialTokens.tool_calls.value,
    SpecialTokens.args.value,
    SpecialTokens.call_id.value,
    SpecialTokens.begin_think.value,
    SpecialTokens.end_think.value,
    SpecialTokens.end_inst.value,
    SpecialTokens.end_tool_results.value,
    SpecialTokens.eos.value,
)


@pytest.mark.parametrize("token", _GUARDED_TOKENS, ids=[t.strip("[]<>/") for t in _GUARDED_TOKENS])
def test_no_delimiter_string_literals(token: str) -> None:
    source = Path(mistral_common.integrations.hf_tokenizers.response_template.__file__).read_text(encoding="utf-8")
    assert token not in source
    assert re.escape(token) not in source
    assert re.escape(re.escape(token)) not in source
    assert "re.escape" in source


@pytest.mark.parametrize("pass_tokenizer_object", [False, True], ids=["path", "tokenizer-object"])
def test_convert_tokenizer_to_response_template_tekken_v13(tmp_path: Path, pass_tokenizer_object: bool) -> None:
    path = build_tekken_json(config=TestConfig(version=TokenizerVersion.v13, think=True), output_dir=tmp_path)
    source: str | Path | MistralTokenizer = (
        MistralTokenizer.from_file(tokenizer_filename=str(path), mode=ValidationMode.test)
        if pass_tokenizer_object
        else path
    )
    result = convert_tokenizer_to_response_template(tokenizer=source, plain_thinking_support=None)
    assert result == generate_response_template(
        tokenizer_version=TokenizerVersion.v13,
        thinking_support=True,
        plain_thinking_support=False,
    )


def test_convert_v11_plain_thinking_heuristic(
    tmp_path: Path, expected_v11_plain_thinking: dict[str, Any], expected_v11_no_thinking: dict[str, Any]
) -> None:
    path = build_tekken_json(config=TestConfig(version=TokenizerVersion.v11), output_dir=tmp_path)
    auto = convert_tokenizer_to_response_template(tokenizer=path, plain_thinking_support=None)
    assert auto == expected_v11_plain_thinking
    forced_off = convert_tokenizer_to_response_template(tokenizer=path, plain_thinking_support=False)
    assert forced_off == expected_v11_no_thinking


def _encode_assistant_generation(
    version: TokenizerVersion,
    think: bool,
    tmp_path: Path,
    assistant: AssistantMessage,
) -> tuple[MistralTokenizer, list[int], str]:
    r"""Encode [user, assistant] and return (tokenizer, generation tokens, decoded KEEP text)."""
    path = build_tekken_json(config=TestConfig(version=version, think=think), output_dir=tmp_path)
    tokenizer = MistralTokenizer.from_file(tokenizer_filename=str(path), mode=ValidationMode.test)
    messages: list[ChatMessage] = [UserMessage(content="hi"), assistant]
    request = ChatCompletionRequest(messages=messages)
    tokens = tokenizer.encode_chat_completion(request=request).tokens
    inst_id = tokenizer.instruct_tokenizer.tokenizer.get_special_token(SpecialTokens.end_inst.value)
    last_inst = len(tokens) - 1 - tokens[::-1].index(inst_id)
    generation = tokens[last_inst + 1 :]
    decoded = tokenizer.decode(tokens=generation, special_token_policy=SpecialTokenPolicy.KEEP)
    return tokenizer, generation, decoded


def _sample_tool_calls() -> list[ToolCall]:
    return [
        ToolCall(
            id="abcdefghi", function=FunctionCall(name="get_weather", arguments=json.dumps(obj={"city": "Paris"}))
        ),
        ToolCall(id="abcdefghj", function=FunctionCall(name="get_time", arguments=json.dumps(obj={"tz": "CET"}))),
    ]


@pytest.mark.parametrize(
    ("version", "think", "n_tool_calls", "with_thinking", "format_kind"),
    [
        (TokenizerVersion.v13, True, 2, True, "pattern-no-id"),
        (TokenizerVersion.v7, False, 1, False, "json-array"),
        (TokenizerVersion.v11, False, 2, False, "pattern-with-id"),
    ],
    ids=["v13", "v7", "v11"],
)
def test_matches_experimental_parser(
    version: TokenizerVersion,
    think: bool,
    n_tool_calls: int,
    with_thinking: bool,
    format_kind: str,
    tmp_path: Path,
) -> None:
    content: Any = [ThinkChunk(thinking="let me check"), TextChunk(text="Sure.")] if with_thinking else "Checking."
    assistant = AssistantMessage(content=content, tool_calls=_sample_tool_calls()[:n_tool_calls], prefix=True)
    tokenizer, generation, _ = _encode_assistant_generation(
        version=version, think=think, tmp_path=tmp_path, assistant=assistant
    )
    template = generate_response_template(
        tokenizer_version=version,
        thinking_support=version.uses_special_thinking,
        plain_thinking_support=False,
    )

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
    assert template["fields"]["content"]["close"][0] == SpecialTokens.tool_calls.value

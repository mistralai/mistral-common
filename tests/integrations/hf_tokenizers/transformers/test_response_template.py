from pathlib import Path
from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.integrations.hf_tokenizers.response_template import ThinkingMode, generate_response_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import TestConfig, build_tekken_json, render_template
from tests.integrations.chat_templates.hf_utils import build_hf_tokenizer


def _build(
    version: TokenizerVersion, thinking_mode: ThinkingMode, tmp_path: Path, with_system: bool = False
) -> tuple[str, Any, list[dict[str, Any]]]:
    tokenizer_config = TestConfig(
        version=version,
        think=thinking_mode is ThinkingMode.special,
        plain_think=thinking_mode is ThinkingMode.plain,
    )
    tokenizer_path = build_tekken_json(config=tokenizer_config, output_dir=tmp_path)
    chat_template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=False,
        audio_support=False,
        thinking_support=thinking_mode is ThinkingMode.special,
        default_system_prompt=None,
        plain_thinking_support=thinking_mode is ThinkingMode.plain,
        use_special_token_variables=True,
        model_settings_fields=None,
        reasoning_effort_values=None,
        reasoning_effort_default=None,
    )
    hf_tokenizer = build_hf_tokenizer(tekken_path=tokenizer_path, chat_template=chat_template)
    hf_tokenizer.response_template = generate_response_template(
        tokenizer_version=version,
        thinking_mode=thinking_mode,
    )
    messages: list[dict[str, Any]] = []
    if with_system:
        messages.append({"role": "system", "content": "be brief"})
    messages.append({"role": "user", "content": "hi"})
    return chat_template, hf_tokenizer, messages


def _assistant_message(
    with_thinking: bool, with_tool_calls: bool = True, with_id: bool = True, with_content: bool = True
) -> dict[str, Any]:
    if with_thinking:
        assistant_content: str | list[dict[str, str]] = [
            {"type": "thinking", "thinking": "let me check"},
            {"type": "text", "text": "Sure."},
        ]
    elif with_content:
        assistant_content = "Sure."
    else:
        assistant_content = None
    message: dict[str, Any] = {"role": "assistant", "content": assistant_content}
    if with_tool_calls:
        tool_call: dict[str, Any] = {
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
        }
        if with_id:
            tool_call["id"] = "abcdefghi"
        message["tool_calls"] = [tool_call]
    return message


@pytest.mark.parametrize(
    ("version", "thinking_mode", "with_system", "with_tool_calls"),
    [
        (TokenizerVersion.v13, ThinkingMode.special, False, True),
        (TokenizerVersion.v13, ThinkingMode.special, True, True),
        (TokenizerVersion.v11, ThinkingMode.plain, True, True),
        (TokenizerVersion.v7, ThinkingMode.none, True, True),
        (TokenizerVersion.v2, ThinkingMode.none, True, True),
        (TokenizerVersion.v1, ThinkingMode.none, True, False),
    ],
    ids=["v13-think", "v13-think-system", "v11-plain-system", "v7-system", "v2-system", "v1-system"],
)
def test_parse_response_roundtrip(
    version: TokenizerVersion,
    thinking_mode: ThinkingMode,
    with_system: bool,
    with_tool_calls: bool,
    tmp_path: Path,
) -> None:
    chat_template, hf_tokenizer, messages = _build(
        version=version, thinking_mode=thinking_mode, tmp_path=tmp_path, with_system=with_system
    )
    with_thinking = thinking_mode is not ThinkingMode.none
    # v2 forbids content together with tool calls; v1 has no tool support at all.
    assistant_message = _assistant_message(
        with_thinking=with_thinking,
        with_tool_calls=with_tool_calls,
        with_content=not (with_tool_calls and version == TokenizerVersion.v2),
    )

    prompt = render_template(template=chat_template, messages=messages)
    full_response = render_template(template=chat_template, messages=[*messages, assistant_message])
    assert full_response.startswith(prompt)
    generation = full_response[len(prompt) :]

    parsed_message = hf_tokenizer.parse_response(response=generation, prefix=prompt)

    # The system prompt and earlier turns must not leak into the parsed content.
    if version != TokenizerVersion.v2:
        assert parsed_message["content"] == "Sure."
    if with_tool_calls:
        assert parsed_message["tool_calls"][0]["function"]["name"] == "get_weather"
        assert parsed_message["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
        if version >= TokenizerVersion.v3 and version != TokenizerVersion.v13:
            assert parsed_message["tool_calls"][0]["id"] == "abcdefghi"
        else:
            assert "id" not in parsed_message["tool_calls"][0]
    else:
        assert "tool_calls" not in parsed_message
        assert parsed_message["content"] == "Sure."
    if with_thinking:
        assert parsed_message["reasoning_content"] == "let me check"


def test_parse_response_v11_without_call_id(tmp_path: Path) -> None:
    # v11 emits [CALL_ID] only for non-null ids; the pattern must accept both forms.
    # Build the generation by hand: the v11 chat template refuses to render an
    # assistant tool call without an id.
    chat_template, hf_tokenizer, messages = _build(
        version=TokenizerVersion.v11, thinking_mode=ThinkingMode.none, tmp_path=tmp_path
    )
    prompt = render_template(template=chat_template, messages=messages)
    generation = 'Sure.[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}</s>'

    parsed_message = hf_tokenizer.parse_response(response=generation, prefix=prompt)

    assert parsed_message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert parsed_message["tool_calls"][0]["id"] == ""
    assert parsed_message["content"] == "Sure."


def test_parse_response_system_then_assistant(tmp_path: Path) -> None:
    # A system message directly before the final assistant turn: only the
    # [/SYSTEM_PROMPT] start_anchor keeps earlier content out of the parse.
    # The v13 chat template rejects a system message after an assistant turn, so
    # the prefix is built by hand.
    _, hf_tokenizer, _ = _build(version=TokenizerVersion.v13, thinking_mode=ThinkingMode.special, tmp_path=tmp_path)
    prefix = (
        "<s>[SYSTEM_PROMPT]old sys[/SYSTEM_PROMPT][INST]hi[/INST]old answer</s>[SYSTEM_PROMPT]new sys[/SYSTEM_PROMPT]"
    )
    generation = 'Hello[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}</s>'

    parsed_message = hf_tokenizer.parse_response(response=generation, prefix=prefix)

    assert parsed_message["content"] == "Hello"
    assert parsed_message["tool_calls"][0]["function"]["name"] == "get_weather"

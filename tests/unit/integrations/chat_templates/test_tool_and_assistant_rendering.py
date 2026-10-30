r"""Regression tests for chat template rendering bugs.

- mistralai/mistral-common#311: tool-call arguments must not be HTML-escaped when the
  template is rendered with autoescape enabled (e.g. transformers).
- mistralai/mistral-common#399: list tool message content must be flattened like the
  tokenizer does (text chunks joined with two newlines), not printed as a Python repr.
- mistralai/mistral-common#424: trailing spaces in assistant string content must be
  stripped like the tokenizer does, keeping the 'CONTINUE_FINAL_MESSAGE_TAG ' suffix.
"""

from typing import Any

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion

_PRE_V15_VERSIONS = [
    TokenizerVersion.v2,
    TokenizerVersion.v3,
    TokenizerVersion.v7,
    TokenizerVersion.v11,
    TokenizerVersion.v13,
]
_ALL_VERSIONS = [*_PRE_V15_VERSIONS, TokenizerVersion.v15]


def _generate(version: TokenizerVersion) -> str:
    return generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=False,
        audio_support=False,
        thinking_support=False,
        default_system_prompt=None,
        plain_thinking_support=False,
        use_special_token_variables=True,
        model_settings_fields=None,
        reasoning_effort_values=("none", "high") if version >= TokenizerVersion.v15 else None,
        reasoning_effort_default="none" if version >= TokenizerVersion.v15 else None,
    )


def _render(template: str, messages: list[Any], *, autoescape: bool = True) -> str:
    def raise_exception(msg: str) -> None:
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True, autoescape=autoescape
    )
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, bos_token="<s>", eos_token="</s>")


def _tool_call_messages(arguments: Any) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "123456789", "function": {"name": "get_weather", "arguments": arguments}}],
        },
    ]


@pytest.mark.parametrize("version", _PRE_V15_VERSIONS)
@pytest.mark.parametrize("arguments", ({"city": "Paris"}, '{"city": "Paris"}'))
def test_tool_call_arguments_not_html_escaped(version: TokenizerVersion, arguments: Any) -> None:
    output = _render(_generate(version), _tool_call_messages(arguments))

    assert "&#34;" not in output
    assert '{"city": "Paris"}' in output


@pytest.mark.parametrize(
    ("version", "expected_snippet"),
    [
        # v2/v3 serialize the tool result as JSON: newlines appear escaped inside the JSON string.
        (TokenizerVersion.v2, "Sunny\\n\\n22C"),
        (TokenizerVersion.v3, "Sunny\\n\\n22C"),
        (TokenizerVersion.v7, "Sunny\n\n22C"),
        (TokenizerVersion.v11, "Sunny\n\n22C"),
        (TokenizerVersion.v13, "Sunny\n\n22C"),
    ],
)
def test_list_tool_message_content_flattened_not_repr(version: TokenizerVersion, expected_snippet: str) -> None:
    if version == TokenizerVersion.v2:
        tool_message: dict[str, Any] = {
            "role": "tool",
            "name": "get_weather",
            "content": [{"type": "text", "text": "Sunny"}, {"type": "text", "text": "22C"}],
        }
    else:
        tool_message = {
            "role": "tool",
            "content": [{"type": "text", "text": "Sunny"}, {"type": "text", "text": "22C"}],
            "tool_call_id": "123456789",
        }
    messages = [*_tool_call_messages({"city": "Paris"}), tool_message]

    output = _render(_generate(version), messages)

    assert "[{'type'" not in output
    assert expected_snippet in output


@pytest.mark.parametrize("version", _ALL_VERSIONS)
def test_assistant_content_trailing_spaces_stripped(version: TokenizerVersion) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello   "},
    ]
    output = _render(_generate(version), messages, autoescape=False)

    assert "Hello</s>" in output


@pytest.mark.parametrize("version", _ALL_VERSIONS)
def test_assistant_content_keeps_trailing_newline(version: TokenizerVersion) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "```python\n"},
    ]
    output = _render(_generate(version), messages, autoescape=False)

    assert "```python\n</s>" in output


@pytest.mark.parametrize("version", _ALL_VERSIONS)
def test_assistant_content_preserves_continue_final_message_tag_space(version: TokenizerVersion) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "```python\n  CONTINUE_FINAL_MESSAGE_TAG "},
    ]
    output = _render(_generate(version), messages, autoescape=False)

    assert "```python\nCONTINUE_FINAL_MESSAGE_TAG </s>" in output

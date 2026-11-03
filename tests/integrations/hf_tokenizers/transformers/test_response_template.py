from pathlib import Path
from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.integrations.hf_tokenizers.response_template import generate_response_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import TestConfig, build_tekken_json, render_template
from tests.integrations.chat_templates.hf_utils import build_hf_tokenizer


@pytest.mark.parametrize(
    ("version", "thinking_support", "plain_thinking_support"),
    [
        (TokenizerVersion.v13, True, False),
        (TokenizerVersion.v11, False, True),
        (TokenizerVersion.v7, False, False),
    ],
    ids=["v13-think", "v11-plain", "v7"],
)
def test_parse_response_roundtrip(
    version: TokenizerVersion,
    thinking_support: bool,
    plain_thinking_support: bool,
    tmp_path: Path,
) -> None:
    tokenizer_config = TestConfig(
        version=version,
        think=thinking_support,
        plain_think=plain_thinking_support,
    )
    tokenizer_path = build_tekken_json(config=tokenizer_config, output_dir=tmp_path)
    chat_template = generate_chat_template(
        spm=False,
        tokenizer_version=version,
        image_support=False,
        audio_support=False,
        thinking_support=thinking_support,
        default_system_prompt=None,
        plain_thinking_support=plain_thinking_support,
        use_special_token_variables=True,
        model_settings_fields=None,
        reasoning_effort_values=None,
        reasoning_effort_default=None,
    )
    hf_tokenizer = build_hf_tokenizer(tekken_path=tokenizer_path, chat_template=chat_template)
    hf_tokenizer.response_template = generate_response_template(
        tokenizer_version=version,
        thinking_support=thinking_support,
        plain_thinking_support=plain_thinking_support,
    )

    user_message: dict[str, Any] = {"role": "user", "content": "hi"}
    if thinking_support or plain_thinking_support:
        assistant_content: str | list[dict[str, str]] = [
            {"type": "thinking", "thinking": "let me check"},
            {"type": "text", "text": "Sure."},
        ]
    else:
        assistant_content = "Sure."
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content,
        "tool_calls": [
            {
                "id": "abcdefghi",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ],
    }
    prompt = render_template(template=chat_template, messages=[user_message])
    full_response = render_template(template=chat_template, messages=[user_message, assistant_message])
    assert full_response.startswith(prompt)
    generation = full_response[len(prompt) :]

    parsed_message = hf_tokenizer.parse_response(response=generation, prefix=prompt)

    assert parsed_message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert parsed_message["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert parsed_message["content"] == "Sure."
    if thinking_support or plain_thinking_support:
        assert parsed_message["reasoning_content"] == "let me check"

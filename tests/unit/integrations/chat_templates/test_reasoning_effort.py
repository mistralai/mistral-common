from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.integrations.chat_templates.template_generator import TemplateConfig, build_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import render_template


def test_template_config_rejects_invalid_reasoning_effort_options() -> None:
    invalid_options: list[tuple[dict[str, Any], str]] = [
        (
            {
                "reasoning_effort_values": (),
                "reasoning_effort_default": "none",
            },
            "non-empty tuple",
        ),
        ({"reasoning_effort_values": ("low", "low")}, "duplicates"),
        ({"reasoning_effort_values": ("unknown",)}, "Unknown reasoning effort values"),
        (
            {"model_settings_fields": frozenset({"json_schema"}), "reasoning_effort_values": ("low",)},
            "require the reasoning_effort model settings field",
        ),
        (
            {
                "model_settings_fields": frozenset({"reasoning_effort"}),
                "reasoning_effort_values": ("none", "high"),
            },
            "default is required",
        ),
        (
            {"model_settings_fields": frozenset({"json_schema"}), "reasoning_effort_default": "low"},
            "default requires the reasoning_effort model settings field",
        ),
        (
            {"reasoning_effort_values": ("low",), "reasoning_effort_default": "unknown"},
            "one of the exposed values",
        ),
    ]

    for options, error in invalid_options:
        with pytest.raises(ValueError, match=error):
            TemplateConfig(version=TokenizerVersion.v15, **options)


def test_template_config_requires_effort_values_when_field_is_enabled() -> None:
    with pytest.raises(ValueError, match="Reasoning effort values are required"):
        TemplateConfig(version=TokenizerVersion.v15, reasoning_effort_default="none")


def test_template_config_requires_effort_default_when_field_is_enabled() -> None:
    with pytest.raises(ValueError, match="default is required"):
        TemplateConfig(version=TokenizerVersion.v15, reasoning_effort_values=("none", "high"))


def test_template_config_requires_default_in_exposed_values() -> None:
    with pytest.raises(ValueError, match="one of the exposed values"):
        TemplateConfig(
            version=TokenizerVersion.v15,
            reasoning_effort_values=("low", "high"),
            reasoning_effort_default="medium",
        )


def test_generated_template_uses_configured_default_and_rejects_unexposed_value() -> None:
    template = build_chat_template(
        TemplateConfig(
            version=TokenizerVersion.v15,
            reasoning_effort_values=("low", "high"),
            reasoning_effort_default="high",
        )
    )
    messages = [{"role": "user", "content": "hello"}]

    assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in render_template(
        template=template, messages=messages
    )
    with pytest.raises(ValueError, match='reasoning_effort must be either "low" or "high"'):
        render_template(template=template, messages=messages, reasoning_effort="medium")


def test_generate_chat_template_accepts_effort_values_and_default() -> None:
    template = generate_chat_template(
        spm=False,
        tokenizer_version=TokenizerVersion.v15,
        image_support=False,
        audio_support=False,
        thinking_support=False,
        default_system_prompt=None,
        plain_thinking_support=False,
        use_special_token_variables=True,
        model_settings_fields=None,
        reasoning_effort_values=("minimal", "max"),
        reasoning_effort_default="minimal",
    )

    assert "'minimal', 'max'" in template

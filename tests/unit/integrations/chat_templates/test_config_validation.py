from typing import Any

import pytest

from mistral_common.integrations.chat_templates.template_generator import TemplateConfig, build_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.utils.chat_templates.configs import ALL_CONFIGS, format_config_id
from tests.utils.chat_templates.helpers import TestConfig, build_template_config


class TestInvalidConfigs:
    @pytest.mark.parametrize(
        ("kwargs", "error_match"),
        [
            pytest.param(
                {"version": TokenizerVersion.v11, "spm": True},
                "SPM tokenizer is not supported",
                id="v11_spm",
            ),
            pytest.param(
                {"version": TokenizerVersion.v7, "image_support": True, "audio_support": True},
                "Image and audio support are mutually exclusive",
                id="v7_image_audio",
            ),
            pytest.param(
                {"version": TokenizerVersion.v1, "image_support": True},
                "Image support is only available",
                id="v1_image",
            ),
            pytest.param(
                {"version": TokenizerVersion.v3, "audio_support": True},
                "Audio support is only available",
                id="v3_audio",
            ),
            pytest.param(
                {"version": TokenizerVersion.v7, "thinking_support": True},
                "Thinking support is only available",
                id="v7_thinking",
            ),
            pytest.param(
                {"version": TokenizerVersion.v13, "audio_support": True, "thinking_support": True},
                "Audio and thinking support are mutually exclusive",
                id="v13_audio_thinking",
            ),
            pytest.param(
                {"version": TokenizerVersion.v11, "thinking_support": True, "plain_thinking_support": True},
                "Plain thinking support and thinking support are mutually exclusive",
                id="v11_think_plain_think",
            ),
            pytest.param(
                {
                    "version": TokenizerVersion.v15,
                    "plain_thinking_support": True,
                    "reasoning_effort_values": ("none", "high"),
                    "reasoning_effort_default": "none",
                },
                "Plain thinking support is only available for tokenizer version v11",
                id="v15_plain_think",
            ),
            pytest.param(
                {"version": TokenizerVersion.v11, "audio_support": True, "plain_thinking_support": True},
                "Audio and plain thinking support are mutually exclusive",
                id="v11_audio_plain_think",
            ),
        ],
    )
    def test_invalid_config(self, kwargs: dict[str, Any], error_match: str) -> None:
        with pytest.raises(ValueError, match=error_match):
            TemplateConfig(**kwargs)


class TestValidConfigs:
    @pytest.mark.parametrize("config", ALL_CONFIGS, ids=[format_config_id(c) for c in ALL_CONFIGS])
    def test_all_configs_are_valid(self, config: TestConfig) -> None:
        template_config = build_template_config(config=config)
        assert template_config.any_thinking_support == (config.think or config.plain_think)

    @pytest.mark.parametrize("config", ALL_CONFIGS, ids=[format_config_id(c) for c in ALL_CONFIGS])
    def test_tools_support_matches_version_policy(self, config: TestConfig) -> None:
        template = build_chat_template(build_template_config(config=config))
        has_tools_branch = "{%- set has_tools = true %}" in template
        assert has_tools_branch == (config.version >= TokenizerVersion.v2)

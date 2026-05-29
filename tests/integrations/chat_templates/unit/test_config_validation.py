from typing import Any

import pytest

from mistral_common.integrations.chat_templates.template_generator import TemplateConfig
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.conftest import ALL_CONFIGS
from tests.integrations.chat_templates.helpers import _make_config


class TestInvalidConfigs:
    @pytest.mark.parametrize(
        ("kwargs", "error_match"),
        [
            ({"version": TokenizerVersion.v11, "spm": True}, "SPM tokenizer is not supported"),
            (
                {"version": TokenizerVersion.v7, "image_support": True, "audio_support": True},
                "Image and audio support are mutually exclusive",
            ),
            ({"version": TokenizerVersion.v1, "image_support": True}, "Image support is only available"),
            ({"version": TokenizerVersion.v3, "audio_support": True}, "Audio support is only available"),
            ({"version": TokenizerVersion.v7, "thinking_support": True}, "Thinking support is only available"),
            (
                {"version": TokenizerVersion.v13, "audio_support": True, "thinking_support": True},
                "Audio and thinking support are mutually exclusive",
            ),
            (
                {"version": TokenizerVersion.v11, "thinking_support": True, "plain_thinking_support": True},
                "Plain thinking support and thinking support are mutually exclusive",
            ),
            (
                {"version": TokenizerVersion.v15, "plain_thinking_support": True},
                "Plain thinking support is only available for tokenizer version v11",
            ),
            (
                {"version": TokenizerVersion.v11, "audio_support": True, "plain_thinking_support": True},
                "Audio and plain thinking support are mutually exclusive",
            ),
        ],
    )
    def test_invalid_config(self, kwargs: dict[str, Any], error_match: str) -> None:
        with pytest.raises(ValueError, match=error_match):
            TemplateConfig(**kwargs)


class TestValidConfigs:
    @pytest.mark.parametrize("config_tuple", ALL_CONFIGS)
    def test_valid_config_construction(
        self, config_tuple: tuple[TokenizerVersion, bool, bool, bool, bool, bool]
    ) -> None:
        config = _make_config(config_tuple)
        assert isinstance(config.has_tools, bool)
        assert isinstance(config.any_thinking_support, bool)

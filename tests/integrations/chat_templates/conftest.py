r"""Chat template test configuration.

Provides package-level fixtures, parametrize config lists, and invalid-message
fixtures used across all chat template test layers.

Reusable helper functions live in `helpers.py`.
Test data constants (`REQUEST_*`) live in `fixtures_data.py`.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest

from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import _IMAGE, TestConfig

# All configurations for output comparison tests (including SPM).
ALL_CONFIGS: list[TestConfig] = [
    # Non-SPM
    TestConfig(version=TokenizerVersion.v1),
    TestConfig(version=TokenizerVersion.v2),
    TestConfig(version=TokenizerVersion.v3),
    TestConfig(version=TokenizerVersion.v3, image=True),
    TestConfig(version=TokenizerVersion.v7),
    TestConfig(version=TokenizerVersion.v7, image=True),
    TestConfig(version=TokenizerVersion.v7, audio=True),
    TestConfig(version=TokenizerVersion.v11),
    TestConfig(version=TokenizerVersion.v11, image=True),
    TestConfig(version=TokenizerVersion.v11, audio=True),
    TestConfig(version=TokenizerVersion.v13),
    TestConfig(version=TokenizerVersion.v13, image=True),
    TestConfig(version=TokenizerVersion.v13, audio=True),
    TestConfig(version=TokenizerVersion.v13, think=True),
    TestConfig(version=TokenizerVersion.v13, image=True, think=True),
    TestConfig(version=TokenizerVersion.v15),
    TestConfig(version=TokenizerVersion.v15, image=True),
    TestConfig(version=TokenizerVersion.v15, audio=True),
    TestConfig(version=TokenizerVersion.v15, think=True),
    TestConfig(version=TokenizerVersion.v15, image=True, think=True),
    # SPM
    TestConfig(version=TokenizerVersion.v1, spm=True),
    TestConfig(version=TokenizerVersion.v2, spm=True),
    TestConfig(version=TokenizerVersion.v3, spm=True),
    TestConfig(version=TokenizerVersion.v3, spm=True, image=True),
    TestConfig(version=TokenizerVersion.v7, spm=True),
    TestConfig(version=TokenizerVersion.v7, spm=True, image=True),
    # Plain thinking (v11 only)
    TestConfig(version=TokenizerVersion.v11, plain_think=True),
    TestConfig(version=TokenizerVersion.v11, image=True, plain_think=True),
]

# Parametrization configs for transformers tests (excludes plain_think).
ALL_TRANSFORMERS_CONFIGS: list[TestConfig] = [c for c in ALL_CONFIGS if not c.plain_think]


@pytest.fixture(scope="module")
def invalid_sp_think() -> dict[str, Any]:
    r"""Invalid system message containing a think chunk."""
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "think", "thinking": "Hello"},
                ],
            }
        ]
    }


@pytest.fixture(scope="module")
def invalid_sp_random() -> dict[str, Any]:
    r"""Invalid system message containing an unknown chunk type."""
    return {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "random", "random": "Hello"},
                ],
            }
        ]
    }


@pytest.fixture(scope="module")
def invalid_assistant_think() -> dict[str, Any]:
    r"""Invalid assistant message containing a think chunk."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "think", "thinking": "Hello"},
                ],
            },
        ]
    }


@pytest.fixture(scope="module")
def invalid_assistant_random() -> dict[str, Any]:
    r"""Invalid assistant message containing an unknown chunk type."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "random", "random": "Hello"},
                ],
            },
        ]
    }


@pytest.fixture(scope="module")
def invalid_user_image() -> dict[str, Any]:
    r"""Invalid user message containing an image chunk."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "image_url": "Hello"},
                ],
            }
        ]
    }


@pytest.fixture(scope="module")
def invalid_user_audio() -> dict[str, Any]:
    r"""Invalid user message containing an audio chunk."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "audio", "audio_url": "Hello"},
                ],
            }
        ]
    }


@pytest.fixture(scope="module")
def invalid_user_random() -> dict[str, Any]:
    r"""Invalid user message containing an unknown chunk type."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "random", "random": "Hello"},
                ],
            }
        ]
    }


def _config_id(c: TestConfig) -> str:
    r"""Generate a human-readable test ID for a config."""
    parts = [c.version.value]
    if c.spm:
        parts.append("spm")
    if c.image:
        parts.append("img")
    if c.audio:
        parts.append("aud")
    if c.think:
        parts.append("think")
    if c.plain_think:
        parts.append("plain_think")
    return "_".join(parts)


@pytest.fixture(autouse=True, scope="package")
def mock_download_image() -> Generator[None, None, None]:
    r"""Mock `download_image` to return a dummy image for all tests."""
    with patch("mistral_common.image.download_image") as mock_download:
        mock_download.return_value = _IMAGE
        with patch("mistral_common.tokens.tokenizers.image.download_image") as mock_download2:
            mock_download2.return_value = _IMAGE
            yield

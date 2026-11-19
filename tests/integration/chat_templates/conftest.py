r"""Chat-template parity test fixtures.

Provides the invalid-message fixtures and the package-scoped image-download
patch used by the parity tests under this subtree. Configuration values come
from the neutral `tests/utils/chat_templates` package.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest

from tests.utils.chat_templates.helpers import DUMMY_IMAGE


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


@pytest.fixture(autouse=True, scope="package")
def mock_download_image() -> Generator[None, None, None]:
    r"""Mock `download_image` to return a dummy image for all tests."""
    with patch("mistral_common.image.download_image") as mock_download:
        mock_download.return_value = DUMMY_IMAGE
        with patch("mistral_common.tokens.tokenizers.image.download_image") as mock_download2:
            mock_download2.return_value = DUMMY_IMAGE
            yield

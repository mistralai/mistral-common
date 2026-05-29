r"""Chat template test configuration.

Provides package-level fixtures, parametrize config lists, and module-level
constants used across all chat template test layers.

Reusable helper functions live in `helpers.py`.
Test data constants (`REQUEST_*`) live in `fixtures_data.py`.
"""

from collections.abc import Generator
from unittest.mock import patch

import pytest

from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import _IMAGE

# All configurations for output comparison tests (including SPM).
# Tuples: (version, spm, image, audio, think, plain_think)
ALL_CONFIGS: list[tuple[TokenizerVersion, bool, bool, bool, bool, bool]] = [
    # Non-SPM
    (TokenizerVersion.v1, False, False, False, False, False),
    (TokenizerVersion.v2, False, False, False, False, False),
    (TokenizerVersion.v3, False, False, False, False, False),
    (TokenizerVersion.v3, False, True, False, False, False),
    (TokenizerVersion.v7, False, False, False, False, False),
    (TokenizerVersion.v7, False, True, False, False, False),
    (TokenizerVersion.v7, False, False, True, False, False),
    (TokenizerVersion.v11, False, False, False, False, False),
    (TokenizerVersion.v11, False, True, False, False, False),
    (TokenizerVersion.v11, False, False, True, False, False),
    (TokenizerVersion.v13, False, False, False, False, False),
    (TokenizerVersion.v13, False, True, False, False, False),
    (TokenizerVersion.v13, False, False, True, False, False),
    (TokenizerVersion.v13, False, False, False, True, False),
    (TokenizerVersion.v13, False, True, False, True, False),
    (TokenizerVersion.v15, False, False, False, False, False),
    (TokenizerVersion.v15, False, True, False, False, False),
    (TokenizerVersion.v15, False, False, True, False, False),
    (TokenizerVersion.v15, False, False, False, True, False),
    (TokenizerVersion.v15, False, True, False, True, False),
    # SPM
    (TokenizerVersion.v1, True, False, False, False, False),
    (TokenizerVersion.v2, True, False, False, False, False),
    (TokenizerVersion.v3, True, False, False, False, False),
    (TokenizerVersion.v3, True, True, False, False, False),
    (TokenizerVersion.v7, True, False, False, False, False),
    (TokenizerVersion.v7, True, True, False, False, False),
    # Plain thinking (v11 only)
    (TokenizerVersion.v11, False, False, False, False, True),
    (TokenizerVersion.v11, False, True, False, False, True),
]

# Parametrization configs for transformers tests, formatted as (spm, version, image, audio, think)
ALL_TRANSFORMERS_CONFIGS: list[tuple[bool, TokenizerVersion, bool, bool, bool]] = [
    (spm, version, image, audio, think)
    for version, spm, image, audio, think, plain_think in ALL_CONFIGS
    if not plain_think  # plain_think has separate tests
]


def _config_id(c: tuple[TokenizerVersion, bool, bool, bool, bool, bool]) -> str:
    r"""Generate a human-readable test ID for a config tuple."""
    v, spm, img, aud, think, plain = c
    parts = [v.value]
    if spm:
        parts.append("spm")
    if img:
        parts.append("img")
    if aud:
        parts.append("aud")
    if think:
        parts.append("think")
    if plain:
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

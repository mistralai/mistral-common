r"""Shared version matrix utilities for the test suite.

Provides a single source of truth for `TokenizerVersion` groupings and
test parametrization helpers used across tokenizer and integration tests.
"""

from dataclasses import dataclass

from mistral_common.tokens.tokenizers.base import TokenizerVersion

ALL_VERSIONS: tuple[TokenizerVersion, ...] = tuple(TokenizerVersion)

SPM_VERSIONS: tuple[TokenizerVersion, ...] = (
    TokenizerVersion.v1,
    TokenizerVersion.v2,
    TokenizerVersion.v3,
    TokenizerVersion.v7,
)

# Tekken v1 and v2 were never released; no real artefact exists to validate against.
TEKKEN_VERSIONS: tuple[TokenizerVersion, ...] = tuple(v for v in TokenizerVersion if v >= TokenizerVersion.v3)

AUDIO_VERSIONS: tuple[TokenizerVersion, ...] = tuple(v for v in TokenizerVersion if v >= TokenizerVersion.v7)

THINK_VERSIONS: tuple[TokenizerVersion, ...] = tuple(v for v in TokenizerVersion if v >= TokenizerVersion.v13)

MODEL_SETTINGS_VERSIONS: tuple[TokenizerVersion, ...] = tuple(v for v in TokenizerVersion if v.supports_model_settings)


@dataclass(frozen=True)
class TestConfig:
    r"""Single source of truth for tokenizer test parametrization.

    Attributes:
        version: The tokenizer version under test.
        spm: Whether to use the SentencePiece backend.
        image: Whether image support is enabled.
        audio: Whether audio support is enabled.
        think: Whether think (special-token) mode is enabled.
        plain_think: Whether plain-text think mode is enabled.
    """

    __test__ = False

    version: TokenizerVersion
    spm: bool = False
    image: bool = False
    audio: bool = False
    think: bool = False
    plain_think: bool = False


def config_id(config: TestConfig) -> str:
    r"""Return a human-readable pytest parametrize ID for a `TestConfig`.

    Suffixes are appended in fixed order: `spm`, `img`, `aud`, `think`,
    `plain_think`. Only truthy flags contribute a suffix.

    Args:
        config: The test configuration to encode.

    Returns:
        A `_`-joined string starting with the version value followed by
        any enabled feature suffixes, e.g. `"v13_img_think"`.

    Examples:
        >>> from mistral_common.tokens.tokenizers.base import TokenizerVersion
        >>> config_id(TestConfig(version=TokenizerVersion.v13, image=True, think=True))
        'v13_img_think'
    """
    parts = [config.version.value]
    if config.spm:
        parts.append("spm")
    if config.image:
        parts.append("img")
    if config.audio:
        parts.append("aud")
    if config.think:
        parts.append("think")
    if config.plain_think:
        parts.append("plain_think")
    return "_".join(parts)


# Config matrices for the `tekkenizer` / `spm` fixtures. Each list is a valid subset
# tests can narrow to via `@pytest.mark.parametrize("tekkenizer", <subset>, indirect=True)`.
BASE_TEKKEN_CONFIGS: tuple[TestConfig, ...] = tuple(TestConfig(version=v) for v in TEKKEN_VERSIONS)

IMAGE_TEKKEN_CONFIGS: tuple[TestConfig, ...] = tuple(TestConfig(version=v, image=True) for v in TEKKEN_VERSIONS)

AUDIO_TEKKEN_CONFIGS: tuple[TestConfig, ...] = tuple(TestConfig(version=v, audio=True) for v in AUDIO_VERSIONS)

THINK_TEKKEN_CONFIGS: tuple[TestConfig, ...] = tuple(TestConfig(version=v, think=True) for v in THINK_VERSIONS)

# Full Tekken matrix (every version, plus each single modality where supported).
TEKKEN_CONFIGS: tuple[TestConfig, ...] = (
    BASE_TEKKEN_CONFIGS + IMAGE_TEKKEN_CONFIGS + AUDIO_TEKKEN_CONFIGS + THINK_TEKKEN_CONFIGS
)

# Full SentencePiece matrix (every shipped SPM version).
SPM_CONFIGS: tuple[TestConfig, ...] = tuple(TestConfig(version=v, spm=True) for v in SPM_VERSIONS)

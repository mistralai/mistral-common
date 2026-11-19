r"""Chat-template test configurations shared by the unit and integration lanes.

`TestConfig` describes one tokenizer capability profile; the `ALL_*` lists
enumerate the supported profiles. `GOLDEN_CONFIGS` are the profiles that have
a reviewed static template under `tests/data/chat_templates/`.
"""

from dataclasses import dataclass

from mistral_common.tokens.tokenizers.base import TokenizerVersion


@dataclass(frozen=True)
class TestConfig:
    r"""Test configuration for chat template parametrization."""

    __test__ = False

    version: TokenizerVersion
    spm: bool = False
    image: bool = False
    audio: bool = False
    think: bool = False
    plain_think: bool = False
    model_settings_fields: frozenset[str] | None = None


# Every supported configuration (including SPM), for output comparison tests.
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
    TestConfig(
        version=TokenizerVersion.v15,
        model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
    ),
    TestConfig(
        version=TokenizerVersion.v15,
        image=True,
        model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
    ),
    TestConfig(
        version=TokenizerVersion.v15,
        think=True,
        model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
    ),
    TestConfig(
        version=TokenizerVersion.v15,
        image=True,
        think=True,
        model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
    ),
]


def has_released_template(config: TestConfig) -> bool:
    r"""Whether a config matches a released chat template; only those keep a golden template."""
    if config.audio and config.version in (TokenizerVersion.v11, TokenizerVersion.v15):
        return False
    if config.version == TokenizerVersion.v15:
        return config.image and config.think
    return True


# Configurations with a golden template in tests/data/chat_templates.
GOLDEN_CONFIGS: list[TestConfig] = [c for c in ALL_CONFIGS if has_released_template(c)]

# Parametrization configs for transformers tests (excludes plain_think).
ALL_TRANSFORMERS_CONFIGS: list[TestConfig] = [c for c in ALL_CONFIGS if not c.plain_think]


def format_config_id(c: TestConfig) -> str:
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
    if c.model_settings_fields is not None:
        parts.extend(sorted(c.model_settings_fields))
    return "_".join(parts)

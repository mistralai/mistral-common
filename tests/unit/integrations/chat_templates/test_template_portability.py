import re

import pytest

from mistral_common.integrations.chat_templates.template_generator import TemplateConfig, build_chat_template
from mistral_common.protocol.instruct.request import ReasoningEffort
from mistral_common.tokens.tokenizers.base import TokenizerVersion

# Python-only Jinja constructs that non-Python Jinja runtimes (e.g. HF Transformers'
# minijinja) do not support: dict methods and Python boolean/None literals.
_PYTHON_ONLY_PATTERNS = [
    re.compile(r"\.get\("),
    re.compile(r"\.lower\("),
    re.compile(r"\.strip\("),
    re.compile(r"\.items\("),
    re.compile(r"\bTrue\b"),
    re.compile(r"\bFalse\b"),
    re.compile(r"\bNone\b"),
]

_REASONING_EFFORT_VALUES = tuple(str(effort) for effort in ReasoningEffort)


def _model_settings_kwargs(version: TokenizerVersion) -> dict[str, frozenset[str] | tuple[str, ...] | str]:
    r"""Model settings arguments required when `version` supports model settings.

    Args:
        version: The tokenizer version to check.

    Returns:
        Keyword arguments enabling both model settings fields for v15+, else an empty dict.
    """
    if version.supports_model_settings:
        return {
            "model_settings_fields": frozenset({"reasoning_effort", "json_schema"}),
            "reasoning_effort_values": _REASONING_EFFORT_VALUES,
            "reasoning_effort_default": "none",
        }
    return {}


def _configs_for_version(version: TokenizerVersion) -> list[TemplateConfig]:
    r"""Build every feature variant of `version` that changes the generated template.

    Args:
        version: The tokenizer version to build variants for.

    Returns:
        One `TemplateConfig` per supported feature combination for `version`.
    """
    settings = _model_settings_kwargs(version)

    def make(**kwargs: bool) -> TemplateConfig:
        return TemplateConfig(version=version, **settings, **kwargs)  # type: ignore[arg-type]

    configs = [make()]
    if version >= TokenizerVersion.v3:
        configs.append(make(image_support=True))
    if version >= TokenizerVersion.v7:
        configs.append(make(audio_support=True))
    if version >= TokenizerVersion.v13:
        configs.append(make(thinking_support=True))
    if version == TokenizerVersion.v11:
        configs.append(make(plain_thinking_support=True))
    if version < TokenizerVersion.v11:
        configs.append(make(spm=True))
        if version >= TokenizerVersion.v3:
            configs.append(make(spm=True, image_support=True))
    configs.append(make(use_special_token_variables=True))
    return configs


@pytest.mark.parametrize("version", list(TokenizerVersion))
def test_generated_template_avoids_python_only_jinja(version: TokenizerVersion) -> None:
    # Non-Python Jinja runtimes (minijinja in HF Transformers) reject dict methods and
    # Python literals; generated templates must use `x['k'] if 'k' in x else default`
    # and the `true`/`false`/`none` literals instead.
    for config in _configs_for_version(version):
        template = build_chat_template(config)
        for pattern in _PYTHON_ONLY_PATTERNS:
            assert pattern.search(template) is None, (
                f"{version.value} config {config} emits Python-only construct {pattern.pattern!r}"
            )


@pytest.mark.parametrize("version", [v for v in TokenizerVersion if v >= TokenizerVersion.v13])
def test_thinking_templates_use_portable_closed_and_reasoning_expressions(version: TokenizerVersion) -> None:
    config = TemplateConfig(version=version, thinking_support=True, **_model_settings_kwargs(version))  # type: ignore[arg-type]
    template = build_chat_template(config)

    assert "(block['closed'] if 'closed' in block else true)" in template
    assert (
        "msg['reasoning_content'] if 'reasoning_content' in msg else (msg['reasoning'] if 'reasoning' in msg else none)"
        in template
    )


def test_v15_template_uses_portable_response_format_expressions() -> None:
    config = TemplateConfig(version=TokenizerVersion.v15, **_model_settings_kwargs(TokenizerVersion.v15))  # type: ignore[arg-type]
    template = build_chat_template(config)

    assert "response_format['type'] if 'type' in response_format else 'text'" in template
    assert "response_format['json_schema'] if 'json_schema' in response_format else none" in template
    assert "('schema' not in response_json_schema or response_json_schema['schema'] is none)" in template


@pytest.mark.parametrize("version", [v for v in TokenizerVersion if v >= TokenizerVersion.v2])
def test_tool_definition_uses_portable_key_expressions(version: TokenizerVersion) -> None:
    config = TemplateConfig(version=version, **_model_settings_kwargs(version))  # type: ignore[arg-type]
    template = build_chat_template(config)

    assert "tool_function['description'] if 'description' in tool_function else none" in template
    assert "tool_function['parameters'] if 'parameters' in tool_function else none" in template
    assert "tool['type'] if 'type' in tool else 'function'" in template

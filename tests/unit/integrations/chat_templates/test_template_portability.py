import re
from typing import TYPE_CHECKING, Any

import pytest

from mistral_common.integrations.chat_templates.template_generator import TemplateConfig, build_chat_template
from mistral_common.protocol.instruct.request import ReasoningEffort
from mistral_common.tokens.tokenizers.base import TokenizerVersion

if TYPE_CHECKING:
    import minijinja
else:
    minijinja = pytest.importorskip("minijinja")

_REASONING_EFFORT_VALUES = tuple(str(effort) for effort in ReasoningEffort)

_RENDER_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "answer", "reasoning_content": "trace"},
    {"role": "user", "content": "again"},
    {"role": "assistant", "content": "done"},
]
_RENDER_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "f", "description": None, "parameters": {"type": "object"}}},
    {"function": {"name": "g"}},
]


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


def _rewrite_generation_blocks(template: str) -> str:
    r"""Rewrite HF Transformers' `{% generation %}` blocks as scoped `{% with %}` blocks.

    minijinja has no `generation` tag. Like vLLM's Rust frontend, only the keywords are
    replaced, so the delimiters and their whitespace control are kept.

    Args:
        template: The chat template source.

    Returns:
        The template with each `generation`/`endgeneration` tag replaced by `with`/`endwith`.
    """
    return re.sub(r"(\{%[-+]?\s*)(end)?generation(\s*[-+]?%\})", r"\1\2with\3", template)


def _minijinja_env() -> "minijinja.Environment":
    r"""Build a minijinja environment in non-Python-compat mode, like HF Transformers.

    `pycompat=False` disables the Python-specific methods (e.g. `dict.get`) that
    minijinja otherwise tolerates, matching the non-Python Jinja runtimes the
    templates must support. The `raise_exception` global matches what HF registers.

    Returns:
        A minijinja `Environment` with `pycompat` disabled and `raise_exception` registered.
    """

    def raise_exception(message: str) -> None:
        raise minijinja.TemplateError(message)

    env = minijinja.Environment(pycompat=False)
    env.add_global("raise_exception", raise_exception)
    return env


@pytest.mark.parametrize("version", list(TokenizerVersion))
def test_generated_template_renders_under_minijinja(version: TokenizerVersion) -> None:
    # HF Transformers renders chat templates with a non-Python Jinja runtime (minijinja
    # in Rust with pycompat off), where Python-only constructs like `dict.get()` are
    # unknown methods. Rendering must succeed for every version and feature variant.
    env = _minijinja_env()
    for config in _configs_for_version(version):
        template = _rewrite_generation_blocks(build_chat_template(config))
        render_kwargs: dict[str, Any] = {
            "messages": _RENDER_MESSAGES,
            "tools": _RENDER_TOOLS,
            "response_format": {"type": "json_object"},
            "add_generation_prompt": False,
            "bos_token": "<s>",
            "eos_token": "</s>",
        }
        if config.version.supports_model_settings:
            render_kwargs["reasoning_effort"] = "low"
        rendered = env.render_str(template, **render_kwargs)
        assert rendered.startswith("<s>")

from typing import Any

import pytest

from mistral_common.integrations.chat_templates.template_generator import build_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.conftest import ALL_CONFIGS, _config_id
from tests.integrations.chat_templates.fixtures_data import _iter_render_args, _with_dict_tool_arguments
from tests.integrations.chat_templates.helpers import (
    TestConfig,
    _load_golden_template,
    _make_config,
    render_template,
)


@pytest.mark.parametrize(
    "config",
    [c for c in ALL_CONFIGS if not c.plain_think],
    ids=[_config_id(c) for c in ALL_CONFIGS if not c.plain_think],
)
def test_dynamic_matches_static(config: TestConfig) -> None:
    template_config = _make_config(config)
    dynamic = build_chat_template(template_config)
    static = _load_golden_template(template_config)
    assert dynamic == static


@pytest.mark.parametrize(
    "config",
    [c for c in ALL_CONFIGS if not c.plain_think],
    ids=[_config_id(c) for c in ALL_CONFIGS if not c.plain_think],
)
def test_dynamic_template_produces_same_output(config: TestConfig) -> None:
    template_config = _make_config(config)
    static_template = _load_golden_template(template_config)
    dynamic_template = build_chat_template(template_config)

    # Test with a simple conversation
    messages: list[Any] = [
        {"role": "user", "content": "Hello"},
    ]

    # Only add assistant message for training validation
    if config.version != TokenizerVersion.v1:
        messages.append({"role": "assistant", "content": "Hi there!"})

    static_output = render_template(static_template, messages)
    dynamic_output = render_template(dynamic_template, messages)

    assert static_output == dynamic_output, (
        f"Output mismatch for {config}\n\nStatic output: {static_output}\nDynamic output: {dynamic_output}"
    )


# Comprehensive functional tests — covers all configs except plain_think
# (plain think template parity is tested separately in test_unit.py::test_plain_think_static_dynamic_parity)
@pytest.mark.parametrize(
    "config",
    [c for c in ALL_CONFIGS if not c.plain_think],
    ids=[_config_id(c) for c in ALL_CONFIGS if not c.plain_think],
)
def test_dynamic_template_comprehensive(config: TestConfig) -> None:
    template_config = _make_config(config)
    static_template = _load_golden_template(template_config)
    dynamic_template = build_chat_template(template_config)

    for mode, idx, render_args in _iter_render_args(config):
        static_output = render_template(static_template, **render_args)
        dynamic_output = render_template(dynamic_template, **render_args)
        assert static_output == dynamic_output, (
            f"Output mismatch for {config}, mode={mode}, conversation={idx}\n\n"
            f"Static output: {static_output}\n"
            f"Dynamic output: {dynamic_output}"
        )


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=[_config_id(c) for c in ALL_CONFIGS])
def test_dict_tool_arguments_render_like_string_arguments(config: TestConfig) -> None:
    # Stock Jinja's `tojson` and `safe` return Markup, and `str + Markup` HTML-escapes the string side.
    template = build_chat_template(_make_config(config))

    for mode, idx, render_args in _iter_render_args(config):
        string_output = render_template(template, **render_args)
        dict_output = render_template(template, **_with_dict_tool_arguments(render_args))

        assert dict_output == string_output, f"Output mismatch for {config}, mode={mode}, conversation={idx}"

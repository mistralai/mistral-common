from typing import Any

import pytest

from mistral_common.integrations.chat_templates.template_generator import build_chat_template
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.conftest import ALL_CONFIGS, _config_id
from tests.integrations.chat_templates.fixtures_data import _get_conversations
from tests.integrations.chat_templates.helpers import (
    TestConfig,
    _load_golden_template,
    _make_config,
    render_template,
)


def _request_to_render_args(request: ChatCompletionRequest) -> dict[str, Any]:
    r"""Convert a ChatCompletionRequest to render_template kwargs.

    Enriches tool messages with the ``name`` field resolved from the preceding
    assistant's ``tool_calls``, which the v2 Jinja template requires but
    ``ToolMessage.to_openai()`` omits.
    """
    openai = request.to_openai()
    messages = openai["messages"]

    # Build a mapping from tool_call_id -> function name so tool messages
    # can carry the ``name`` field expected by v2 templates.
    tool_call_names: dict[str, str] = {}
    for msg in messages:
        for tc in msg.get("tool_calls", []):
            tc_id = tc.get("id")
            fn_name = tc.get("function", {}).get("name")
            if tc_id and fn_name:
                tool_call_names[tc_id] = fn_name

    for msg in messages:
        if msg.get("role") == "tool" and "name" not in msg:
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id in tool_call_names:
                msg["name"] = tool_call_names[tc_id]

    kwargs: dict[str, Any] = {
        "messages": messages,
    }
    if "tools" in openai and openai["tools"]:
        kwargs["tools"] = openai["tools"]
    reasoning = openai.get("reasoning_effort")
    if reasoning is not None:
        kwargs["reasoning_effort"] = reasoning
    return kwargs


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

    for mode in (ValidationMode.finetuning, ValidationMode.test):
        conversations = _get_conversations(config.version, mode, config.image, config.audio, config.think)
        for idx, request in enumerate(conversations):
            render_args = _request_to_render_args(request)
            static_output = render_template(static_template, **render_args)
            dynamic_output = render_template(dynamic_template, **render_args)
            assert static_output == dynamic_output, (
                f"Output mismatch for {config}, mode={mode}, conversation={idx}\n\n"
                f"Static output: {static_output}\n"
                f"Dynamic output: {dynamic_output}"
            )

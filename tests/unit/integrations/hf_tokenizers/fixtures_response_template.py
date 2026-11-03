r"""Shared expected `response_template` dicts for the hf_tokenizers tests.

Each fixture is the exact dict `generate_response_template` must return for one
configuration, written out literally (not derived from the implementation, so a
regression in the builder cannot silently regenerate the expectation). `PR_*`
entries match the HF PR blocks verbatim (v1: Mistral-7B-Instruct-v0.1 #190, v7:
Mistral-Small-3.1-24B-Instruct-2503 #86, v13: Mistral-Large-4-1T-A52B #8) and are
additionally compared as JSON strings in `test_expected_fixtures`. `V11_*` entries
have no PR counterpart and follow the v11 wire format. All fixtures are frozen
with `types.MappingProxyType` (recursively) so tests cannot mutate them.
"""

import json
from types import MappingProxyType
from typing import Any

from mistral_common.tokens.tokenizers.base import (
    PLAIN_BEGIN_THINK,
    PLAIN_END_THINK,
    SpecialTokens,
)


def _freeze(obj: Any) -> Any:
    r"""Recursively wrap dicts in MappingProxyType (lists stay lists)."""
    if isinstance(obj, dict):
        return MappingProxyType({key: _freeze(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_freeze(item) for item in obj]
    return obj


def _expected(raw: dict[str, Any]) -> MappingProxyType[str, Any]:
    r"""Freeze a literal expected dict."""
    frozen = _freeze(raw)
    assert isinstance(frozen, MappingProxyType)
    return frozen


def _expected_json(raw: dict[str, Any]) -> str:
    r"""Serialize the literal (unfrozen) dict to the exact comparison string."""
    return json.dumps(obj=raw, indent=2)


_COMMON_FIELDS_TAIL: dict[str, Any] = {
    "content": {
        "close": [SpecialTokens.tool_calls.value, SpecialTokens.eos.value],
        "content": "text",
        "repeats": True,
        "join": "",
    },
}

_PR_V1_RAW: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "start_anchor": SpecialTokens.end_inst.value,
    "fields": {
        "content": {"close": SpecialTokens.eos.value, "content": "text"},
    },
}

_PR_V7_RAW: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "start_anchor": [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value],
    "fields": {
        "tool_calls": {
            "open": SpecialTokens.tool_calls.value,
            "close": SpecialTokens.eos.value,
            "content": "json",
            "transform_each": True,
            "transform": {
                "id": "{id}",
                "type": "function",
                "function": {"name": "{name}", "arguments": "{arguments}"},
            },
        },
        **_COMMON_FIELDS_TAIL,
    },
}

_PR_V13_RAW: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "start_anchor": [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value],
    "fields": {
        "reasoning_content": {
            "open": SpecialTokens.begin_think.value,
            "close": SpecialTokens.end_think.value,
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": "\\[TOOL_CALLS\\](?P<name>.+?)\\[ARGS\\]",
            "close_pattern": "(?=\\[TOOL_CALLS\\])|</s>",
            "repeats": True,
            "content": "json",
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        **_COMMON_FIELDS_TAIL,
    },
}

_V11_PLAIN_RAW: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "start_anchor": [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value],
    "fields": {
        "reasoning_content": {
            "open": PLAIN_BEGIN_THINK,
            "close": PLAIN_END_THINK,
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": "\\[TOOL_CALLS\\](?P<name>.+?)\\[CALL_ID\\](?P<id>.+?)\\[ARGS\\]",
            "close_pattern": "(?=\\[TOOL_CALLS\\])|</s>",
            "repeats": True,
            "content": "json",
            "transform": {
                "type": "function",
                "id": "{id}",
                "function": {"name": "{name}", "arguments": "{content}"},
            },
        },
        **_COMMON_FIELDS_TAIL,
    },
}

_V11_NO_THINKING_RAW: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "start_anchor": [SpecialTokens.end_inst.value, SpecialTokens.end_tool_results.value],
    "fields": {
        "tool_calls": _V11_PLAIN_RAW["fields"]["tool_calls"],
        **_COMMON_FIELDS_TAIL,
    },
}

PR_V1 = _expected(_PR_V1_RAW)
PR_V7 = _expected(_PR_V7_RAW)
PR_V13 = _expected(_PR_V13_RAW)
V11_PLAIN_THINKING = _expected(_V11_PLAIN_RAW)
V11_NO_THINKING = _expected(_V11_NO_THINKING_RAW)

# Exact JSON text of the HF PR blocks, for the byte-for-byte string comparison.
PR_V1_JSON = _expected_json(_PR_V1_RAW)
PR_V7_JSON = _expected_json(_PR_V7_RAW)
PR_V13_JSON = _expected_json(_PR_V13_RAW)

assert PLAIN_BEGIN_THINK == "<think>" and PLAIN_END_THINK == "</think>"
assert json.loads(PR_V13_JSON) == _PR_V13_RAW  # fixtures must stay JSON-serializable

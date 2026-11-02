import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mistral_common.integrations.hf_tokenizers.response_template import (
    convert_tokenizer_to_response_template,
    generate_response_template,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import TestConfig, build_tekken_json

V1_PR_FIXTURE = """{
  "defaults": {
    "role": "assistant"
  },
  "start_anchor": "[/INST]",
  "fields": {
    "content": {
      "close": "</s>",
      "content": "text"
    }
  }
}"""

V7_PR_FIXTURE = """{
  "defaults": {
    "role": "assistant"
  },
  "start_anchor": [
    "[/INST]",
    "[/TOOL_RESULTS]"
  ],
  "fields": {
    "tool_calls": {
      "open": "[TOOL_CALLS]",
      "close": "</s>",
      "content": "json",
      "transform_each": true,
      "transform": {
        "id": "{id}",
        "type": "function",
        "function": {
          "name": "{name}",
          "arguments": "{arguments}"
        }
      }
    },
    "content": {
      "close": [
        "[TOOL_CALLS]",
        "</s>"
      ],
      "content": "text",
      "repeats": true,
      "join": ""
    }
  }
}"""

V13_PR_FIXTURE = """{
  "defaults": {
    "role": "assistant"
  },
  "start_anchor": [
    "[/INST]",
    "[/TOOL_RESULTS]"
  ],
  "fields": {
    "reasoning_content": {
      "open": "[THINK]",
      "close": "[/THINK]",
      "content": "text"
    },
    "tool_calls": {
      "open_pattern": "\\\\[TOOL_CALLS\\\\](?P<name>.+?)\\\\[ARGS\\\\]",
      "close_pattern": "(?=\\\\[TOOL_CALLS\\\\])|</s>",
      "repeats": true,
      "content": "json",
      "transform": {
        "type": "function",
        "function": {
          "name": "{name}",
          "arguments": "{content}"
        }
      }
    },
    "content": {
      "close": [
        "[TOOL_CALLS]",
        "</s>"
      ],
      "content": "text",
      "repeats": true,
      "join": ""
    }
  }
}"""


def test_top_level_keys_and_json_roundtrip() -> None:
    for version in (
        TokenizerVersion.v1,
        TokenizerVersion.v2,
        TokenizerVersion.v3,
        TokenizerVersion.v7,
        TokenizerVersion.v11,
        TokenizerVersion.v13,
        TokenizerVersion.v15,
    ):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )

        assert json.loads(s=json.dumps(obj=template)) == template
        assert set(template) == {"defaults", "start_anchor", "fields"}
        assert template["defaults"] == {"role": "assistant"}


def test_byte_for_byte_pr_fixtures() -> None:
    fixtures: tuple[tuple[TokenizerVersion, bool, bool, str], ...] = (
        (TokenizerVersion.v1, False, False, V1_PR_FIXTURE),
        (TokenizerVersion.v7, False, False, V7_PR_FIXTURE),
        (TokenizerVersion.v13, True, False, V13_PR_FIXTURE),
    )

    for version, thinking_support, plain_thinking_support, fixture in fixtures:
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=thinking_support,
            plain_thinking_support=plain_thinking_support,
        )

        assert json.dumps(obj=template, indent=2) == fixture


def test_start_anchor_per_version() -> None:
    v1 = generate_response_template(
        tokenizer_version=TokenizerVersion.v1,
        thinking_support=False,
        plain_thinking_support=False,
    )
    assert v1["start_anchor"] == "[/INST]"

    for version in (
        TokenizerVersion.v2,
        TokenizerVersion.v3,
        TokenizerVersion.v7,
        TokenizerVersion.v11,
        TokenizerVersion.v13,
        TokenizerVersion.v15,
    ):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        assert template["start_anchor"] == ["[/INST]", "[/TOOL_RESULTS]"]


def test_v13_tool_calls_pattern() -> None:
    template = generate_response_template(
        tokenizer_version=TokenizerVersion.v13,
        thinking_support=True,
        plain_thinking_support=False,
    )

    assert template["fields"]["tool_calls"] == {
        "open_pattern": "\\[TOOL_CALLS\\](?P<name>.+?)\\[ARGS\\]",
        "close_pattern": "(?=\\[TOOL_CALLS\\])|</s>",
        "repeats": True,
        "content": "json",
        "transform": {
            "type": "function",
            "function": {"name": "{name}", "arguments": "{content}"},
        },
    }


def test_v11_tool_calls_include_call_id() -> None:
    template = generate_response_template(
        tokenizer_version=TokenizerVersion.v11,
        thinking_support=False,
        plain_thinking_support=True,
    )

    tool_calls = template["fields"]["tool_calls"]
    assert "\\[CALL_ID\\](?P<id>.+?)" in tool_calls["open_pattern"]
    assert tool_calls["transform"] == {
        "type": "function",
        "id": "{id}",
        "function": {"name": "{name}", "arguments": "{content}"},
    }
    assert template["fields"]["reasoning_content"] == {
        "open": "<think>",
        "close": "</think>",
        "content": "text",
    }

    without_plain_thinking = generate_response_template(
        tokenizer_version=TokenizerVersion.v11,
        thinking_support=False,
        plain_thinking_support=False,
    )
    assert "reasoning_content" not in without_plain_thinking["fields"]


def test_v3_v7_tool_calls_json_array() -> None:
    expected = {
        "open": "[TOOL_CALLS]",
        "close": "</s>",
        "content": "json",
        "transform_each": True,
        "transform": {
            "id": "{id}",
            "type": "function",
            "function": {"name": "{name}", "arguments": "{arguments}"},
        },
    }

    for version in (TokenizerVersion.v3, TokenizerVersion.v7):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        assert template["fields"]["tool_calls"] == expected


def test_v2_tool_calls_no_id() -> None:
    template = generate_response_template(
        tokenizer_version=TokenizerVersion.v2,
        thinking_support=False,
        plain_thinking_support=False,
    )

    assert template["fields"]["tool_calls"]["transform"] == {
        "type": "function",
        "function": {"name": "{name}", "arguments": "{arguments}"},
    }


def test_reasoning_field_rules() -> None:
    expected_reasoning = {"open": "[THINK]", "close": "[/THINK]", "content": "text"}
    for version in (TokenizerVersion.v13, TokenizerVersion.v15):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        assert template["fields"]["reasoning_content"] == expected_reasoning

    for version in (TokenizerVersion.v2, TokenizerVersion.v3, TokenizerVersion.v7):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        assert "reasoning_content" not in template["fields"]


def test_content_field() -> None:
    v1 = generate_response_template(
        tokenizer_version=TokenizerVersion.v1,
        thinking_support=False,
        plain_thinking_support=False,
    )
    assert v1["fields"]["content"] == {"close": "</s>", "content": "text"}

    for version in (
        TokenizerVersion.v2,
        TokenizerVersion.v3,
        TokenizerVersion.v7,
        TokenizerVersion.v11,
        TokenizerVersion.v13,
        TokenizerVersion.v15,
    ):
        template = generate_response_template(
            tokenizer_version=version,
            thinking_support=False,
            plain_thinking_support=False,
        )
        content: dict[str, Any] = template["fields"]["content"]
        assert "open" not in content
        assert "open_pattern" not in content
        assert content["close"] == ["[TOOL_CALLS]", "</s>"]
        assert content["repeats"] is True
        assert content["join"] == ""
        assert content["content"] == "text"


def test_invalid_thinking_combos_raise() -> None:
    with pytest.raises(ValueError, match="Plain thinking support is only available"):
        generate_response_template(
            tokenizer_version=TokenizerVersion.v7,
            thinking_support=False,
            plain_thinking_support=True,
        )

    with pytest.raises(ValueError, match="Special-token thinking support requires"):
        generate_response_template(
            tokenizer_version=TokenizerVersion.v7,
            thinking_support=True,
            plain_thinking_support=False,
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_response_template(
            tokenizer_version=TokenizerVersion.v13,
            thinking_support=True,
            plain_thinking_support=True,
        )


def test_no_chat_templates_import() -> None:
    module_import = (
        "import sys; import mistral_common.integrations.hf_tokenizers.response_template; "
        "assert 'mistral_common.integrations.chat_templates' not in sys.modules"
    )
    result = subprocess.run(
        args=[sys.executable, "-c", module_import],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[4],
    )

    assert result.returncode == 0, result.stderr


def test_no_delimiter_string_literals() -> None:
    module_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "mistral_common"
        / "integrations"
        / "hf_tokenizers"
        / "response_template.py"
    )
    source = module_path.read_text(encoding="utf-8")

    for delimiter in (
        "[" + "TOOL_CALLS]",
        "[" + "/INST]",
        "[" + "TOOL_RESULTS]",
        "[" + "ARGS]",
        "[" + "CALL_ID]",
        "[" + "THINK]",
        "<" + "/s>",
    ):
        assert delimiter not in source
    assert "re.escape" in source


def test_convert_tokenizer_to_response_template_tekken_v13(tmp_path: Path) -> None:
    path = build_tekken_json(
        config=TestConfig(version=TokenizerVersion.v13, think=True),
        output_dir=tmp_path,
    )

    result = convert_tokenizer_to_response_template(tokenizer_file=path, plain_thinking_support=None)
    expected = generate_response_template(
        tokenizer_version=TokenizerVersion.v13,
        thinking_support=True,
        plain_thinking_support=False,
    )
    assert result == expected


def test_convert_v11_plain_thinking_heuristic(tmp_path: Path) -> None:
    path = build_tekken_json(
        config=TestConfig(version=TokenizerVersion.v11),
        output_dir=tmp_path,
    )

    heuristic_result = convert_tokenizer_to_response_template(
        tokenizer_file=path,
        plain_thinking_support=None,
    )
    assert heuristic_result["fields"]["reasoning_content"]["open"] == "<think>"

    override_result = convert_tokenizer_to_response_template(
        tokenizer_file=path,
        plain_thinking_support=False,
    )
    assert "reasoning_content" not in override_result["fields"]

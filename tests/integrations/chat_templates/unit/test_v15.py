import json
import re
from pathlib import Path
from typing import Any

import pytest
from jinja2 import BaseLoader
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mistral_common.integrations.chat_templates.chat_templates import (
    convert_tokenizer_to_chat_template,
    generate_chat_template,
)
from mistral_common.integrations.chat_templates.template_generator import TemplateConfig, build_chat_template
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integrations.chat_templates.fixtures_data import PARITY_JSON_SCHEMA, PARITY_JSON_SCHEMA_NON_ASCII
from tests.integrations.chat_templates.helpers import (
    LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
    LOW_HIGH_MAX_REASONING_EFFORT_MAPPED_DEFAULT,
    LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
    LOW_HIGH_MAX_REASONING_EFFORT_VALUES,
    NONE_HIGH_REASONING_EFFORT_DEFAULT,
    NONE_HIGH_REASONING_EFFORT_VALUES,
    build_reasoning_effort_tekken_json,
    build_reasoning_effort_template,
    build_render_args,
    encode_mistral_common,
    render_template,
)
from tests.utils import decode_keep

_MESSAGES = [{"role": "user", "content": "Hello"}]


def _reasoning_effort_settings(reasoning_effort: str) -> str:
    r"""The `[MODEL_SETTINGS]` block emitted for a reasoning effort value."""
    return f'[MODEL_SETTINGS]{{"reasoning_effort": "{reasoning_effort}"}}[/MODEL_SETTINGS]'


class TestV15ModelSettings:
    @pytest.fixture()
    def v15_template(self) -> str:
        return generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none",
        )

    def test_v15_template_contains_model_settings(self, v15_template: str) -> None:
        assert "[MODEL_SETTINGS]" in v15_template
        assert "reasoning_effort" in v15_template

    @pytest.mark.parametrize(
        ("reasoning_effort", "expected_value"),
        [
            (None, "none"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "max"),
        ],
    )
    def test_v15_reasoning_effort_rendering(
        self, v15_template: str, reasoning_effort: str | None, expected_value: str
    ) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        output = render_template(v15_template, messages, reasoning_effort=reasoning_effort)
        assert f'[MODEL_SETTINGS]{{"reasoning_effort": "{expected_value}"}}[/MODEL_SETTINGS]' in output

    def test_v15_with_image_support(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=True,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none",
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": "http://example.com/image.png"},
                ],
            },
            {"role": "assistant", "content": "It's an image."},
        ]

        output = render_template(template, messages, reasoning_effort="high")
        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output
        assert "[IMG]" in output

    def test_v15_with_thinking_support(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none",
        )

        messages = [
            {"role": "user", "content": "Solve this problem"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "The answer is 42."},
                ],
            },
        ]

        output = render_template(template, messages, reasoning_effort="none")
        assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output
        assert "[THINK]Let me think...[/THINK]" in output

    def test_v15_with_tools(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none",
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "The city and state"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
        ]

        output = render_template(template, messages, tools=tools, reasoning_effort="high")
        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output
        assert "[AVAILABLE_TOOLS]" in output
        assert "get_weather" in output

    @pytest.mark.parametrize(
        ("has_system", "has_tools", "reasoning_effort"),
        [
            (True, True, "high"),
            (True, False, "high"),
            (False, True, "high"),
            (False, False, "high"),
            (True, True, "none"),
            (False, False, "none"),
            (True, True, None),
            (False, False, None),
        ],
    )
    def test_v15_available_tools_and_settings_ordering(
        self, has_system: bool, has_tools: bool, reasoning_effort: str | None
    ) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none",
        )

        messages: list[dict[str, Any]] = []
        if has_system:
            messages.append({"role": "system", "content": "You are helpful."})
        messages.extend(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        )

        tools: list[dict[str, Any]] | None = None
        if has_tools:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "tool1",
                        "description": "",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        output = render_template(template, messages, tools=tools, reasoning_effort=reasoning_effort)

        if has_system:
            assert "[SYSTEM_PROMPT]You are helpful.[/SYSTEM_PROMPT]" in output
            sp_pos = output.index("[SYSTEM_PROMPT]")
        else:
            assert "[SYSTEM_PROMPT]" not in output
            sp_pos = -1

        if has_tools:
            assert "[AVAILABLE_TOOLS]" in output
            tools_pos = output.index("[AVAILABLE_TOOLS]")
            assert tools_pos > sp_pos
        else:
            assert "[AVAILABLE_TOOLS]" not in output
            tools_pos = sp_pos

        assert "[MODEL_SETTINGS]" in output
        settings_pos = output.index("[MODEL_SETTINGS]")
        assert settings_pos > tools_pos

        inst_pos = output.index("[INST]")
        assert inst_pos > settings_pos

    @pytest.mark.parametrize("image", [False, True])
    def test_v15_think_template_rejects_think_in_system(self, image: bool) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=image,
            audio_support=False,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none",
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "System text."},
                    {"type": "thinking", "thinking": "System thinking."},
                ],
            },
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        with pytest.raises(ValueError, match="Only text chunks are supported in system message contents"):
            render_template(template, messages)

    def test_v15_json_schema_only_renders_custom_schema(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": PARITY_JSON_SCHEMA},
        }

        output = render_template(template=template, messages=messages, response_format=response_format)

        expected_settings = f'[MODEL_SETTINGS]{{"json_schema": {json.dumps(obj=PARITY_JSON_SCHEMA)}}}[/MODEL_SETTINGS]'
        assert expected_settings in output

    def test_v15_json_schema_non_ascii_html_specials_full_parity(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": PARITY_JSON_SCHEMA_NON_ASCII},
        }

        output = render_template(template=template, messages=messages, response_format=response_format)

        expected_settings = (
            f'[MODEL_SETTINGS]{{"json_schema": '
            f"{json.dumps(obj=PARITY_JSON_SCHEMA_NON_ASCII, ensure_ascii=False)}}}[/MODEL_SETTINGS]"
        )
        assert expected_settings in output
        assert "\\u00e9" not in output
        assert "&lt;" not in output
        assert "&gt;" not in output
        assert "&amp;" not in output

    def test_v15_json_schema_only_json_object_generic(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        output = render_template(
            template=template,
            messages=messages,
            response_format={"type": "json_object"},
        )

        assert (
            '[MODEL_SETTINGS]{"json_schema": {"anyOf": [{"type": "object"}, '
            '{"type": "array"}]}}[/MODEL_SETTINGS]' in output
        )

    def test_v15_json_schema_only_text_no_block(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        output = render_template(
            template=template,
            messages=messages,
            response_format={"type": "text"},
        )

        assert "[MODEL_SETTINGS]" not in output

    def test_v15_unknown_response_format_type_raises(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        with pytest.raises(
            ValueError,
            match=re.escape("Response format type must be one of text, json_object or json_schema, got xml."),
        ):
            render_template(template=template, messages=messages, response_format={"type": "xml"})

    def test_v15_json_object_ignores_schema(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        output = render_template(
            template=template,
            messages=messages,
            response_format={"type": "json_object", "json_schema": {"name": "answer", "schema": {"type": "string"}}},
        )

        assert '[MODEL_SETTINGS]{"json_schema": {"anyOf": [{"type": "object"}, {"type": "array"}]}}' in output

    def test_v15_response_format_without_type_defaults_to_text(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        output = render_template(template=template, messages=messages, response_format={})

        assert "[MODEL_SETTINGS]" not in output

    def test_v15_json_schema_only_missing_kwarg_no_block(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        output = render_template(template=template, messages=messages)

        assert "[MODEL_SETTINGS]" not in output

    @pytest.mark.parametrize(
        "json_schema",
        [{"name": "answer"}, {"name": "answer", "schema": None}, None],
        ids=["missing_schema", "null_schema", "null_json_schema"],
    )
    def test_v15_json_schema_missing_schema_raises(self, json_schema: dict[str, Any] | None) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        with pytest.raises(ValueError, match="Response format `json_schema` must define the schema"):
            render_template(
                template=template,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": json_schema},
            )

    @pytest.mark.parametrize(
        "model_settings_fields",
        [frozenset({"json_schema"}), frozenset({"reasoning_effort", "json_schema"})],
        ids=["json_schema", "both"],
    )
    def test_v15_json_schema_renders_with_stock_jinja_tojson(self, model_settings_fields: frozenset[str]) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=model_settings_fields,
                reasoning_effort_default="none" if "reasoning_effort" in model_settings_fields else None,
            )
        )
        environment = ImmutableSandboxedEnvironment(loader=BaseLoader())

        output = environment.from_string(template).render(
            messages=[{"role": "user", "content": "Hello"}],
            bos_token="<s>",
            eos_token="</s>",
            response_format={"type": "json_object"},
        )

        assert '[MODEL_SETTINGS]{"json_schema": {"anyOf": ' in output

    def test_v15_both_fields_sorted_keys(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
                reasoning_effort_default="none",
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": PARITY_JSON_SCHEMA},
        }

        output = render_template(
            template=template,
            messages=messages,
            reasoning_effort="high",
            response_format=response_format,
        )

        expected_settings = (
            f'[MODEL_SETTINGS]{{"json_schema": {json.dumps(obj=PARITY_JSON_SCHEMA)}, '
            '"reasoning_effort": "high"}[/MODEL_SETTINGS]'
        )
        assert expected_settings in output

    def test_v15_both_fields_json_object(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
                reasoning_effort_default="none",
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        output = render_template(
            template=template,
            messages=messages,
            response_format={"type": "json_object"},
        )

        assert (
            '[MODEL_SETTINGS]{"json_schema": {"anyOf": [{"type": "object"}, '
            '{"type": "array"}]}, "reasoning_effort": "none"}[/MODEL_SETTINGS]' in output
        )

    def test_v15_both_fields_text_reasoning_effort_only(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
                reasoning_effort_default="none",
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        output = render_template(
            template=template,
            messages=messages,
            reasoning_effort="high",
            response_format={"type": "text"},
        )

        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output
        assert '"json_schema"' not in output

    def test_v15_both_fields_invalid_reasoning_effort_raises(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
                reasoning_effort_default="none",
            )
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        with pytest.raises(ValueError, match="reasoning_effort must be one of"):
            render_template(
                template=template,
                messages=messages,
                reasoning_effort="minimal",
            )

    def test_v15_reasoning_effort_only_ignores_response_format(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(version=TokenizerVersion.v15, reasoning_effort_default="none")
        )
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": PARITY_JSON_SCHEMA},
        }

        output = render_template(template=template, messages=messages, response_format=response_format)

        assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output
        assert '"json_schema"' not in output

    def test_v15_both_fields_with_tools_placement(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
                reasoning_effort_default="none",
            )
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Find a value",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": PARITY_JSON_SCHEMA},
        }

        output = render_template(
            template=template,
            messages=messages,
            tools=tools,
            reasoning_effort="high",
            response_format=response_format,
        )

        tools_position = output.index("[AVAILABLE_TOOLS]")
        settings_position = output.index("[MODEL_SETTINGS]")
        first_user_position = output.index("[INST]")
        assert tools_position < settings_position < first_user_position


class TestGeneratedTemplateMapping:
    def test_mapping_is_baked_into_template(self) -> None:
        template = build_reasoning_effort_template(
            reasoning_effort_mapping=LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        assert "{%- set reasoning_effort_mapping = {'low': 'low', 'high': 'medium', 'max': 'high'} %}" in template
        assert "{%- if reasoning_effort not in reasoning_effort_mapping %}" in template
        assert "{%- set reasoning_effort = reasoning_effort_mapping[reasoning_effort] %}" in template

    def test_template_without_mapping_validates_the_full_ladder(self) -> None:
        template = build_reasoning_effort_template(reasoning_effort_default="none")
        assert "reasoning_effort_mapping" not in template
        assert "{#- The model was trained" not in template
        assert "{%- if reasoning_effort not in ['none', 'low', 'medium', 'high', 'xhigh', 'max'] %}" in template

    def test_template_without_mapping_validates_exposed_values(self) -> None:
        template = build_reasoning_effort_template(
            reasoning_effort_values=NONE_HIGH_REASONING_EFFORT_VALUES,
            reasoning_effort_default=NONE_HIGH_REASONING_EFFORT_DEFAULT,
        )
        assert "reasoning_effort_mapping" not in template
        assert "{#- The model was trained" not in template
        assert "{%- if reasoning_effort not in ['none', 'high'] %}" in template
        assert 'reasoning_effort must be either "none" or "high"' in template

    def test_mapping_comment_explains_the_remap(self) -> None:
        template = build_reasoning_effort_template(
            reasoning_effort_mapping=LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        comment = "{#- Exposed reasoning efforts are remapped onto the values the model was trained on. -#}"
        assert comment in template
        assert template.index(comment) < template.index("{%- set reasoning_effort_mapping")

    def test_default_is_baked_into_template(self) -> None:
        template = build_reasoning_effort_template(
            reasoning_effort_mapping=LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        assert "{%- set reasoning_effort = 'high' %}" in template

    def test_template_without_default_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Reasoning effort default is required"):
            build_reasoning_effort_template()

    @pytest.mark.parametrize("reasoning_effort", NONE_HIGH_REASONING_EFFORT_VALUES)
    def test_render_without_mapping_keeps_exposed_value(self, reasoning_effort: str) -> None:
        template = build_reasoning_effort_template(
            reasoning_effort_values=NONE_HIGH_REASONING_EFFORT_VALUES,
            reasoning_effort_default=NONE_HIGH_REASONING_EFFORT_DEFAULT,
        )
        rendered = render_template(template, messages=_MESSAGES, reasoning_effort=reasoning_effort)
        assert _reasoning_effort_settings(reasoning_effort) in rendered

    def test_render_unset_effort_uses_default(self) -> None:
        template = build_reasoning_effort_template(
            reasoning_effort_values=NONE_HIGH_REASONING_EFFORT_VALUES,
            reasoning_effort_default=NONE_HIGH_REASONING_EFFORT_DEFAULT,
        )
        rendered = render_template(template, messages=_MESSAGES)
        assert _reasoning_effort_settings(NONE_HIGH_REASONING_EFFORT_DEFAULT) in rendered

    def test_render_rejects_value_outside_exposed_values(self) -> None:
        template = build_reasoning_effort_template(
            reasoning_effort_values=NONE_HIGH_REASONING_EFFORT_VALUES,
            reasoning_effort_default=NONE_HIGH_REASONING_EFFORT_DEFAULT,
        )
        with pytest.raises(ValueError, match='reasoning_effort must be either "none" or "high"'):
            render_template(template, messages=_MESSAGES, reasoning_effort="low")


class TestNoneHighReasoningEffortWithoutRemapping:
    r"""The v15 configuration exposing `none` and `high`, both supported as-is."""

    @pytest.mark.parametrize("reasoning_effort", [ReasoningEffort.none, ReasoningEffort.high])
    def test_exposed_values_encode_as_is(self, tmp_path: Path, reasoning_effort: ReasoningEffort) -> None:
        tekken_path = build_reasoning_effort_tekken_json(
            output_dir=tmp_path,
            reasoning_effort_values=NONE_HIGH_REASONING_EFFORT_VALUES,
            reasoning_effort_default=NONE_HIGH_REASONING_EFFORT_DEFAULT,
        )
        tokenizer = MistralTokenizer.from_file(str(tekken_path))
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=reasoning_effort)
        text = decode_keep(tokenizer, tokenizer.encode_chat_completion(request))
        assert _reasoning_effort_settings(reasoning_effort.value) in text

    def test_unset_effort_encodes_default(self, tmp_path: Path) -> None:
        tekken_path = build_reasoning_effort_tekken_json(
            output_dir=tmp_path,
            reasoning_effort_values=NONE_HIGH_REASONING_EFFORT_VALUES,
            reasoning_effort_default=NONE_HIGH_REASONING_EFFORT_DEFAULT,
        )
        tokenizer = MistralTokenizer.from_file(str(tekken_path))
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")])
        text = decode_keep(tokenizer, tokenizer.encode_chat_completion(request))
        assert _reasoning_effort_settings(NONE_HIGH_REASONING_EFFORT_DEFAULT) in text

    @pytest.mark.parametrize(
        "reasoning_effort",
        [ReasoningEffort.none, ReasoningEffort.high, None],
        ids=["none", "high", "unset"],
    )
    def test_template_and_tokenizer_agree(self, tmp_path: Path, reasoning_effort: ReasoningEffort | None) -> None:
        tekken_path = build_reasoning_effort_tekken_json(
            output_dir=tmp_path,
            reasoning_effort_values=NONE_HIGH_REASONING_EFFORT_VALUES,
            reasoning_effort_default=NONE_HIGH_REASONING_EFFORT_DEFAULT,
        )
        tokenizer = MistralTokenizer.from_file(str(tekken_path))
        template = convert_tokenizer_to_chat_template(tokenizer_file=tekken_path)
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=reasoning_effort)

        tokenized = encode_mistral_common(mistral_tokenizer=tokenizer, chat_request=request, spm=False)
        rendered = render_template(template, **build_render_args(request))

        expected = _reasoning_effort_settings(
            reasoning_effort.value if reasoning_effort is not None else NONE_HIGH_REASONING_EFFORT_DEFAULT
        )
        assert expected in tokenized
        assert tokenized == rendered


class TestLowHighMaxReasoningEffortRemapping:
    r"""The v15 configuration exposing `low`, `high` and `max`, remapped onto `low`, `medium` and `high`."""

    @pytest.mark.parametrize("reasoning_effort", LOW_HIGH_MAX_REASONING_EFFORT_VALUES)
    def test_render_maps_exposed_values(self, reasoning_effort: str) -> None:
        template = build_reasoning_effort_template(
            reasoning_effort_mapping=LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        rendered = render_template(template, messages=_MESSAGES, reasoning_effort=reasoning_effort)
        model_side_value = LOW_HIGH_MAX_REASONING_EFFORT_MAPPING[reasoning_effort]
        assert _reasoning_effort_settings(model_side_value) in rendered

    def test_render_unset_effort_maps_default(self) -> None:
        template = build_reasoning_effort_template(
            reasoning_effort_mapping=LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        rendered = render_template(template, messages=_MESSAGES)
        assert _reasoning_effort_settings(LOW_HIGH_MAX_REASONING_EFFORT_MAPPED_DEFAULT) in rendered

    def test_render_rejects_value_outside_mapping(self) -> None:
        template = build_reasoning_effort_template(
            reasoning_effort_mapping=LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        with pytest.raises(ValueError, match='reasoning_effort must be one of "low", "high", "max"'):
            render_template(template, messages=_MESSAGES, reasoning_effort="none")

    @pytest.mark.parametrize("reasoning_effort", [ReasoningEffort.low, ReasoningEffort.high, ReasoningEffort.max])
    def test_exposed_values_encode_as_model_side_value(self, tmp_path: Path, reasoning_effort: ReasoningEffort) -> None:
        tekken_path = build_reasoning_effort_tekken_json(
            output_dir=tmp_path,
            reasoning_effort_mapping=LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        tokenizer = MistralTokenizer.from_file(str(tekken_path))
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=reasoning_effort)
        text = decode_keep(tokenizer, tokenizer.encode_chat_completion(request))
        model_side_value = LOW_HIGH_MAX_REASONING_EFFORT_MAPPING[reasoning_effort.value]
        assert _reasoning_effort_settings(model_side_value) in text

    def test_unset_effort_encodes_mapped_default(self, tmp_path: Path) -> None:
        # An unset effort resolves to the exposed default "high", remapped to "medium".
        tekken_path = build_reasoning_effort_tekken_json(
            output_dir=tmp_path,
            reasoning_effort_mapping=LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        tokenizer = MistralTokenizer.from_file(str(tekken_path))
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")])
        text = decode_keep(tokenizer, tokenizer.encode_chat_completion(request))
        assert _reasoning_effort_settings(LOW_HIGH_MAX_REASONING_EFFORT_MAPPED_DEFAULT) in text

    @pytest.mark.parametrize(
        "reasoning_effort",
        [ReasoningEffort.low, ReasoningEffort.high, ReasoningEffort.max, None],
        ids=["low", "high", "max", "unset"],
    )
    def test_template_and_tokenizer_agree(self, tmp_path: Path, reasoning_effort: ReasoningEffort | None) -> None:
        tekken_path = build_reasoning_effort_tekken_json(
            output_dir=tmp_path,
            reasoning_effort_mapping=LOW_HIGH_MAX_REASONING_EFFORT_MAPPING,
            reasoning_effort_default=LOW_HIGH_MAX_REASONING_EFFORT_DEFAULT,
        )
        tokenizer = MistralTokenizer.from_file(str(tekken_path))
        template = convert_tokenizer_to_chat_template(tokenizer_file=tekken_path)
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=reasoning_effort)

        tokenized = encode_mistral_common(mistral_tokenizer=tokenizer, chat_request=request, spm=False)
        rendered = render_template(template, **build_render_args(request))

        if reasoning_effort is None:
            assert _reasoning_effort_settings(LOW_HIGH_MAX_REASONING_EFFORT_MAPPED_DEFAULT) in tokenized
        else:
            model_side_value = LOW_HIGH_MAX_REASONING_EFFORT_MAPPING[reasoning_effort.value]
            assert _reasoning_effort_settings(model_side_value) in tokenized
        assert tokenized == rendered


class TestV15MultimodalContent:
    def test_v15_tool_message_with_image_content(self) -> None:
        r"""V15 image template renders tool message with image content using render_content."""
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=True,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none",
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Use tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "test12345", "function": {"name": "fn", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "result"},
                    {"type": "image_url", "image_url": "http://example.com/img.png"},
                ],
                "tool_call_id": "test12345",
            },
            {"role": "assistant", "content": "Done"},
        ]

        tools = [{"type": "function", "function": {"name": "fn", "description": "test", "parameters": {}}}]
        output = render_template(template, messages, tools=tools, reasoning_effort="none")
        assert "[TOOL_RESULTS]" in output
        assert "[IMG]" in output
        assert "result" in output

    def test_v15_tool_message_with_audio_content(self) -> None:
        r"""V15 audio template renders tool message with audio content."""
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=True,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none",
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Use tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "test12345", "function": {"name": "fn", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "result"},
                    {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
                ],
                "tool_call_id": "test12345",
            },
            {"role": "assistant", "content": "Done"},
        ]

        tools = [{"type": "function", "function": {"name": "fn", "description": "test", "parameters": {}}}]
        output = render_template(template, messages, tools=tools, reasoning_effort="none")
        assert "[TOOL_RESULTS]" in output
        assert "[AUDIO]" in output
        assert "result" in output

    def test_v15_system_message_with_audio_content(self) -> None:
        r"""V15 audio template renders system message with audio content."""
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=True,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none",
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Listen to context"},
                    {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
                ],
            },
            {"role": "user", "content": "Summarize"},
            {"role": "assistant", "content": "Done"},
        ]

        output = render_template(template, messages, reasoning_effort="none")
        assert "[SYSTEM_PROMPT]" in output
        assert "Listen to context" in output
        assert "[AUDIO]" in output

    def test_pre_v15_image_template_rejects_image_in_assistant(self) -> None:
        r"""Pre-V15 image template rejects image chunks in assistant message."""
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=True,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Show me"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here"},
                    {"type": "image_url", "image_url": "http://example.com/img.png"},
                ],
            },
        ]

        with pytest.raises(ValueError, match="Only text chunks are supported in assistant message contents"):
            render_template(template, messages)

    def test_pre_v15_audio_template_rejects_audio_in_assistant(self) -> None:
        r"""Pre-V15 audio template rejects audio chunks in assistant message."""
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=True,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Listen"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here"},
                    {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
                ],
            },
        ]

        with pytest.raises(ValueError, match="Only text chunks are supported in assistant message contents"):
            render_template(template, messages)

    def test_pre_v15_audio_template_rejects_audio_in_system(self) -> None:
        r"""Pre-V15 audio template rejects audio chunks in system message."""
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=True,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Context"},
                    {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
                ],
            },
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        with pytest.raises(ValueError, match="Only text chunks are supported in system message contents"):
            render_template(template, messages)

    def test_v15_image_think_template_rejects_think_in_tool(self) -> None:
        r"""V15 image+think template raises when a thinking chunk appears in tool message content."""
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=True,
            audio_support=False,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none",
        )

        tools = [{"type": "function", "function": {"name": "fn", "description": "test", "parameters": {}}}]

        messages_with_think: list[dict[str, Any]] = [
            {"role": "user", "content": "Use tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "test12345", "function": {"name": "fn", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "thinking", "thinking": "secret reasoning"},
                    {"type": "text", "text": "result"},
                ],
                "tool_call_id": "test12345",
            },
        ]

        with pytest.raises(ValueError, match="tool message contents") as excinfo:
            render_template(template, messages_with_think, tools=tools)
        assert "secret reasoning" not in str(excinfo.value)

        messages_text_only: list[dict[str, Any]] = [
            {"role": "user", "content": "Use tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "test12345", "function": {"name": "fn", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "result"},
                    {"type": "image_url", "image_url": "http://example.com/img.png"},
                ],
                "tool_call_id": "test12345",
            },
        ]

        output = render_template(template, messages_text_only, tools=tools)
        assert "result" in output

    def test_pre_v15_image_template_rejects_image_in_tool(self) -> None:
        r"""Pre-V15 image template rejects image chunks in tool message content."""
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=True,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Use tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "test12345", "function": {"name": "fn", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "text", "text": "result"},
                    {"type": "image_url", "image_url": "http://example.com/img.png"},
                ],
                "tool_call_id": "test12345",
            },
            {"role": "assistant", "content": "Done"},
        ]

        tools = [{"type": "function", "function": {"name": "fn", "description": "test", "parameters": {}}}]
        # Pre-V15 uses message['content']|string which coerces list to string representation
        # rather than rendering through render_content — no [IMG] token produced
        output = render_template(template, messages, tools=tools)
        assert "[IMG]" not in output

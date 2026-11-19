import json
import re
from typing import Any

import pytest
from jinja2 import BaseLoader
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mistral_common.integrations.chat_templates.template_generator import TemplateConfig, build_chat_template
from mistral_common.protocol.instruct.chunk import AudioURLChunk, ImageURLChunk, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ReasoningFieldFormat,
    SystemMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import (
    JsonSchema,
    ReasoningEffort,
    ResponseFormat,
    ResponseFormats,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.fixtures.audio import get_dummy_audio_chunk
from tests.integration.chat_recipes import (
    build_conversation_request,
    build_function_tool,
    build_system_content_request,
    build_thinking_assistant_request,
    build_tool_call_request,
    build_user_assistant_request,
)
from tests.utils.chat_templates.configs import TestConfig
from tests.utils.chat_templates.helpers import (
    PARITY_JSON_SCHEMA,
    PARITY_JSON_SCHEMA_NON_ASCII,
    build_render_args,
    build_test_template,
    render_template,
)

_V15_WEATHER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"location": {"type": "string", "description": "The city and state"}},
    "required": ["location"],
}


def _json_schema_response_format(schema: dict[str, Any]) -> ResponseFormat:
    return ResponseFormat(type=ResponseFormats.json_schema, json_schema=JsonSchema(name="answer", schema=schema))


class TestV15ModelSettings:
    @pytest.fixture()
    def v15_template(self) -> str:
        return build_test_template(TestConfig(version=TokenizerVersion.v15))

    def test_v15_template_contains_model_settings(self, v15_template: str) -> None:
        assert "[MODEL_SETTINGS]" in v15_template
        assert "reasoning_effort" in v15_template

    @pytest.mark.parametrize(
        ("reasoning_effort", "expected_value"),
        [(None, "none"), ("high", "high"), ("none", "none")],
    )
    def test_v15_reasoning_effort_rendering(
        self, v15_template: str, reasoning_effort: str | None, expected_value: str
    ) -> None:
        request = build_user_assistant_request(
            user_content="Hello",
            assistant_content="Hi there!",
            reasoning_effort=None if reasoning_effort is None else ReasoningEffort(reasoning_effort),
        )

        output = render_template(v15_template, **build_render_args(request))
        assert f'[MODEL_SETTINGS]{{"reasoning_effort": "{expected_value}"}}[/MODEL_SETTINGS]' in output

    def test_v15_with_image_support(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v15, image=True))
        request = build_user_assistant_request(
            user_content=[TextChunk(text="What is this?"), ImageURLChunk(image_url="http://example.com/image.png")],
            assistant_content="It's an image.",
            reasoning_effort=ReasoningEffort.high,
        )

        output = render_template(template, **build_render_args(request))
        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output
        assert "[IMG]" in output

    def test_v15_with_thinking_support(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v15, think=True))
        request = build_thinking_assistant_request(
            user_content="Solve this problem",
            thinking="Let me think...",
            text="The answer is 42.",
            reasoning_effort=ReasoningEffort.none,
        )

        output = render_template(
            template, **build_render_args(request, reasoning_field_format=ReasoningFieldFormat.thinking_chunks)
        )
        assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output
        assert "[THINK]Let me think...[/THINK]" in output

    def test_v15_with_tools(self) -> None:
        template = build_test_template(TestConfig(version=TokenizerVersion.v15))
        request = build_conversation_request(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="What's the weather in Paris?"),
            ],
            tools=[
                build_function_tool(
                    name="get_weather", description="Get the current weather", parameters=_V15_WEATHER_PARAMETERS
                )
            ],
            reasoning_effort=ReasoningEffort.high,
        )

        output = render_template(template, **build_render_args(request))
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
        template = build_test_template(TestConfig(version=TokenizerVersion.v15))
        messages: list[Any] = []
        if has_system:
            messages.append(SystemMessage(content="You are helpful."))
        messages.extend(
            [
                UserMessage(content="Hello"),
                AssistantMessage(content="Hi"),
            ]
        )
        request = build_conversation_request(
            messages=messages,
            tools=(
                [build_function_tool(name="tool1", description="", parameters={"type": "object", "properties": {}})]
                if has_tools
                else None
            ),
            reasoning_effort=None if reasoning_effort is None else ReasoningEffort(reasoning_effort),
        )

        output = render_template(template, **build_render_args(request))

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
        template = build_test_template(TestConfig(version=TokenizerVersion.v15, image=image, think=True))
        request = build_system_content_request(
            system_content=[TextChunk(text="System text."), ThinkChunk(thinking="System thinking.")]
        )

        with pytest.raises(ValueError, match="Only text chunks are supported in system message contents"):
            render_template(template, **build_render_args(request))

    def test_v15_json_schema_only_renders_custom_schema(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        request = build_user_assistant_request(
            user_content="Hello",
            assistant_content="Hi",
            response_format=_json_schema_response_format(PARITY_JSON_SCHEMA),
        )

        output = render_template(template, **build_render_args(request))

        expected_settings = f'[MODEL_SETTINGS]{{"json_schema": {json.dumps(obj=PARITY_JSON_SCHEMA)}}}[/MODEL_SETTINGS]'
        assert expected_settings in output

    def test_v15_json_schema_non_ascii_html_specials_full_parity(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"json_schema"}),
            )
        )
        request = build_user_assistant_request(
            user_content="Hello",
            assistant_content="Hi",
            response_format=_json_schema_response_format(PARITY_JSON_SCHEMA_NON_ASCII),
        )

        output = render_template(template, **build_render_args(request))

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
        request = build_user_assistant_request(
            user_content="Hello",
            assistant_content="Hi",
            response_format=ResponseFormat(type=ResponseFormats.json_object),
        )

        output = render_template(template, **build_render_args(request))

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
        request = build_user_assistant_request(
            user_content="Hello",
            assistant_content="Hi",
            response_format=ResponseFormat(type=ResponseFormats.text),
        )

        output = render_template(template, **build_render_args(request))

        assert "[MODEL_SETTINGS]" not in output

    def test_v15_unknown_response_format_type_raises(self) -> None:
        # An unknown response-format type is rejected by the request model at
        # construction, so the template claim renders the raw OpenAI dict.
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
        request = build_user_assistant_request(
            user_content="Hello",
            assistant_content="Hi",
            response_format=ResponseFormat(
                type=ResponseFormats.json_object,
                json_schema=JsonSchema(name="answer", schema={"type": "string"}),
            ),
        )

        output = render_template(template, **build_render_args(request))

        assert '[MODEL_SETTINGS]{"json_schema": {"anyOf": [{"type": "object"}, {"type": "array"}]}}' in output

    def test_v15_response_format_without_type_defaults_to_text(self) -> None:
        # A response format without a type is not constructible as a
        # ResponseFormat (type defaults to text there), so render the raw dict.
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
        request = build_user_assistant_request(user_content="Hello", assistant_content="Hi")

        output = render_template(template, **build_render_args(request))

        assert "[MODEL_SETTINGS]" not in output

    @pytest.mark.parametrize(
        "json_schema",
        [{"name": "answer"}, {"name": "answer", "schema": None}, None],
        ids=["missing_schema", "null_schema", "null_json_schema"],
    )
    def test_v15_json_schema_missing_schema_raises(self, json_schema: dict[str, Any] | None) -> None:
        # JsonSchema requires a schema at construction, so the template-level
        # rejection renders the raw OpenAI dict directly.
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
        has_reasoning_effort = "reasoning_effort" in model_settings_fields
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=model_settings_fields,
                reasoning_effort_values=("none", "high") if has_reasoning_effort else None,
                reasoning_effort_default="none" if has_reasoning_effort else None,
            )
        )
        request = build_conversation_request(
            messages=[UserMessage(content="Hello")],
            response_format=ResponseFormat(type=ResponseFormats.json_object),
        )
        environment = ImmutableSandboxedEnvironment(loader=BaseLoader())
        render_kwargs = build_render_args(request)

        output = environment.from_string(template).render(
            bos_token="<s>",
            eos_token="</s>",
            **render_kwargs,
        )

        assert '[MODEL_SETTINGS]{"json_schema": {"anyOf": ' in output

    def test_v15_both_fields_sorted_keys(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
                reasoning_effort_values=("none", "high"),
                reasoning_effort_default="none",
            )
        )
        request = build_user_assistant_request(
            user_content="Hello",
            assistant_content="Hi",
            reasoning_effort=ReasoningEffort.high,
            response_format=_json_schema_response_format(PARITY_JSON_SCHEMA),
        )

        output = render_template(template, **build_render_args(request))

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
                reasoning_effort_values=("none", "high"),
                reasoning_effort_default="none",
            )
        )
        request = build_user_assistant_request(
            user_content="Hello",
            assistant_content="Hi",
            response_format=ResponseFormat(type=ResponseFormats.json_object),
        )

        output = render_template(template, **build_render_args(request))

        assert (
            '[MODEL_SETTINGS]{"json_schema": {"anyOf": [{"type": "object"}, '
            '{"type": "array"}]}, "reasoning_effort": "none"}[/MODEL_SETTINGS]' in output
        )

    def test_v15_both_fields_text_reasoning_effort_only(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
                reasoning_effort_values=("none", "high"),
                reasoning_effort_default="none",
            )
        )
        request = build_user_assistant_request(
            user_content="Hello",
            assistant_content="Hi",
            reasoning_effort=ReasoningEffort.high,
            response_format=ResponseFormat(type=ResponseFormats.text),
        )

        output = render_template(template, **build_render_args(request))

        assert '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]' in output
        assert '"json_schema"' not in output

    def test_v15_both_fields_invalid_reasoning_effort_raises(self) -> None:
        # "bogus" is not a ReasoningEffort, so the request model rejects it at
        # construction; the template claim renders the raw value directly.
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
                reasoning_effort_values=("none", "high"),
                reasoning_effort_default="none",
            )
        )
        request = build_user_assistant_request(user_content="Hello", assistant_content="Hi")

        with pytest.raises(ValueError, match='reasoning_effort must be either "none" or "high"'):
            render_template(template, **build_render_args(request), reasoning_effort="bogus")

    def test_v15_reasoning_effort_only_ignores_response_format(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                reasoning_effort_values=("none", "high"),
                reasoning_effort_default="none",
            )
        )
        request = build_user_assistant_request(
            user_content="Hello",
            assistant_content="Hi",
            response_format=_json_schema_response_format(PARITY_JSON_SCHEMA),
        )

        output = render_template(template, **build_render_args(request))

        assert '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]' in output
        assert '"json_schema"' not in output

    def test_v15_both_fields_with_tools_placement(self) -> None:
        template = build_chat_template(
            config=TemplateConfig(
                version=TokenizerVersion.v15,
                model_settings_fields=frozenset({"reasoning_effort", "json_schema"}),
                reasoning_effort_values=("none", "high"),
                reasoning_effort_default="none",
            )
        )
        request = build_conversation_request(
            messages=[UserMessage(content="Hello"), AssistantMessage(content="Hi")],
            tools=[
                build_function_tool(
                    name="lookup", description="Find a value", parameters={"type": "object", "properties": {}}
                )
            ],
            reasoning_effort=ReasoningEffort.high,
            response_format=_json_schema_response_format(PARITY_JSON_SCHEMA),
        )

        output = render_template(template, **build_render_args(request))

        tools_position = output.index("[AVAILABLE_TOOLS]")
        settings_position = output.index("[MODEL_SETTINGS]")
        first_user_position = output.index("[INST]")
        assert tools_position < settings_position < first_user_position


class TestV15MultimodalContent:
    def test_v15_tool_message_with_image_content(self) -> None:
        r"""V15 image template renders tool message with image content using render_content."""
        template = build_test_template(TestConfig(version=TokenizerVersion.v15, image=True))
        request = build_tool_call_request(
            tool_name="fn",
            call_id="test12345",
            arguments="{}",
            result_content=[TextChunk(text="result"), ImageURLChunk(image_url="http://example.com/img.png")],
            assistant_content="",
            terminal_content="Done",
            tools=[build_function_tool(name="fn", description="test", parameters={})],
        )

        output = render_template(template, **build_render_args(request))
        assert "[TOOL_RESULTS]" in output
        assert "[IMG]" in output
        assert "result" in output

    def test_v15_tool_message_with_audio_content(self) -> None:
        r"""V15 audio template renders tool message with audio content."""
        template = build_test_template(TestConfig(version=TokenizerVersion.v15, audio=True))
        request = build_tool_call_request(
            tool_name="fn",
            call_id="test12345",
            arguments="{}",
            result_content=[
                TextChunk(text="result"),
                AudioURLChunk(audio_url="data:audio/wav;base64,abc"),
            ],
            assistant_content="",
            terminal_content="Done",
            tools=[build_function_tool(name="fn", description="test", parameters={})],
        )

        output = render_template(template, **build_render_args(request))
        assert "[TOOL_RESULTS]" in output
        assert "[AUDIO]" in output
        assert "result" in output

    def test_v15_system_message_with_audio_content(self) -> None:
        r"""V15 audio template renders system message with audio content."""
        template = build_test_template(TestConfig(version=TokenizerVersion.v15, audio=True))
        request = build_system_content_request(
            system_content=[TextChunk(text="Listen to context"), get_dummy_audio_chunk()],
            user_content="Summarize",
            assistant_content="Done",
        )

        output = render_template(template, **build_render_args(request))
        assert "[SYSTEM_PROMPT]" in output
        assert "Listen to context" in output
        assert "[AUDIO]" in output

    def test_pre_v15_image_template_rejects_image_in_assistant(self) -> None:
        r"""Pre-V15 image template rejects image chunks in assistant message."""
        # AssistantMessage only accepts text/think chunks, so the invalid chunk
        # type renders through the raw OpenAI dict.
        template = build_test_template(TestConfig(version=TokenizerVersion.v13, image=True))
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
        # AssistantMessage only accepts text/think chunks, so the invalid chunk
        # type renders through the raw OpenAI dict.
        template = build_test_template(TestConfig(version=TokenizerVersion.v13, audio=True))
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
        template = build_test_template(TestConfig(version=TokenizerVersion.v13, audio=True))
        request = build_system_content_request(system_content=[TextChunk(text="Context"), get_dummy_audio_chunk()])

        with pytest.raises(ValueError, match="Only text chunks are supported in system message contents"):
            render_template(template, **build_render_args(request))

    def test_v15_image_think_template_rejects_think_in_tool(self) -> None:
        r"""V15 image+think template raises when a thinking chunk appears in tool message content."""
        # ToolMessage does not accept ThinkChunks, so the invalid chunk type
        # renders through the raw OpenAI dict.
        template = build_test_template(TestConfig(version=TokenizerVersion.v15, image=True, think=True))
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

        request = build_tool_call_request(
            tool_name="fn",
            call_id="test12345",
            arguments="{}",
            result_content=[TextChunk(text="result"), ImageURLChunk(image_url="http://example.com/img.png")],
            assistant_content="",
            tools=[build_function_tool(name="fn", description="test", parameters={})],
        )

        output = render_template(template, **build_render_args(request))
        assert "result" in output

    def test_pre_v15_image_template_rejects_image_in_tool(self) -> None:
        r"""Pre-V15 image template rejects image chunks in tool message content."""
        template = build_test_template(TestConfig(version=TokenizerVersion.v13, image=True))
        request = build_tool_call_request(
            tool_name="fn",
            call_id="test12345",
            arguments="{}",
            result_content=[TextChunk(text="result"), ImageURLChunk(image_url="http://example.com/img.png")],
            assistant_content="",
            terminal_content="Done",
            tools=[build_function_tool(name="fn", description="test", parameters={})],
        )
        # Pre-V15 uses message['content']|string which coerces list to string representation
        # rather than rendering through render_content — no [IMG] token produced
        output = render_template(template, **build_render_args(request))
        assert "[IMG]" not in output

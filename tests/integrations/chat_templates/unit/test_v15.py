from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import TestConfig, _load_golden_template, _make_config, render_template


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
        )

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
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        output = render_template(v15_template, messages, reasoning_effort=reasoning_effort)
        assert f'[MODEL_SETTINGS]{{"reasoning_effort": "{expected_value}"}}[/MODEL_SETTINGS]' in output

    def test_v15_static_dynamic_parity(self, v15_template: str) -> None:
        config = _make_config(TestConfig(version=TokenizerVersion.v15))
        static_template = _load_golden_template(config)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        static_output = render_template(static_template, messages, reasoning_effort="high")
        dynamic_output = render_template(v15_template, messages, reasoning_effort="high")

        assert static_output == dynamic_output

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

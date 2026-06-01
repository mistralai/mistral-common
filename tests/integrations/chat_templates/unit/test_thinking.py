from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.integrations.chat_templates.template_generator import TemplateConfig, build_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import TestConfig, _load_golden_template, _make_config, render_template

THINK_CONFIGS = [
    (TokenizerVersion.v13, False, False),
    (TokenizerVersion.v13, True, False),
    (TokenizerVersion.v15, False, False),
    (TokenizerVersion.v15, True, False),
]


class TestReasoningContentConversion:
    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_content_to_thinking_chunk(self, version: TokenizerVersion, image: bool, audio: bool) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "reasoning_content": "Let me add 2 and 2.", "content": "The answer is 4."},
        ]

        output = render_template(template, messages)
        assert "[THINK]Let me add 2 and 2.[/THINK]The answer is 4." in output

    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_field_to_thinking_chunk(self, version: TokenizerVersion, image: bool, audio: bool) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "reasoning": "Adding three and three.", "content": "The answer is 6."},
        ]

        output = render_template(template, messages)
        assert "[THINK]Adding three and three.[/THINK]The answer is 6." in output

    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_content_takes_precedence_over_reasoning(
        self, version: TokenizerVersion, image: bool, audio: bool
    ) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "reasoning_content": "RC wins",
                "reasoning": "R loses",
                "content": "Hello!",
            },
        ]

        output = render_template(template, messages)
        assert "[THINK]RC wins[/THINK]Hello!" in output
        assert "R loses" not in output

    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_content_only_no_text_content(self, version: TokenizerVersion, image: bool, audio: bool) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "reasoning_content": "Just thinking...", "content": ""},
        ]

        output = render_template(template, messages)
        assert "[THINK]Just thinking...[/THINK]" in output

    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_content_with_tool_calls(self, version: TokenizerVersion, image: bool, audio: bool) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "reasoning_content": "I should call a tool.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    },
                ],
            },
            {"role": "tool", "name": "get_weather", "content": "22C", "tool_call_id": "abc123"},
            {"role": "assistant", "content": "It's 22C in Paris."},
        ]

        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]

        output = render_template(template, messages, tools=tools)
        assert "[THINK]I should call a tool.[/THINK]" in output
        assert "get_weather" in output

    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_aggregation_consecutive_assistants(
        self, version: TokenizerVersion, image: bool, audio: bool
    ) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Solve this."},
            {"role": "assistant", "reasoning_content": "Step 1 reasoning", "content": "Partial answer."},
            {"role": "assistant", "reasoning_content": "Step 2 reasoning", "content": "Final answer."},
        ]

        output = render_template(template, messages)
        # Both reasoning traces should appear as thinking chunks
        assert "[THINK]Step 1 reasoning[/THINK]" in output
        assert "[THINK]Step 2 reasoning[/THINK]" in output
        # Text from both messages should be aggregated
        assert "Partial answer." in output
        assert "Final answer." in output

    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_static_dynamic_parity(self, version: TokenizerVersion, image: bool, audio: bool) -> None:
        config = _make_config(TestConfig(version=version, image=image, audio=audio, think=True))
        static_template = _load_golden_template(config)
        dynamic_template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "reasoning_content": "Thinking...", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "reasoning": "Considering the question.", "content": "I'm well!"},
        ]

        static_output = render_template(static_template, messages)
        dynamic_output = render_template(dynamic_template, messages)

        assert static_output == dynamic_output

    def test_reasoning_content_with_none_content(self) -> None:
        config = TemplateConfig(version=TokenizerVersion.v15, thinking_support=True)
        template = build_chat_template(config)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Let me think about this.",
            },
        ]
        result = render_template(template, messages=messages)
        assert "[THINK]" in result
        assert "Let me think about this." in result

    @pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
    @pytest.mark.parametrize(("version", "image", "audio"), THINK_CONFIGS)
    def test_reasoning_with_existing_think_chunks_raises(
        self, version: TokenizerVersion, image: bool, audio: bool, field: str
    ) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        reasoning_text = (
            "Top-level reasoning" if field == "reasoning_content" else "Top-level reasoning via reasoning field"
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                field: reasoning_text,
                "content": [
                    {"type": "thinking", "thinking": "Inline thinking"},
                    {"type": "text", "text": "Hello!"},
                ],
            },
        ]

        with pytest.raises(
            ValueError,
            match="Message cannot have both thinking chunks in content and a top-level"
            " `reasoning` or `reasoning_content` field",
        ):
            render_template(template, messages)


class TestReasoningEdgeCases:
    def test_non_think_template_ignores_reasoning_field(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "reasoning_content": "This is ignored", "content": "Hello!"},
        ]

        output = render_template(template, messages)
        assert "[THINK]" not in output
        assert "Hello!" in output

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_or_none_reasoning_ignored(self, value: str | None) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=False,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "reasoning_content": value, "content": "Hello!"},
        ]

        output = render_template(template, messages)
        assert "[THINK]" not in output
        assert "Hello!" in output


class TestClosedAttribute:
    @pytest.mark.parametrize(
        ("version", "image"),
        [
            (TokenizerVersion.v13, False),
            (TokenizerVersion.v13, True),
            (TokenizerVersion.v15, False),
            (TokenizerVersion.v15, True),
        ],
    )
    def test_closed_true_emits_close_tag(self, version: TokenizerVersion, image: bool) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=False,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Done thinking"},
                    {"type": "text", "text": "Hello!"},
                ],
            },
        ]

        output = render_template(template, messages)
        assert "[THINK]Done thinking[/THINK]Hello!" in output

    @pytest.mark.parametrize(
        ("version", "image"),
        [
            (TokenizerVersion.v13, False),
            (TokenizerVersion.v13, True),
            (TokenizerVersion.v15, False),
            (TokenizerVersion.v15, True),
        ],
    )
    def test_closed_false_omits_close_tag(self, version: TokenizerVersion, image: bool) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=image,
            audio_support=False,
            thinking_support=True,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Ongoing thought", "closed": False},
                ],
            },
        ]

        output = render_template(template, messages)
        assert "[THINK]Ongoing thought" in output
        assert "[/THINK]" not in output


class TestPlainThink:
    def test_plain_think_template_produces_correct_output(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Solve this"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "The answer is 42."},
                ],
            },
        ]

        output = render_template(template, messages)
        assert "<think>Let me think...</think>The answer is 42." in output
        assert "[THINK]" not in output
        assert "[/THINK]" not in output

    @pytest.mark.parametrize("image", [False, True])
    def test_plain_think_static_dynamic_parity(self, image: bool) -> None:
        config = _make_config(TestConfig(version=TokenizerVersion.v11, image=image, plain_think=True))
        static_template = _load_golden_template(config)
        dynamic_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=image,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Thinking..."},
                    {"type": "text", "text": "Hi!"},
                ],
            },
        ]

        static_output = render_template(static_template, messages)
        dynamic_output = render_template(dynamic_template, messages)
        assert static_output == dynamic_output

    def test_plain_think_reasoning_content_conversion(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "reasoning_content": "Let me add.", "content": "4."},
        ]

        output = render_template(template, messages)
        assert "<think>Let me add.</think>4." in output
        assert "[THINK]" not in output

    def test_plain_think_closed_false(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Still thinking...", "closed": False},
                ],
            },
        ]

        output = render_template(template, messages)
        assert "<think>Still thinking..." in output
        assert "</think>" not in output

    def test_plain_think_image_template(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=True,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": "http://example.com/img.png"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "It looks like..."},
                    {"type": "text", "text": "A red square."},
                ],
            },
        ]

        output = render_template(template, messages)
        assert "[IMG]" in output
        assert "<think>It looks like...</think>A red square." in output
        assert "[THINK]" not in output

    def test_plain_think_reasoning_with_existing_think_chunks_raises(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v11,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=True,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "reasoning_content": "Top-level reasoning",
                "content": [
                    {"type": "thinking", "thinking": "Inline thinking"},
                    {"type": "text", "text": "Hello!"},
                ],
            },
        ]

        with pytest.raises(
            ValueError,
            match="Message cannot have both thinking chunks in content and a top-level"
            " `reasoning` or `reasoning_content` field",
        ):
            render_template(template, messages)

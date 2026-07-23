from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.integrations.chat_templates.template_generator import (
    TemplateConfig,
    _render_content_call,
    build_chat_template,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import TestConfig, _load_golden_template, _make_config, render_template


class TestGenerateChatTemplateAPI:
    @pytest.fixture()
    def v13_template(self) -> str:
        return generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

    def test_basic_generation(self, v13_template: str) -> None:
        assert "{%- set default_system_message = '' %}" in v13_template
        assert "bos_token" in v13_template

    def test_default_system_prompt(self) -> None:
        template_with_prompt = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt="You are a helpful assistant.",
            plain_thinking_support=False,
            use_special_token_variables=True,
        )
        assert "You are a helpful assistant." in template_with_prompt

    def test_basic_static_dynamic_parity(self, v13_template: str) -> None:
        config = _make_config(TestConfig(version=TokenizerVersion.v13))
        static_template = _load_golden_template(config)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        static_output = render_template(static_template, messages)
        dynamic_output = render_template(v13_template, messages)

        assert static_output == dynamic_output

    def test_empty_messages_raises(self) -> None:
        config = TemplateConfig(version=TokenizerVersion.v15)
        template = build_chat_template(config)
        with pytest.raises(Exception, match="list object has no element 0"):
            render_template(template, messages=[])

    def test_double_quotes_in_default_system_prompt(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt='You are "the best" assistant.',
            plain_thinking_support=False,
            use_special_token_variables=True,
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]
        result = render_template(template, messages=messages)
        assert 'You are "the best" assistant.' in result


class TestTokenVariables:
    @pytest.mark.parametrize(
        ("use_vars", "present", "absent"),
        [
            (True, ["bos_token", "eos_token"], ["{{- '<s>' }}", "{{- '</s>' }}"]),
            (False, ["'<s>'", "'</s>'"], ["bos_token", "eos_token"]),
        ],
    )
    def test_token_variable_content(self, use_vars: bool, present: list[str], absent: list[str]) -> None:
        config = TemplateConfig(version=TokenizerVersion.v7, use_special_token_variables=use_vars)
        template = build_chat_template(config)
        for expected in present:
            assert expected in template
        for unexpected in absent:
            assert unexpected not in template

    def test_use_special_token_variables_false_renders_correctly(self) -> None:
        config = TemplateConfig(version=TokenizerVersion.v15, use_special_token_variables=False)
        template = build_chat_template(config)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = render_template(template, messages=messages)
        assert "<s>" in result
        assert "</s>" in result

    def test_use_special_token_variables_default_is_false(self) -> None:
        config = TemplateConfig(version=TokenizerVersion.v7)
        assert config.use_special_token_variables is False


class TestMessageContentValidation:
    @pytest.mark.parametrize("content", [[], None])
    @pytest.mark.parametrize("version", [TokenizerVersion.v3, TokenizerVersion.v7, TokenizerVersion.v15])
    def test_user_empty_content_renders_empty_turn(self, version: TokenizerVersion, content: list[Any] | None) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": content},
        ]

        output = render_template(template, messages)
        assert "[INST]" in output
        assert "[/INST]" in output

    @pytest.mark.parametrize("version", [TokenizerVersion.v7, TokenizerVersion.v13, TokenizerVersion.v15])
    def test_assistant_empty_content_no_tool_calls_raises(self, version: TokenizerVersion) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
        ]

        with pytest.raises(
            ValueError,
            match="Assistant message must have a string or a list of chunks in content or a list of tool calls",
        ):
            render_template(template, messages)

    def test_v1_empty_assistant_content_raises(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v1,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": []},
        ]

        with pytest.raises(ValueError, match="Assistant message content must be non-empty"):
            render_template(template, messages)

    def test_v1_invalid_assistant_chunk_type_raises(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v1,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [{"type": "image", "url": "http://example.com/img.png"}]},
        ]

        with pytest.raises(ValueError, match="Only text chunks are supported in assistant message contents"):
            render_template(template, messages)


class TestRenderContentCall:
    r"""Unit tests for `_render_content_call` helper."""

    def test_minimal_text_only_config(self) -> None:
        """Text-only config emits only content and context_name."""
        config = TemplateConfig(version=TokenizerVersion.v7)
        result = _render_content_call(
            config=config,
            content_expr="message['content']",
            context_name="user message content",
            supported_types_desc="text",
            support_thinking=False,
            support_images=False,
            support_audio=False,
        )
        assert result == "render_content(content=message['content'], context_name='user message content')"

    @pytest.mark.parametrize(
        ("config", "call_kwargs", "present", "absent"),
        [
            pytest.param(
                TemplateConfig(version=TokenizerVersion.v13, thinking_support=True),
                {
                    "content_expr": "message['content']",
                    "context_name": "assistant message contents",
                    "supported_types_desc": "text and thinking",
                    "support_thinking": True,
                    "support_images": False,
                    "support_audio": False,
                },
                [
                    "content=message['content']",
                    "context_name='assistant message contents'",
                    "supported_types_desc='text and thinking'",
                    "support_thinking=true",
                ],
                ["support_images", "support_audio"],
                id="thinking_assistant",
            ),
            pytest.param(
                TemplateConfig(version=TokenizerVersion.v7, image_support=True),
                {
                    "content_expr": "user_content",
                    "context_name": "user message content",
                    "supported_types_desc": "text, image and image_url",
                    "support_thinking": False,
                    "support_images": True,
                    "support_audio": False,
                },
                ["support_images=true", "supported_types_desc='text, image and image_url'"],
                ["support_thinking", "support_audio"],
                id="image_user",
            ),
            pytest.param(
                TemplateConfig(version=TokenizerVersion.v7, audio_support=True),
                {
                    "content_expr": "message['content']",
                    "context_name": "user message content",
                    "supported_types_desc": "text, input_audio and audio_url",
                    "support_thinking": False,
                    "support_images": False,
                    "support_audio": True,
                },
                ["support_audio=true", "supported_types_desc='text, input_audio and audio_url'"],
                ["support_thinking", "support_images"],
                id="audio_user",
            ),
        ],
    )
    def test_feature_config_emits_correct_args(
        self,
        config: TemplateConfig,
        call_kwargs: dict,
        present: list[str],
        absent: list[str],
    ) -> None:
        """Feature-specific config emits the expected args and omits absent ones."""
        result = _render_content_call(config=config, **call_kwargs)
        for substring in present:
            assert substring in result
        for substring in absent:
            assert substring not in result

    def test_exact_thinking_image_assistant(self) -> None:
        """Full exact-string check for thinking+image config at assistant site."""
        config = TemplateConfig(version=TokenizerVersion.v13, thinking_support=True, image_support=True)
        result = _render_content_call(
            config=config,
            content_expr="message['content']",
            context_name="assistant message contents",
            supported_types_desc="text and thinking",
            support_thinking=True,
            support_images=False,
            support_audio=False,
        )
        assert result == (
            "render_content("
            "content=message['content'], "
            "context_name='assistant message contents', "
            "supported_types_desc='text and thinking', "
            "support_thinking=true, "
            "support_images=false)"
        )

    def test_exact_spm_image_user(self) -> None:
        """Full exact-string check for SPM+image config at user site."""
        config = TemplateConfig(version=TokenizerVersion.v3, spm=True, image_support=True)
        result = _render_content_call(
            config=config,
            content_expr="user_content",
            context_name="user message content",
            supported_types_desc="text, image and image_url",
            support_thinking=False,
            support_images=True,
            support_audio=False,
            initial_prev_img="not added_sp",
        )
        assert result == (
            "render_content("
            "content=user_content, "
            "context_name='user message content', "
            "supported_types_desc='text, image and image_url', "
            "support_images=true, "
            "initial_prev_img=not added_sp)"
        )

    def test_tool_ban_thinking_not_in_desc(self) -> None:
        """Tool call site with thinking config must have support_thinking=false and no 'thinking' in desc."""
        config = TemplateConfig(version=TokenizerVersion.v15, thinking_support=True)
        result = _render_content_call(
            config=config,
            content_expr="message['content']",
            context_name="tool message contents",
            supported_types_desc="text",
            support_thinking=False,
            support_images=False,
            support_audio=False,
        )
        assert "support_thinking=false" in result
        # The supported_types_desc must not mention thinking
        desc_value = result.split("supported_types_desc='")[1].split("'")[0]
        assert "thinking" not in desc_value

    def test_arg_order_matches_macro_declaration(self) -> None:
        """Args follow macro-declared order: content, context_name, supported_types_desc, support_thinking, support_images."""  # noqa: E501
        config = TemplateConfig(version=TokenizerVersion.v13, thinking_support=True, image_support=True)
        result = _render_content_call(
            config=config,
            content_expr="message['content']",
            context_name="assistant message contents",
            supported_types_desc="text and thinking",
            support_thinking=True,
            support_images=False,
            support_audio=False,
        )
        idx_content = result.index("content=")
        idx_context = result.index("context_name=")
        idx_desc = result.index("supported_types_desc=")
        idx_thinking = result.index("support_thinking=")
        idx_images = result.index("support_images=")
        assert idx_content < idx_context < idx_desc < idx_thinking < idx_images

    def test_image_config_emits_support_images(self) -> None:
        """Image config emits support_images, no support_thinking or support_audio."""
        config = TemplateConfig(version=TokenizerVersion.v7, image_support=True)
        result = _render_content_call(
            config=config,
            content_expr="user_content",
            context_name="user message content",
            supported_types_desc="text, image and image_url",
            support_thinking=False,
            support_images=True,
            support_audio=False,
        )
        assert "support_images=true" in result
        assert "support_thinking" not in result
        assert "support_audio" not in result

    def test_spm_image_emits_initial_prev_img(self) -> None:
        """SPM+image config emits initial_prev_img."""
        config = TemplateConfig(version=TokenizerVersion.v3, spm=True, image_support=True)
        result = _render_content_call(
            config=config,
            content_expr="user_content",
            context_name="user message content",
            supported_types_desc="text, image and image_url",
            support_thinking=False,
            support_images=True,
            support_audio=False,
            initial_prev_img="not added_sp",
        )
        assert "initial_prev_img=not added_sp" in result

    def test_non_spm_does_not_emit_initial_prev_img(self) -> None:
        """Non-SPM image config does not emit initial_prev_img."""
        config = TemplateConfig(version=TokenizerVersion.v7, image_support=True)
        result = _render_content_call(
            config=config,
            content_expr="user_content",
            context_name="user message content",
            supported_types_desc="text, image and image_url",
            support_thinking=False,
            support_images=True,
            support_audio=False,
        )
        assert "initial_prev_img" not in result

    def test_audio_config_emits_support_audio(self) -> None:
        """Audio config emits support_audio and supported_types_desc, no image or thinking."""
        config = TemplateConfig(version=TokenizerVersion.v7, audio_support=True)
        result = _render_content_call(
            config=config,
            content_expr="message['content']",
            context_name="user message content",
            supported_types_desc="text, input_audio and audio_url",
            support_thinking=False,
            support_images=False,
            support_audio=True,
        )
        assert "support_audio=true" in result
        assert "supported_types_desc='text, input_audio and audio_url'" in result
        assert "support_thinking" not in result
        assert "support_images" not in result

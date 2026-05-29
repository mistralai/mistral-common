from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.integrations.chat_templates.template_generator import TemplateConfig, build_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import _load_golden_template, _make_config, render_template


class TestGenerateChatTemplateAPI:
    @pytest.fixture()
    def v13_template(self) -> str:
        return generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v13,
            image_support=False,
            audio_support=False,
            thinking_support=False,
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
        )
        assert "You are a helpful assistant." in template_with_prompt

    def test_basic_static_dynamic_parity(self, v13_template: str) -> None:
        config = _make_config((TokenizerVersion.v13, False, False, False, False, False))
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
        config = TemplateConfig(version=TokenizerVersion.v7, use_token_variables=use_vars)
        template = build_chat_template(config)
        for expected in present:
            assert expected in template
        for unexpected in absent:
            assert unexpected not in template

    def test_use_token_variables_false_renders_correctly(self) -> None:
        config = TemplateConfig(version=TokenizerVersion.v15, use_token_variables=False)
        template = build_chat_template(config)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = render_template(template, messages=messages)
        assert "<s>" in result
        assert "</s>" in result

    def test_use_token_variables_default_is_true(self) -> None:
        config = TemplateConfig(version=TokenizerVersion.v7)
        assert config.use_token_variables is True


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
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [{"type": "image", "url": "http://example.com/img.png"}]},
        ]

        with pytest.raises(ValueError, match="Only text chunks are supported in assistant message contents"):
            render_template(template, messages)

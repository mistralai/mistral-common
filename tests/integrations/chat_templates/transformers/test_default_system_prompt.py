import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import encode_transformers_from_openai

CONV_NO_SYSTEM: dict[str, list[dict[str, str]]] = {
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
}


class TestDefaultSystemPromptInjection:
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

    CONV_WITH_SYSTEM: dict[str, list[dict[str, str]]] = {
        "messages": [
            {"role": "system", "content": "You are a custom assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }

    @pytest.mark.parametrize(
        ("version", "spm", "expected_fragment"),
        [
            (TokenizerVersion.v3, False, "[INST]You are a helpful AI assistant.\n\nHello[/INST]"),
            (TokenizerVersion.v7, False, "[SYSTEM_PROMPT]You are a helpful AI assistant.[/SYSTEM_PROMPT]"),
            (TokenizerVersion.v3, True, "[INST] You are a helpful AI assistant.\n\nHello[/INST]"),
            (TokenizerVersion.v7, True, "[SYSTEM_PROMPT]You are a helpful AI assistant.[/SYSTEM_PROMPT]"),
            (TokenizerVersion.v11, False, "[SYSTEM_PROMPT]You are a helpful AI assistant.[/SYSTEM_PROMPT]"),
            (TokenizerVersion.v13, False, "[SYSTEM_PROMPT]You are a helpful AI assistant.[/SYSTEM_PROMPT]"),
        ],
    )
    def test_default_prompt_used(
        self,
        version: TokenizerVersion,
        spm: bool,
        expected_fragment: str,
    ) -> None:
        chat_template = generate_chat_template(
            spm=spm,
            tokenizer_version=version,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        output = encode_transformers_from_openai(chat_template, CONV_NO_SYSTEM)

        assert self.DEFAULT_SYSTEM_PROMPT in output
        assert expected_fragment in output

    @pytest.mark.parametrize(
        "version",
        [TokenizerVersion.v3, TokenizerVersion.v7, TokenizerVersion.v11, TokenizerVersion.v13],
    )
    def test_default_prompt_ignored_when_user_provides(self, version: TokenizerVersion) -> None:
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        output = encode_transformers_from_openai(chat_template, self.CONV_WITH_SYSTEM)

        assert self.DEFAULT_SYSTEM_PROMPT not in output
        assert "You are a custom assistant." in output

    def test_legacy_no_default_prompt(self) -> None:
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v3,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        output = encode_transformers_from_openai(chat_template, CONV_NO_SYSTEM)

        assert "[INST]Hello[/INST]" in output

    def test_modern_no_default_prompt(self) -> None:
        chat_template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        output = encode_transformers_from_openai(chat_template, CONV_NO_SYSTEM)

        assert "[SYSTEM_PROMPT]" not in output
        assert "[/SYSTEM_PROMPT]" not in output

    def test_v15_default_system_prompt(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        result = encode_transformers_from_openai(template, CONV_NO_SYSTEM)
        assert self.DEFAULT_SYSTEM_PROMPT in result
        assert "[SYSTEM_PROMPT]" in result
        assert "[MODEL_SETTINGS]" in result

        result_with_system = encode_transformers_from_openai(template, self.CONV_WITH_SYSTEM)
        assert self.DEFAULT_SYSTEM_PROMPT not in result_with_system
        assert "You are a custom assistant." in result_with_system


class TestSystemPromptEscaping:
    @pytest.mark.parametrize(
        "prompt",
        [
            "You're a helpful assistant! Use \"quotes\" and 'apostrophes'.",
            "use \\ for escaping",
            "Line 1\nLine 2\nLine 3",
            r"Step \1: initialize, Step \2: run",
            "Use {{ variable }} and {% if true %}block{% endif %}",
        ],
    )
    def test_special_chars_preserved(self, prompt: str) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v7,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=prompt,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        result = encode_transformers_from_openai(template, CONV_NO_SYSTEM)

        assert prompt in result

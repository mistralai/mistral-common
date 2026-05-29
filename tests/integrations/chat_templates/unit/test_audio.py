from typing import Any

import pytest

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import render_template


class TestAudioSystemPrompt:
    @pytest.mark.parametrize(
        "version",
        [TokenizerVersion.v7, TokenizerVersion.v11, TokenizerVersion.v13],
    )
    def test_audio_with_system_prompt_raises(self, version: TokenizerVersion) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=version,
            image_support=False,
            audio_support=True,
            thinking_support=False,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this audio?"},
                    {"type": "input_audio", "input_audio": {"data": "base64data", "format": "wav"}},
                ],
            },
            {"role": "assistant", "content": "An audio clip."},
        ]

        with pytest.raises(
            ValueError, match="Audio chunks are not supported in user message content when system prompt is provided"
        ):
            render_template(template, messages)

    def test_v15_audio_with_system_prompt_succeeds(self) -> None:
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=True,
            thinking_support=False,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this audio?"},
                    {"type": "input_audio", "input_audio": {"data": "base64data", "format": "wav"}},
                ],
            },
            {"role": "assistant", "content": "An audio clip."},
        ]

        output = render_template(template, messages)
        assert "[AUDIO]" in output
        assert "[SYSTEM_PROMPT]" in output

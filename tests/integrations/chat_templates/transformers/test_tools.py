import pytest
from jinja2.exceptions import TemplateError

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import encode_transformers


class TestToolCallValidation:
    @pytest.mark.parametrize(
        ("spm", "version", "image", "audio", "think"),
        [
            (False, TokenizerVersion.v3, False, False, False),
            (True, TokenizerVersion.v3, False, False, False),
            (False, TokenizerVersion.v3, True, False, False),
            (True, TokenizerVersion.v3, True, False, False),
            (False, TokenizerVersion.v7, False, False, False),
            (True, TokenizerVersion.v7, False, False, False),
            (False, TokenizerVersion.v7, True, False, False),
            (True, TokenizerVersion.v7, True, False, False),
            (False, TokenizerVersion.v7, False, True, False),
            (False, TokenizerVersion.v11, False, False, False),
            (False, TokenizerVersion.v11, True, False, False),
            (False, TokenizerVersion.v11, False, True, False),
        ],
    )
    def test_tool_call_id_length(
        self,
        spm: bool,
        version: TokenizerVersion,
        image: bool,
        audio: bool,
        think: bool,
    ) -> None:
        invalid_id_conv = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "1", "function": {"name": "func", "arguments": "{}"}}],
                },
            ]
        }

        chat_template = generate_chat_template(
            spm=spm, tokenizer_version=version, image_support=image, audio_support=audio, thinking_support=think
        )
        with pytest.raises(TemplateError, match="Tool call must have an id of 9 characters or numbers."):
            encode_transformers(chat_template, invalid_id_conv)

    @pytest.mark.parametrize(
        ("spm", "version", "image", "audio", "think"),
        [
            (False, TokenizerVersion.v2, False, False, False),
            (True, TokenizerVersion.v2, False, False, False),
            (False, TokenizerVersion.v3, False, False, False),
            (True, TokenizerVersion.v3, False, False, False),
            (False, TokenizerVersion.v3, True, False, False),
            (True, TokenizerVersion.v3, True, False, False),
        ],
    )
    def test_content_with_tool_calls_rejected(
        self,
        spm: bool,
        version: TokenizerVersion,
        image: bool,
        audio: bool,
        think: bool,
    ) -> None:
        invalid_message_conv = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "hey",
                    "tool_calls": [{"id": "123456789", "function": {"name": "func", "arguments": "{}"}}],
                },
            ]
        }

        chat_template = generate_chat_template(
            spm=spm, tokenizer_version=version, image_support=image, audio_support=audio, thinking_support=think
        )
        with pytest.raises(TemplateError, match="Assistant message cannot have both content and tool calls."):
            encode_transformers(chat_template, invalid_message_conv)

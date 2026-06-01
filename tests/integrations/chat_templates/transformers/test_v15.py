from typing import Optional

import pytest
from jinja2.exceptions import TemplateError

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.integrations.chat_templates.helpers import encode_transformers


class TestV15ReasoningEffort:
    @pytest.mark.parametrize(
        ("spm", "version", "image", "audio", "think"),
        [
            (False, TokenizerVersion.v15, False, False, False),
            (False, TokenizerVersion.v15, True, False, False),
            (False, TokenizerVersion.v15, False, False, True),
            (False, TokenizerVersion.v15, True, False, True),
        ],
    )
    @pytest.mark.parametrize(
        "reasoning_effort",
        [None, "high", "none"],
        ids=["no_effort", "high", "none"],
    )
    def test_valid_reasoning_effort(
        self,
        spm: bool,
        version: TokenizerVersion,
        image: bool,
        audio: bool,
        think: bool,
        reasoning_effort: Optional[str],
    ) -> None:
        chat_template = generate_chat_template(
            spm=spm,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=think,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        conv: dict = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        if reasoning_effort is not None:
            conv["reasoning_effort"] = reasoning_effort

        result = encode_transformers(chat_template, conv)
        assert "[MODEL_SETTINGS]" in result

    @pytest.mark.parametrize(
        ("spm", "version", "image", "audio", "think"),
        [
            (False, TokenizerVersion.v15, False, False, False),
            (False, TokenizerVersion.v15, True, False, False),
            (False, TokenizerVersion.v15, False, False, True),
            (False, TokenizerVersion.v15, True, False, True),
        ],
    )
    @pytest.mark.parametrize(
        "reasoning_effort",
        ["low", "medium", "invalid_value"],
    )
    def test_invalid_reasoning_effort(
        self,
        spm: bool,
        version: TokenizerVersion,
        image: bool,
        audio: bool,
        think: bool,
        reasoning_effort: str,
    ) -> None:
        chat_template = generate_chat_template(
            spm=spm,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=think,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
        )

        conv = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "reasoning_effort": reasoning_effort,
        }

        with pytest.raises(TemplateError, match='reasoning_effort must be either "none" or "high"'):
            encode_transformers(chat_template, conv)

import copy
from pathlib import Path

import pytest
from jinja2.exceptions import TemplateError

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.integrations.chat_templates.template_generator import build_chat_template
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integrations.chat_templates.conftest import ALL_TRANSFORMERS_CONFIGS, format_config_id
from tests.integrations.chat_templates.fixtures_data import OPENAI_REQUEST_NON_CANONICAL_TOOLS
from tests.integrations.chat_templates.helpers import (
    TestConfig,
    build_spm_path,
    build_tekken_json,
    build_template_config,
    encode_mistral_common,
)
from tests.integrations.chat_templates.hf_utils import encode_transformers_from_openai

_TOOLS_CONFIGS = [c for c in ALL_TRANSFORMERS_CONFIGS if c.version >= TokenizerVersion.v2]


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
            spm=spm,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=think,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none" if version.supports_model_settings else None,
            reasoning_effort_values=("none", "high") if version.supports_model_settings else None,
        )
        with pytest.raises(TemplateError, match="Tool call must have an id of 9 characters or numbers."):
            encode_transformers_from_openai(chat_template, invalid_id_conv)

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
            spm=spm,
            tokenizer_version=version,
            image_support=image,
            audio_support=audio,
            thinking_support=think,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=None,
            reasoning_effort_default="none" if version.supports_model_settings else None,
            reasoning_effort_values=("none", "high") if version.supports_model_settings else None,
        )
        with pytest.raises(TemplateError, match="Assistant message cannot have both content and tool calls."):
            encode_transformers_from_openai(chat_template, invalid_message_conv)


class TestToolDefinitionParity:
    @pytest.mark.parametrize("config", _TOOLS_CONFIGS, ids=[format_config_id(c) for c in _TOOLS_CONFIGS])
    def test_non_canonical_tools_render_like_tokenizer(self, config: TestConfig, tmp_path: Path) -> None:
        # The tokenizer drops `strict` and unknown keys, defaults `description` and `parameters`, and
        # writes its own key order: the template must render the same tool definitions.
        if config.spm:
            tokenizer_path = build_spm_path(config=config, output_dir=tmp_path)
        else:
            tokenizer_path = build_tekken_json(config=config, output_dir=tmp_path)
        mistral_tokenizer = MistralTokenizer.from_file(str(tokenizer_path), mode=ValidationMode.test)
        request = ChatCompletionRequest.from_openai(**copy.deepcopy(OPENAI_REQUEST_NON_CANONICAL_TOOLS))
        chat_template = build_chat_template(config=build_template_config(config))

        rendered = encode_transformers_from_openai(chat_template, copy.deepcopy(OPENAI_REQUEST_NON_CANONICAL_TOOLS))

        assert rendered == encode_mistral_common(mistral_tokenizer, request, config.spm)

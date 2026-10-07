from pathlib import Path
from typing import Optional

import pytest
from jinja2.exceptions import TemplateError

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    JsonSchema,
    ResponseFormat,
    ResponseFormats,
)
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integrations.chat_templates.fixtures_data import (
    PARITY_JSON_SCHEMA,
    PARITY_JSON_SCHEMA_NON_ASCII,
)
from tests.integrations.chat_templates.helpers import TestConfig, _build_tekken_json, encode_mistral_common
from tests.integrations.chat_templates.hf_utils import (
    _build_hf_tokenizer,
    encode_hf_tokens,
    encode_transformers,
    encode_transformers_from_openai,
)


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

        result = encode_transformers_from_openai(chat_template, conv)
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
            encode_transformers_from_openai(chat_template, conv)


class TestV15ResponseFormat:
    def test_response_format_json_schema_parity(self, tmp_path: Path) -> None:
        model_settings_fields = frozenset({"reasoning_effort", "json_schema"})
        tokenizer_path = _build_tekken_json(
            config=TestConfig(version=TokenizerVersion.v15, model_settings_fields=model_settings_fields),
            output_dir=tmp_path,
        )
        mistral_tokenizer = MistralTokenizer.from_file(str(tokenizer_path), mode=ValidationMode.test)
        request = ChatCompletionRequest(
            messages=[UserMessage(content="Hello")],
            reasoning_effort="high",
            response_format=ResponseFormat(
                type=ResponseFormats.json_schema,
                json_schema=JsonSchema(name="answer", schema=PARITY_JSON_SCHEMA),
            ),
        )
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=model_settings_fields,
        )

        assert encode_transformers(chat_template=template, chat_request=request) == encode_mistral_common(
            mistral_tokenizer=mistral_tokenizer, chat_request=request, spm=False
        )

    def test_response_format_json_schema_non_ascii_parity(self, tmp_path: Path) -> None:
        model_settings_fields = frozenset({"reasoning_effort", "json_schema"})
        tokenizer_path = _build_tekken_json(
            config=TestConfig(version=TokenizerVersion.v15, model_settings_fields=model_settings_fields),
            output_dir=tmp_path,
        )
        mistral_tokenizer = MistralTokenizer.from_file(str(tokenizer_path), mode=ValidationMode.test)
        request = ChatCompletionRequest(
            messages=[UserMessage(content="Hello")],
            reasoning_effort="high",
            response_format=ResponseFormat(
                type=ResponseFormats.json_schema,
                json_schema=JsonSchema(name="answer", schema=PARITY_JSON_SCHEMA_NON_ASCII),
            ),
        )
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=model_settings_fields,
        )

        assert encode_transformers(chat_template=template, chat_request=request) == encode_mistral_common(
            mistral_tokenizer=mistral_tokenizer, chat_request=request, spm=False
        )

    def test_response_format_json_object_parity(self, tmp_path: Path) -> None:
        model_settings_fields = frozenset({"reasoning_effort", "json_schema"})
        tokenizer_path = _build_tekken_json(
            config=TestConfig(version=TokenizerVersion.v15, model_settings_fields=model_settings_fields),
            output_dir=tmp_path,
        )
        mistral_tokenizer = MistralTokenizer.from_file(str(tokenizer_path), mode=ValidationMode.test)
        request = ChatCompletionRequest(
            messages=[UserMessage(content="Hello")],
            response_format=ResponseFormat(type=ResponseFormats.json),
        )
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=model_settings_fields,
        )

        assert encode_transformers(chat_template=template, chat_request=request) == encode_mistral_common(
            mistral_tokenizer=mistral_tokenizer, chat_request=request, spm=False
        )

    def test_response_format_text_parity(self, tmp_path: Path) -> None:
        model_settings_fields = frozenset({"reasoning_effort", "json_schema"})
        tokenizer_path = _build_tekken_json(
            config=TestConfig(version=TokenizerVersion.v15, model_settings_fields=model_settings_fields),
            output_dir=tmp_path,
        )
        mistral_tokenizer = MistralTokenizer.from_file(str(tokenizer_path), mode=ValidationMode.test)
        request = ChatCompletionRequest(
            messages=[UserMessage(content="Hello")],
            response_format=ResponseFormat(type=ResponseFormats.text),
        )
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=model_settings_fields,
        )

        assert encode_transformers(chat_template=template, chat_request=request) == encode_mistral_common(
            mistral_tokenizer=mistral_tokenizer, chat_request=request, spm=False
        )

    def test_response_format_json_schema_token_ids_parity(self, tmp_path: Path) -> None:
        model_settings_fields = frozenset({"reasoning_effort", "json_schema"})
        tokenizer_path = _build_tekken_json(
            config=TestConfig(version=TokenizerVersion.v15, model_settings_fields=model_settings_fields),
            output_dir=tmp_path,
        )
        mistral_tokenizer = MistralTokenizer.from_file(str(tokenizer_path), mode=ValidationMode.test)
        request = ChatCompletionRequest(
            messages=[UserMessage(content="Hello")],
            reasoning_effort="high",
            response_format=ResponseFormat(
                type=ResponseFormats.json_schema,
                json_schema=JsonSchema(name="answer", schema=PARITY_JSON_SCHEMA),
            ),
        )
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=model_settings_fields,
        )
        hf_tokenizer = _build_hf_tokenizer(tekken_path=tokenizer_path, chat_template=template)

        assert (
            encode_hf_tokens(hf_tokenizer=hf_tokenizer, chat_request=request)
            == mistral_tokenizer.encode_chat_completion(request=request).tokens
        )

    def test_response_format_missing_schema_raises_in_transformers(self) -> None:
        model_settings_fields = frozenset({"reasoning_effort", "json_schema"})
        template = generate_chat_template(
            spm=False,
            tokenizer_version=TokenizerVersion.v15,
            image_support=False,
            audio_support=False,
            thinking_support=False,
            default_system_prompt=None,
            plain_thinking_support=False,
            use_special_token_variables=True,
            model_settings_fields=model_settings_fields,
        )
        openai_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer"}},
        }

        with pytest.raises(TemplateError, match="Response format `json_schema` must define the schema"):
            encode_transformers_from_openai(template, openai_request)

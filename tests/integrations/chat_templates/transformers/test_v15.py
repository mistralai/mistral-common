from pathlib import Path
from typing import Any, Optional

import pytest
from jinja2.exceptions import TemplateError

from mistral_common.integrations.chat_templates.chat_templates import generate_chat_template
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    JsonSchema,
    ReasoningEffort,
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
from tests.integrations.chat_templates.helpers import TestConfig, build_tekken_json, encode_mistral_common
from tests.integrations.chat_templates.hf_utils import (
    build_hf_tokenizer,
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
            model_settings_fields=None,
            reasoning_effort_default="none",
            reasoning_effort_values=("none", "high"),
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
        ["medium", "invalid_value"],
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
            model_settings_fields=None,
            reasoning_effort_values=("none", "high"),
            reasoning_effort_default="none",
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


_BOTH_FIELDS = frozenset({"reasoning_effort", "json_schema"})
_JSON_SCHEMA_ONLY = frozenset({"json_schema"})


def _json_schema_request(schema: dict[str, Any], reasoning_effort: ReasoningEffort | None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[UserMessage(content="Hello")],
        reasoning_effort=reasoning_effort,
        response_format=ResponseFormat(
            type=ResponseFormats.json_schema,
            json_schema=JsonSchema(name="answer", schema=schema),
        ),
    )


_PARITY_CASES = [
    pytest.param(
        _BOTH_FIELDS,
        _json_schema_request(schema=PARITY_JSON_SCHEMA, reasoning_effort=ReasoningEffort.high),
        id="json_schema",
    ),
    pytest.param(
        _BOTH_FIELDS,
        _json_schema_request(schema=PARITY_JSON_SCHEMA_NON_ASCII, reasoning_effort=ReasoningEffort.high),
        id="json_schema_non_ascii",
    ),
    pytest.param(
        _BOTH_FIELDS,
        ChatCompletionRequest(
            messages=[UserMessage(content="Hello")], response_format=ResponseFormat(type=ResponseFormats.json_object)
        ),
        id="json_object",
    ),
    pytest.param(
        _BOTH_FIELDS,
        ChatCompletionRequest(
            messages=[UserMessage(content="Hello")], response_format=ResponseFormat(type=ResponseFormats.text)
        ),
        id="text",
    ),
    pytest.param(
        _JSON_SCHEMA_ONLY,
        _json_schema_request(schema=PARITY_JSON_SCHEMA, reasoning_effort=None),
        id="json_schema_only_field",
    ),
]


def _build_v15_parity_setup(
    tmp_path: Path, model_settings_fields: frozenset[str]
) -> tuple[Path, MistralTokenizer, str]:
    tokenizer_path = build_tekken_json(
        config=TestConfig(version=TokenizerVersion.v15, model_settings_fields=model_settings_fields),
        output_dir=tmp_path,
    )
    mistral_tokenizer = MistralTokenizer.from_file(tokenizer_filename=str(tokenizer_path), mode=ValidationMode.test)
    has_effort = "reasoning_effort" in model_settings_fields
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
        reasoning_effort_values=("none", "high") if has_effort else None,
        reasoning_effort_default="none" if has_effort else None,
    )
    return tokenizer_path, mistral_tokenizer, template


class TestV15ResponseFormat:
    @pytest.mark.parametrize(("model_settings_fields", "request_"), _PARITY_CASES)
    def test_response_format_text_parity(
        self, tmp_path: Path, model_settings_fields: frozenset[str], request_: ChatCompletionRequest
    ) -> None:
        _, mistral_tokenizer, template = _build_v15_parity_setup(
            tmp_path=tmp_path, model_settings_fields=model_settings_fields
        )

        assert encode_transformers(chat_template=template, chat_request=request_) == encode_mistral_common(
            mistral_tokenizer=mistral_tokenizer, chat_request=request_, spm=False
        )

    @pytest.mark.parametrize(("model_settings_fields", "request_"), _PARITY_CASES)
    def test_response_format_token_ids_parity(
        self, tmp_path: Path, model_settings_fields: frozenset[str], request_: ChatCompletionRequest
    ) -> None:
        tokenizer_path, mistral_tokenizer, template = _build_v15_parity_setup(
            tmp_path=tmp_path, model_settings_fields=model_settings_fields
        )
        hf_tokenizer = build_hf_tokenizer(tekken_path=tokenizer_path, chat_template=template)

        assert (
            encode_hf_tokens(hf_tokenizer=hf_tokenizer, chat_request=request_)
            == mistral_tokenizer.encode_chat_completion(request=request_).tokens
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
            reasoning_effort_values=("none", "high"),
            reasoning_effort_default="none",
        )
        openai_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer"}},
        }

        with pytest.raises(TemplateError, match="Response format `json_schema` must define the schema"):
            encode_transformers_from_openai(chat_template=template, openai_request=openai_request)

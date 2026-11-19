import json
import pickle
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.messages import UATS, UserMessage
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    JsonSchema,
    ModelSettings,
    ReasoningEffort,
    ResponseFormat,
    ResponseFormats,
    SchemaRenderingMode,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import (
    EnumBuilder,
    FieldBuilder,
    JSONSchemaBuilder,
    ModelSettingsBuilder,
    ReasoningEffortEnumBuilder,
    ValidatorType,
)
from tests.utils.chat_templates.helpers import TestConfig, build_tekken_json

_REASONING_EFFORT_ONLY_BUILDER = ModelSettingsBuilder(
    reasoning_effort=ReasoningEffortEnumBuilder(
        accepts_none=True, default=ReasoningEffort.none, values=[ReasoningEffort.none, ReasoningEffort.high]
    )
)


class StringToIntBuilder(FieldBuilder[str, int]):
    type: ValidatorType = ValidatorType.ENUM
    accepts_none: bool = False
    default: int | None = None

    def _convert(self, input_value: str) -> int:
        return int(input_value)

    def _validate_built_value(self, field_name: str, value: int) -> None:
        if value < 0:
            raise InvalidRequestException(f"{field_name} should be non-negative.")


class TestFieldBuilder:
    def test_converts_and_validates_output(self) -> None:
        builder = StringToIntBuilder(accepts_none=False, default=None)

        converted = builder.build_value(field_name="count", value="7")
        assert converted is not None
        assert converted + 1 == 8

        with pytest.raises(InvalidRequestException, match="count should be non-negative"):
            builder.build_value(field_name="count", value="-1")


class TestNoneFactories:
    def test_none_model_settings(self) -> None:
        settings = ModelSettings.none()
        for field_name in ModelSettings.model_fields.keys():
            assert getattr(settings, field_name) is None

    def test_none_model_settings_builder(self) -> None:
        settings = ModelSettingsBuilder.none()
        for field_name in ModelSettingsBuilder.model_fields.keys():
            assert getattr(settings, field_name) is None


class TestEnumBuilder:
    def test_duplicate_values_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate values"):
            EnumBuilder[ReasoningEffort](
                values=[ReasoningEffort.none, ReasoningEffort.high, ReasoningEffort.none],
                accepts_none=False,
                default=None,
            )

    def test_unique_values_accepted(self) -> None:
        builder = EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=False, default=None
        )
        assert set(builder.values) == {ReasoningEffort.none, ReasoningEffort.high}

    def test_empty_values_list(self) -> None:
        EnumBuilder[ReasoningEffort](values=[], accepts_none=True, default=None)
        with pytest.raises(ValidationError, match="Empty list of values"):
            EnumBuilder[ReasoningEffort](values=[], accepts_none=False, default=None)

    def test_missing_required_value_rejected_on_build(self) -> None:
        builder = EnumBuilder[ReasoningEffort](values=[ReasoningEffort.high], accepts_none=False, default=None)
        with pytest.raises(InvalidRequestException, match="reasoning_effort should be set for this model"):
            builder.build_value(field_name="reasoning_effort", value=None)

    def test_empty_allowed_values_reject_set_value_on_build(self) -> None:
        builder = EnumBuilder[ReasoningEffort](values=[], accepts_none=True, default=None)
        with pytest.raises(InvalidRequestException, match="reasoning_effort not supported for this model"):
            builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.high)


class TestReasoningEffortEnumBuilder:
    def test_accepts_configured_values_and_builds_default(self) -> None:
        builder = ReasoningEffortEnumBuilder(
            accepts_none=True,
            default=ReasoningEffort.medium,
            values=[ReasoningEffort.low, ReasoningEffort.medium, ReasoningEffort.max],
        )

        settings = ModelSettingsBuilder(reasoning_effort=builder).build_settings(
            ChatCompletionRequest(messages=[], reasoning_effort=None)
        )

        assert settings == ModelSettings(reasoning_effort=ReasoningEffort.medium)

    def test_accepts_none_is_required(self) -> None:
        with pytest.raises(ValidationError, match="accepts_none"):
            ReasoningEffortEnumBuilder(
                accepts_none=False,
                default=None,
                values=[ReasoningEffort.low],
            )

    def test_default_is_required(self) -> None:
        with pytest.raises(ValidationError, match="must declare a default"):
            ReasoningEffortEnumBuilder(
                accepts_none=True,
                default=None,
                values=[ReasoningEffort.low],
            )

    def test_rejects_effort_outside_configured_values(self) -> None:
        builder = ReasoningEffortEnumBuilder(
            accepts_none=True,
            default=ReasoningEffort.low,
            values=[ReasoningEffort.low, ReasoningEffort.medium],
        )

        with pytest.raises(InvalidRequestException, match="should be one of"):
            builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.max)

    @pytest.mark.parametrize("version", list(TokenizerVersion), ids=lambda v: v.name)
    def test_from_file_admits_builder_only_for_settings_versions(
        self, version: TokenizerVersion, tmp_path: Path
    ) -> None:
        r"""A tokenizer file carrying a builder loads on v15 and is rejected on every pre-v15 version."""
        path = build_tekken_json(config=TestConfig(version=version), output_dir=tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["model_settings_builder"] = _REASONING_EFFORT_ONLY_BUILDER.model_dump(mode="json")
        path.write_text(json.dumps(data), encoding="utf-8")

        if version.supports_model_settings:
            tokenizer = MistralTokenizer.from_file(str(path))
            assert isinstance(
                tokenizer.instruct_tokenizer.tokenizer.model_settings_builder.reasoning_effort,
                ReasoningEffortEnumBuilder,
            )
        else:
            with pytest.raises(ValueError, match="model_settings_builder is not supported"):
                MistralTokenizer.from_file(str(path))

    def test_model_settings_builder_round_trips_through_pickle(self) -> None:
        settings_builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortEnumBuilder(
                accepts_none=True,
                default=ReasoningEffort.none,
                values=[ReasoningEffort.none, ReasoningEffort.high],
            )
        )

        restored = pickle.loads(pickle.dumps(settings_builder))

        assert restored == settings_builder
        assert type(restored.reasoning_effort) is ReasoningEffortEnumBuilder


class TestBuildSettings:
    @pytest.mark.parametrize(
        ("builder", "chat_completion_request", "settings"),
        [
            pytest.param(
                ModelSettingsBuilder(reasoning_effort=None),
                ChatCompletionRequest(messages=[], reasoning_effort=None),
                ModelSettings.none(),
                id="no-builder-unset",
            ),
            pytest.param(
                ModelSettingsBuilder(
                    reasoning_effort=ReasoningEffortEnumBuilder(
                        values=[ReasoningEffort.none, ReasoningEffort.high], default=ReasoningEffort.none
                    )
                ),
                ChatCompletionRequest(messages=[], reasoning_effort=ReasoningEffort.none),
                ModelSettings(reasoning_effort=ReasoningEffort.none),
                id="allowed-none",
            ),
            pytest.param(
                ModelSettingsBuilder(
                    reasoning_effort=ReasoningEffortEnumBuilder(
                        values=[ReasoningEffort.high], default=ReasoningEffort.high
                    )
                ),
                ChatCompletionRequest(messages=[], reasoning_effort=ReasoningEffort.high),
                ModelSettings(reasoning_effort=ReasoningEffort.high),
                id="allowed-high",
            ),
            pytest.param(
                ModelSettingsBuilder(
                    reasoning_effort=ReasoningEffortEnumBuilder(
                        values=[ReasoningEffort.high], default=ReasoningEffort.high
                    ),
                ),
                ChatCompletionRequest(messages=[], reasoning_effort=None),
                ModelSettings(reasoning_effort=ReasoningEffort.high),
                id="default-high",
            ),
        ],
    )
    def test_build_settings_success_scenarios(
        self, builder: ModelSettingsBuilder, chat_completion_request: ChatCompletionRequest, settings: ModelSettings
    ) -> None:
        assert settings == builder.build_settings(chat_completion_request)

    def test_build_settings_disallowed_enum_value_rejected(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortEnumBuilder(values=[ReasoningEffort.high], default=ReasoningEffort.high)
        )
        chat_completion_request = ChatCompletionRequest[UATS](messages=[], reasoning_effort=ReasoningEffort.none)
        with pytest.raises(
            InvalidRequestException,
            match=r"reasoning_effort should be one of \['high'\], got none\.",
        ):
            builder.build_settings(chat_completion_request)

    def test_build_settings_unexpected_value_rejected(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortEnumBuilder(values=[ReasoningEffort.none], default=ReasoningEffort.none)
        )
        chat_completion_request = ChatCompletionRequest[UATS](messages=[], reasoning_effort=ReasoningEffort.high)

        with pytest.raises(InvalidRequestException, match=r"reasoning_effort should be one of"):
            builder.build_settings(chat_completion_request)

    def test_build_settings_ignores_field_without_builder(self) -> None:
        builder = ModelSettingsBuilder(reasoning_effort=None)
        chat_completion_request = ChatCompletionRequest[UATS](messages=[], reasoning_effort=ReasoningEffort.high)

        assert builder.build_settings(chat_completion_request) == ModelSettings.none()

    def test_default_must_be_in_values(self) -> None:
        with pytest.raises(ValidationError, match=r"Default value self.default='high' is not in self.values"):
            ReasoningEffortEnumBuilder(values=[ReasoningEffort.none], default=ReasoningEffort.high)

    def test_default_resolves_unset_value(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortEnumBuilder(values=[ReasoningEffort.none], default=ReasoningEffort.none)
        )
        assert builder.build_settings(ChatCompletionRequest(messages=[], reasoning_effort=None)) == ModelSettings(
            reasoning_effort=ReasoningEffort.none
        )


class TestValidateSettings:
    @pytest.mark.parametrize(
        ("reasoning_effort", "expected_match"),
        [
            pytest.param(None, None, id="unset-accepted"),
            pytest.param(ReasoningEffort.none, r"reasoning_effort not supported for this model", id="none-rejected"),
            pytest.param(ReasoningEffort.high, r"reasoning_effort not supported for this model", id="high-rejected"),
        ],
    )
    def test_validate_settings_without_builder(
        self, reasoning_effort: ReasoningEffort | None, expected_match: str | None
    ) -> None:
        builder = ModelSettingsBuilder()
        settings = ModelSettings(reasoning_effort=reasoning_effort)
        if expected_match is None:
            builder.validate_settings(settings)
        else:
            with pytest.raises(InvalidRequestException, match=expected_match):
                builder.validate_settings(settings)

    @pytest.mark.parametrize(
        ("reasoning_effort", "expected_match"),
        [
            pytest.param(None, r"reasoning_effort should be set for this model", id="unset-rejected"),
            pytest.param(ReasoningEffort.none, None, id="allowed-none-accepted"),
            pytest.param(ReasoningEffort.high, r"reasoning_effort should be one of", id="disallowed-high-rejected"),
        ],
    )
    def test_validate_settings_with_builder(
        self, reasoning_effort: ReasoningEffort | None, expected_match: str | None
    ) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortEnumBuilder(values=[ReasoningEffort.none], default=ReasoningEffort.none)
        )
        settings = ModelSettings(reasoning_effort=reasoning_effort)
        if expected_match is None:
            builder.validate_settings(settings)
        else:
            with pytest.raises(InvalidRequestException, match=expected_match):
                builder.validate_settings(settings)


JSON_SCHEMA_DICT = {"type": "object", "properties": {"a": {"type": "string"}}}
JSON_OBJECT_OR_ARRAY_SCHEMA = {"anyOf": [{"type": "object"}, {"type": "array"}]}


def _req(rf: ResponseFormat) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[UserMessage(content="hi")], response_format=rf)


class TestJSONSchemaBuilder:
    def test_builder_text_yields_none(self) -> None:
        b = ModelSettingsBuilder(json_schema=JSONSchemaBuilder())
        response_format = ResponseFormat(type=ResponseFormats.text)
        assert response_format.get_schema(purpose=SchemaRenderingMode.model_settings) is None
        assert b.build_settings(_req(response_format)).json_schema is None

    def test_builder_json_yields_anyof(self) -> None:
        b = ModelSettingsBuilder(json_schema=JSONSchemaBuilder())
        response_format = ResponseFormat(type=ResponseFormats.json_object)
        assert response_format.get_schema(purpose=SchemaRenderingMode.model_settings) == JSON_OBJECT_OR_ARRAY_SCHEMA
        assert b.build_settings(_req(response_format)).json_schema == JSON_OBJECT_OR_ARRAY_SCHEMA

    @pytest.mark.parametrize("strict", [False, True])
    def test_builder_json_schema_yields_custom_schema(self, strict: bool) -> None:
        b = ModelSettingsBuilder(json_schema=JSONSchemaBuilder())
        response_format = ResponseFormat(
            type=ResponseFormats.json_schema,
            json_schema=JsonSchema(name="x", schema=JSON_SCHEMA_DICT, strict=strict),
        )
        assert response_format.get_schema(purpose=SchemaRenderingMode.model_settings) == JSON_SCHEMA_DICT
        assert b.build_settings(_req(response_format)).json_schema == JSON_SCHEMA_DICT

    @pytest.mark.parametrize(
        ("strict", "expected_grammar_schema"),
        [
            pytest.param(False, JSON_OBJECT_OR_ARRAY_SCHEMA, id="non-strict"),
            pytest.param(True, JSON_SCHEMA_DICT, id="strict"),
        ],
    )
    def test_response_format_grammar_schema_respects_strict(
        self, strict: bool, expected_grammar_schema: dict[str, Any]
    ) -> None:
        response_format = ResponseFormat(
            type=ResponseFormats.json_schema,
            json_schema=JsonSchema(name="x", schema=JSON_SCHEMA_DICT, strict=strict),
        )

        assert response_format.get_schema(purpose=SchemaRenderingMode.grammar) == expected_grammar_schema
        assert response_format.get_schema(purpose=SchemaRenderingMode.model_settings) == JSON_SCHEMA_DICT

    def test_json_schema_rejects_invalid_schema(self) -> None:
        with pytest.raises(ValidationError, match="123 is not valid under any of the given schemas"):
            JsonSchema(name="x", schema={"type": 123})

    def test_json_schema_missing_schema_raises(self) -> None:
        with pytest.raises(ValidationError, match="Response format `json_schema` must define the schema"):
            ResponseFormat(type=ResponseFormats.json_schema)

    def test_builder_without_json_schema_ignores_response_format(self) -> None:
        settings = ModelSettingsBuilder().build_settings(_req(ResponseFormat(type=ResponseFormats.json_object)))
        assert settings.json_schema is None

    def test_validate_settings_rejects_invalid_json_schema(self) -> None:
        builder = ModelSettingsBuilder(json_schema=JSONSchemaBuilder())

        with pytest.raises(InvalidRequestException, match="json_schema is not a valid JSON schema"):
            builder.validate_settings(settings=ModelSettings(json_schema={"type": 123}))


class TestFieldParity:
    def test_all_model_settings_and_builder_fields_match(self) -> None:
        model_fields = set(ModelSettings.model_fields.keys())
        builder_fields = set(ModelSettingsBuilder.model_fields.keys())

        assert model_fields == builder_fields
        assert set(ModelSettingsBuilder._SETTINGS_TO_REQUEST_FIELDS) == builder_fields
        assert set(ModelSettingsBuilder._SETTINGS_TO_REQUEST_FIELDS.values()) <= set(ChatCompletionRequest.model_fields)

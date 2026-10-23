from enum import StrEnum, auto
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
from mistral_common.tokens.tokenizers.model_settings_builder import (
    EnumBuilder,
    FieldBuilder,
    JSONSchemaBuilder,
    ModelSettingsBuilder,
    ReasoningEffortBuilder,
    ValidatorType,
)


class StringToIntBuilder(FieldBuilder[str, int]):
    type: ValidatorType = ValidatorType.ENUM
    accepts_none: bool = False
    default: int | None = None

    def _convert(self, field_name: str, input_value: str) -> int:
        return int(input_value)

    def _validate_built_value(self, field_name: str, value: int) -> None:
        if value < 0:
            raise InvalidRequestException(f"{field_name} should be non-negative.")


def test_field_builder_converts_and_validates_output() -> None:
    builder = StringToIntBuilder(accepts_none=False, default=None)

    converted = builder.build_value(field_name="count", value="7")
    assert converted is not None
    assert converted + 1 == 8

    with pytest.raises(InvalidRequestException, match="count should be non-negative"):
        builder.build_value(field_name="count", value="-1")


class TestModelSettings:
    def test_none_model_settings(self) -> None:
        settings = ModelSettings.none()
        for field_name in ModelSettings.model_fields.keys():
            assert getattr(settings, field_name) is None

    def test_none_model_settings_builder(self) -> None:
        settings = ModelSettingsBuilder.none()
        for field_name in ModelSettingsBuilder.model_fields.keys():
            assert getattr(settings, field_name) is None


class TestEnumBuilder:
    @pytest.mark.parametrize(("values"), [([ReasoningEffort.none, ReasoningEffort.high, ReasoningEffort.none])])
    def test_duplication(self, values: list[ReasoningEffort]) -> None:
        """Test that duplicate enum values are invalid during initialization."""
        if len(set(values)) != len(values):
            with pytest.raises(ValidationError, match="Duplicate values"):
                EnumBuilder[ReasoningEffort](values=values, accepts_none=False, default=None)
        else:
            builder = EnumBuilder[ReasoningEffort](values=values, accepts_none=False, default=None)
            assert set(builder.values) == set(values)

    def test_empty_values_list(self) -> None:
        EnumBuilder[ReasoningEffort](values=[], accepts_none=True, default=None)
        with pytest.raises(ValidationError, match="Empty list of values"):
            EnumBuilder[ReasoningEffort](values=[], accepts_none=False, default=None)

    def test_default_must_be_one_of_the_values(self) -> None:
        with pytest.raises(ValidationError, match="is not in"):
            EnumBuilder[ReasoningEffort](values=[ReasoningEffort.high], accepts_none=True, default=ReasoningEffort.none)

    def test_build_value_rejects_unexposed_value(self) -> None:
        builder = EnumBuilder[ReasoningEffort](values=[ReasoningEffort.high], accepts_none=False, default=None)
        with pytest.raises(InvalidRequestException, match="should be one of"):
            builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.none)

    def test_build_value_requires_value_when_not_accepted(self) -> None:
        builder = EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=False, default=None
        )
        with pytest.raises(InvalidRequestException, match="reasoning_effort should be set for this model"):
            builder.build_value(field_name="reasoning_effort", value=None)

    def test_build_value_empty_values_not_supported(self) -> None:
        builder = EnumBuilder[ReasoningEffort](values=[], accepts_none=True, default=None)
        with pytest.raises(InvalidRequestException, match="not supported for this model"):
            builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.high)


class TestModelSettingsBuilder:
    @pytest.mark.parametrize(
        ("builder", "chat_completion_request", "settings"),
        [
            (
                ModelSettingsBuilder(reasoning_effort=None),
                ChatCompletionRequest(messages=[], reasoning_effort=None),
                ModelSettings.none(),
            ),
            (
                ModelSettingsBuilder(
                    reasoning_effort=ReasoningEffortBuilder(
                        values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=False, default=None
                    )
                ),
                ChatCompletionRequest(messages=[], reasoning_effort=ReasoningEffort.none),
                ModelSettings(reasoning_effort=ReasoningEffort.none),
            ),
            (
                ModelSettingsBuilder(
                    reasoning_effort=ReasoningEffortBuilder(
                        values=[ReasoningEffort.high], accepts_none=False, default=None
                    )
                ),
                ChatCompletionRequest(messages=[], reasoning_effort=ReasoningEffort.high),
                ModelSettings(reasoning_effort=ReasoningEffort.high),
            ),
            (
                ModelSettingsBuilder(
                    reasoning_effort=ReasoningEffortBuilder(
                        values=[ReasoningEffort.high], accepts_none=True, default=ReasoningEffort.high
                    ),
                ),
                ChatCompletionRequest(messages=[], reasoning_effort=None),
                ModelSettings(reasoning_effort=ReasoningEffort.high),
            ),
        ],
    )
    def test_encode_successful_validation_scenarios(
        self, builder: ModelSettingsBuilder, chat_completion_request: ChatCompletionRequest, settings: ModelSettings
    ) -> None:
        assert settings == builder.build_settings(chat_completion_request)

    def test_encode_invalid_enum_value_raises_error(self) -> None:
        builder_high_only = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(values=[ReasoningEffort.high], accepts_none=False, default=None)
        )
        chat_completion_request = ChatCompletionRequest[UATS](messages=[], reasoning_effort=ReasoningEffort.none)
        with pytest.raises(
            InvalidRequestException,
            match=r"should be one of \['high'\], got none.",
        ):
            builder_high_only.build_settings(chat_completion_request)

    def test_encode_missing_required_field_raises_error(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=False, default=None
            )
        )
        chat_completion_request = ChatCompletionRequest[UATS](messages=[], reasoning_effort=None)

        with pytest.raises(
            InvalidRequestException,
            match=r"reasoning_effort should be set.",
        ):
            builder.build_settings(chat_completion_request)

    def test_encode_unexpected_field_raises_error(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.none], accepts_none=True, default=ReasoningEffort.none
            )
        )
        chat_completion_request = ChatCompletionRequest[UATS](messages=[], reasoning_effort=ReasoningEffort.high)

        with pytest.raises(InvalidRequestException, match=r"should be one of \['none'\], got high."):
            builder.build_settings(chat_completion_request)

    def test_ignore_field(self) -> None:
        builder = ModelSettingsBuilder(reasoning_effort=None)
        chat_completion_request = ChatCompletionRequest[UATS](messages=[], reasoning_effort=ReasoningEffort.high)

        assert builder.build_settings(chat_completion_request) == ModelSettings.none()

    def test_reasoning_effort_does_not_exist(self) -> None:
        with pytest.raises(ValidationError, match=r"[type=enum, input_value='blabla', input_type=str]"):
            ModelSettingsBuilder(
                reasoning_effort=ReasoningEffortBuilder(
                    values=[ReasoningEffort.none, "blabla"],  # type: ignore[list-item]
                    accepts_none=False,
                    default=None,
                )
            )

    def test_compatibility_accept_none_default(self) -> None:
        with pytest.raises(ValidationError, match=r"Default values can only be defined for accepts_none fields"):
            ModelSettingsBuilder(
                reasoning_effort=ReasoningEffortBuilder(
                    values=[ReasoningEffort.none], accepts_none=False, default=ReasoningEffort.none
                )
            )

    def test_default_in_values(self) -> None:
        with pytest.raises(ValidationError, match=r"Default value self.default='high' is not in self.values"):
            ModelSettingsBuilder(
                reasoning_effort=ReasoningEffortBuilder(
                    values=[ReasoningEffort.none], accepts_none=True, default=ReasoningEffort.high
                )
            )

    def test_default_works(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.none], accepts_none=True, default=ReasoningEffort.none
            )
        )
        assert builder.build_settings(ChatCompletionRequest(messages=[], reasoning_effort=None)) == ModelSettings(
            reasoning_effort=ReasoningEffort.none
        )

    @pytest.mark.parametrize(
        ("reasoning_effort"),
        ([None, ReasoningEffort.none, ReasoningEffort.high]),
    )
    def test_validator_no_settings(self, reasoning_effort: ReasoningEffort | None) -> None:
        builder = ModelSettingsBuilder()
        settings = ModelSettings(reasoning_effort=reasoning_effort)
        if reasoning_effort is not None:
            with pytest.raises(InvalidRequestException, match=r"reasoning_effort not supported for this model"):
                builder.validate_settings(settings)
        else:
            builder.validate_settings(settings)

    @pytest.mark.parametrize(
        ("reasoning_effort"),
        ([None, ReasoningEffort.none, ReasoningEffort.high]),
    )
    def test_validator_unavailable_setting(self, reasoning_effort: ReasoningEffort | None) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.none], accepts_none=True, default=ReasoningEffort.none
            )
        )
        settings = ModelSettings(reasoning_effort=reasoning_effort)
        if reasoning_effort != ReasoningEffort.none:
            with pytest.raises(InvalidRequestException, match=r"should be"):
                builder.validate_settings(settings)
        else:
            builder.validate_settings(settings)

    @pytest.mark.parametrize(("accepts_none"), ([False, True]))
    def test_validator_rejects_none_reasoning_effort_in_settings(self, accepts_none: bool) -> None:
        # A reasoning effort builder always declares a default, so a built
        # ModelSettings never carries a None reasoning effort.
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.none],
                accepts_none=accepts_none,
                default=ReasoningEffort.none if accepts_none else None,
            )
        )
        settings = ModelSettings(reasoning_effort=None)
        with pytest.raises(InvalidRequestException, match=r"reasoning_effort should be set for this model"):
            builder.validate_settings(settings)


class _Level(StrEnum):
    r"""Enum sharing value names with reasoning efforts, to check the default rule stays effort-specific."""

    low = auto()
    high = auto()


def _request(reasoning_effort: ReasoningEffort | None) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], reasoning_effort=reasoning_effort)


def _remapping_builder() -> ModelSettingsBuilder:
    r"""Builder exposing low, high and max, remapped onto the low, medium and high the model supports.

    The default is an exposed value and is mapped like any request value.
    """
    return ModelSettingsBuilder(
        reasoning_effort=ReasoningEffortBuilder(
            values=[ReasoningEffort.low, ReasoningEffort.high, ReasoningEffort.max],
            mapping={
                ReasoningEffort.low: ReasoningEffort.low,
                ReasoningEffort.high: ReasoningEffort.medium,
                ReasoningEffort.max: ReasoningEffort.high,
            },
            accepts_none=True,
            default=ReasoningEffort.high,
        )
    )


class TestReasoningEffortBuilderMapping:
    def test_mapping_is_none_by_default(self) -> None:
        builder = ReasoningEffortBuilder(
            values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=False, default=None
        )
        assert builder.mapping is None

    def test_mapping_keys_must_match_values(self) -> None:
        with pytest.raises(ValidationError, match="must match"):
            ReasoningEffortBuilder(
                values=[ReasoningEffort.none, ReasoningEffort.high],
                mapping={ReasoningEffort.high: ReasoningEffort.high},
                accepts_none=False,
                default=None,
            )
        with pytest.raises(ValidationError, match="must match"):
            ReasoningEffortBuilder(
                values=[ReasoningEffort.high],
                mapping={ReasoningEffort.none: ReasoningEffort.high, ReasoningEffort.high: ReasoningEffort.high},
                accepts_none=False,
                default=None,
            )

    @pytest.mark.parametrize(
        "mapping",
        [
            {ReasoningEffort.none: ReasoningEffort.none, ReasoningEffort.high: ReasoningEffort.high},
            {
                ReasoningEffort.low: ReasoningEffort.low,
                ReasoningEffort.high: ReasoningEffort.medium,
                ReasoningEffort.max: ReasoningEffort.high,
            },
        ],
    )
    def test_valid_mappings(self, mapping: dict[ReasoningEffort, ReasoningEffort]) -> None:
        ReasoningEffortBuilder(values=list(mapping), mapping=mapping, accepts_none=False, default=None)

    def test_build_value_maps_exposed_values(self) -> None:
        builder = ReasoningEffortBuilder(
            values=[ReasoningEffort.low, ReasoningEffort.high],
            mapping={ReasoningEffort.low: ReasoningEffort.low, ReasoningEffort.high: ReasoningEffort.medium},
            accepts_none=False,
            default=None,
        )
        assert builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.low) == ReasoningEffort.low
        assert builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.high) == ReasoningEffort.medium

    def test_build_value_maps_default(self) -> None:
        builder = ReasoningEffortBuilder(
            values=[ReasoningEffort.low, ReasoningEffort.high],
            mapping={ReasoningEffort.low: ReasoningEffort.low, ReasoningEffort.high: ReasoningEffort.medium},
            accepts_none=True,
            default=ReasoningEffort.high,
        )
        assert builder.build_value(field_name="reasoning_effort", value=None) == ReasoningEffort.medium

    def test_build_value_maps_exposed_to_model_side_values(self) -> None:
        mapping = {
            ReasoningEffort.low: ReasoningEffort.low,
            ReasoningEffort.high: ReasoningEffort.medium,
            ReasoningEffort.max: ReasoningEffort.high,
        }
        builder = ReasoningEffortBuilder(
            values=[ReasoningEffort.low, ReasoningEffort.high, ReasoningEffort.max],
            mapping=mapping,
            accepts_none=False,
            default=None,
        )
        for exposed, model_side in mapping.items():
            assert builder.build_value(field_name="reasoning_effort", value=exposed) == model_side

    def test_build_value_maps_default_to_model_side_value(self) -> None:
        # The default is the exposed "high", which the model receives as "medium".
        builder = ReasoningEffortBuilder(
            values=[ReasoningEffort.low, ReasoningEffort.high, ReasoningEffort.max],
            mapping={
                ReasoningEffort.low: ReasoningEffort.low,
                ReasoningEffort.high: ReasoningEffort.medium,
                ReasoningEffort.max: ReasoningEffort.high,
            },
            accepts_none=True,
            default=ReasoningEffort.high,
        )
        assert builder.build_value(field_name="reasoning_effort", value=None) == ReasoningEffort.medium


class TestModelSettingsBuilderMapping:
    def test_build_settings_with_mapping(self) -> None:
        builder = _remapping_builder()
        assert builder.build_settings(_request(ReasoningEffort.low)) == ModelSettings(
            reasoning_effort=ReasoningEffort.low
        )
        assert builder.build_settings(_request(ReasoningEffort.high)) == ModelSettings(
            reasoning_effort=ReasoningEffort.medium
        )
        assert builder.build_settings(_request(ReasoningEffort.max)) == ModelSettings(
            reasoning_effort=ReasoningEffort.high
        )
        assert builder.build_settings(_request(None)) == ModelSettings(reasoning_effort=ReasoningEffort.medium)

    def test_build_settings_unset_effort_resolves_to_mapped_default(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.low, ReasoningEffort.high, ReasoningEffort.max],
                mapping={
                    ReasoningEffort.low: ReasoningEffort.low,
                    ReasoningEffort.high: ReasoningEffort.medium,
                    ReasoningEffort.max: ReasoningEffort.high,
                },
                accepts_none=True,
                default=ReasoningEffort.high,
            )
        )
        assert builder.build_settings(_request(None)) == ModelSettings(reasoning_effort=ReasoningEffort.medium)

    def test_build_settings_identity_mapping(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.none, ReasoningEffort.high],
                mapping={ReasoningEffort.none: ReasoningEffort.none, ReasoningEffort.high: ReasoningEffort.high},
                accepts_none=False,
                default=None,
            )
        )
        assert builder.build_settings(_request(ReasoningEffort.none)) == ModelSettings(
            reasoning_effort=ReasoningEffort.none
        )

    def test_mapping_must_preserve_reasoning_effort_order(self) -> None:
        with pytest.raises(ValidationError, match="must preserve the order of reasoning efforts"):
            ModelSettingsBuilder(
                reasoning_effort=ReasoningEffortBuilder(
                    values=[ReasoningEffort.none, ReasoningEffort.low],
                    mapping={ReasoningEffort.none: ReasoningEffort.low, ReasoningEffort.low: ReasoningEffort.none},
                    accepts_none=False,
                    default=None,
                )
            )

    def test_mapping_preserving_reasoning_effort_order_is_authorized(self) -> None:
        ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.none, ReasoningEffort.low, ReasoningEffort.medium],
                mapping={
                    ReasoningEffort.none: ReasoningEffort.low,
                    ReasoningEffort.low: ReasoningEffort.low,
                    ReasoningEffort.medium: ReasoningEffort.medium,
                },
                accepts_none=False,
                default=None,
            )
        )

    def test_build_settings_mapping_to_unexposed_value(self) -> None:
        # The model-side value is not part of the exposed values: built values
        # are validated against the mapped values, not the exposed ones.
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.low],
                mapping={ReasoningEffort.low: ReasoningEffort.medium},
                accepts_none=False,
                default=None,
            )
        )
        assert builder.build_settings(_request(ReasoningEffort.low)) == ModelSettings(
            reasoning_effort=ReasoningEffort.medium
        )

    def test_build_settings_unexposed_effort_raises(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.high],
                mapping={ReasoningEffort.high: ReasoningEffort.high},
                accepts_none=False,
                default=None,
            )
        )
        with pytest.raises(InvalidRequestException, match="should be one of"):
            builder.build_settings(_request(ReasoningEffort.none))

    def test_build_settings_missing_effort_raises(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.none, ReasoningEffort.high],
                mapping={ReasoningEffort.none: ReasoningEffort.none, ReasoningEffort.high: ReasoningEffort.high},
                accepts_none=False,
                default=None,
            )
        )
        with pytest.raises(InvalidRequestException, match="reasoning_effort should be set for this model"):
            builder.build_settings(_request(None))

    def test_validate_settings_with_mapping(self) -> None:
        builder = _remapping_builder()
        builder.validate_settings(ModelSettings(reasoning_effort=ReasoningEffort.medium))
        with pytest.raises(
            InvalidRequestException,
            match=r"should be one of \['low', 'medium', 'high'\], got none\.",
        ):
            builder.validate_settings(ModelSettings(reasoning_effort=ReasoningEffort.none))

    def test_validate_settings_rejects_none_when_default_set(self) -> None:
        builder = _remapping_builder()
        with pytest.raises(InvalidRequestException, match="reasoning_effort should be set for this model"):
            builder.validate_settings(ModelSettings(reasoning_effort=None))

    def test_validate_settings_pass_through(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=ReasoningEffortBuilder(
                values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=True, default=ReasoningEffort.none
            )
        )
        builder.validate_settings(ModelSettings(reasoning_effort=ReasoningEffort.none))
        builder.validate_settings(ModelSettings(reasoning_effort=ReasoningEffort.high))

    def test_reasoning_effort_builder_requires_default_when_accepting_none(self) -> None:
        with pytest.raises(ValidationError, match="must declare a default"):
            ReasoningEffortBuilder(values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=True, default=None)
        with pytest.raises(ValidationError, match="must declare a default"):
            ReasoningEffortBuilder(values=[], mapping={}, accepts_none=True, default=None)

    def test_reasoning_effort_field_requires_reasoning_effort_builder(self) -> None:
        with pytest.raises(ValidationError, match="instance of ReasoningEffortBuilder"):
            ModelSettingsBuilder(
                reasoning_effort=EnumBuilder[ReasoningEffort](  # type: ignore[arg-type]
                    values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=True, default=ReasoningEffort.none
                )
            )

    def test_default_not_required_for_other_enum_builders(self) -> None:
        # The default requirement applies to ReasoningEffortBuilder, not to
        # enum builders in general, whatever the enum.
        builder = EnumBuilder[_Level](values=[_Level.low, _Level.high], accepts_none=True, default=None)
        assert builder.build_value(field_name="level", value=None) is None
        generic = EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.low, ReasoningEffort.high], accepts_none=True, default=None
        )
        assert generic.build_value(field_name="reasoning_effort", value=None) is None

    def test_builder_validated_from_dict(self) -> None:
        builder = ModelSettingsBuilder.model_validate(
            {
                "reasoning_effort": {
                    "type": "enum",
                    "accepts_none": True,
                    "default": "high",
                    "values": ["low", "high", "max"],
                    "mapping": {"low": "low", "high": "medium", "max": "high"},
                }
            }
        )
        assert isinstance(builder.reasoning_effort, ReasoningEffortBuilder)
        assert builder.build_settings(_request(ReasoningEffort.max)) == ModelSettings(
            reasoning_effort=ReasoningEffort.high
        )
        assert builder.build_settings(_request(None)) == ModelSettings(reasoning_effort=ReasoningEffort.medium)

    def test_builder_from_dict_requires_default_when_accepting_none(self) -> None:
        with pytest.raises(ValidationError, match="must declare a default"):
            ModelSettingsBuilder.model_validate(
                {
                    "reasoning_effort": {
                        "type": "enum",
                        "accepts_none": True,
                        "default": None,
                        "values": ["none", "high"],
                    }
                }
            )

    def test_model_dump_round_trip(self) -> None:
        builder = _remapping_builder()
        assert ModelSettingsBuilder.model_validate(builder.model_dump()) == builder


JSON_SCHEMA_DICT = {"type": "object", "properties": {"a": {"type": "string"}}}
JSON_OBJECT_OR_ARRAY_SCHEMA = {"anyOf": [{"type": "object"}, {"type": "array"}]}


def _req(rf: ResponseFormat) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[UserMessage(content="hi")], response_format=rf)


def test_builder_text_yields_none() -> None:
    b = ModelSettingsBuilder(json_schema=JSONSchemaBuilder())
    response_format = ResponseFormat(type=ResponseFormats.text)
    assert response_format.get_schema(purpose=SchemaRenderingMode.model_settings) is None
    assert b.build_settings(_req(response_format)).json_schema is None


def test_builder_json_yields_anyof() -> None:
    b = ModelSettingsBuilder(json_schema=JSONSchemaBuilder())
    response_format = ResponseFormat(type=ResponseFormats.json_object)
    assert response_format.get_schema(purpose=SchemaRenderingMode.model_settings) == JSON_OBJECT_OR_ARRAY_SCHEMA
    assert b.build_settings(_req(response_format)).json_schema == JSON_OBJECT_OR_ARRAY_SCHEMA


@pytest.mark.parametrize("strict", [False, True])
def test_builder_json_schema_yields_custom_schema(strict: bool) -> None:
    b = ModelSettingsBuilder(json_schema=JSONSchemaBuilder())
    response_format = ResponseFormat(
        type=ResponseFormats.json_schema,
        json_schema=JsonSchema(name="x", schema=JSON_SCHEMA_DICT, strict=strict),
    )
    assert response_format.get_schema(purpose=SchemaRenderingMode.model_settings) == JSON_SCHEMA_DICT
    assert b.build_settings(_req(response_format)).json_schema == JSON_SCHEMA_DICT


@pytest.mark.parametrize(
    ("strict", "expected_grammar_schema"),
    [(False, JSON_OBJECT_OR_ARRAY_SCHEMA), (True, JSON_SCHEMA_DICT)],
)
def test_response_format_grammar_schema_respects_strict(strict: bool, expected_grammar_schema: dict[str, Any]) -> None:
    response_format = ResponseFormat(
        type=ResponseFormats.json_schema,
        json_schema=JsonSchema(name="x", schema=JSON_SCHEMA_DICT, strict=strict),
    )

    assert response_format.get_schema(purpose=SchemaRenderingMode.grammar) == expected_grammar_schema
    assert response_format.get_schema(purpose=SchemaRenderingMode.model_settings) == JSON_SCHEMA_DICT


def test_json_schema_rejects_invalid_schema() -> None:
    with pytest.raises(ValidationError, match="123 is not valid under any of the given schemas"):
        JsonSchema(name="x", schema={"type": 123})


def test_json_schema_missing_schema_raises() -> None:
    with pytest.raises(ValidationError, match="Response format `json_schema` must define the schema"):
        ResponseFormat(type=ResponseFormats.json_schema)


def test_builder_without_json_schema_ignores_response_format() -> None:
    settings = ModelSettingsBuilder().build_settings(_req(ResponseFormat(type=ResponseFormats.json_object)))
    assert settings.json_schema is None


def test_validate_settings_rejects_invalid_json_schema() -> None:
    builder = ModelSettingsBuilder(json_schema=JSONSchemaBuilder())

    with pytest.raises(InvalidRequestException, match="json_schema is not a valid JSON schema"):
        builder.validate_settings(settings=ModelSettings(json_schema={"type": 123}))


def test_all_model_settings_and_builder_fields_match() -> None:
    model_fields = set(ModelSettings.model_fields.keys())
    builder_fields = set(ModelSettingsBuilder.model_fields.keys())

    assert model_fields == builder_fields
    assert set(ModelSettingsBuilder._SETTINGS_TO_REQUEST_FIELDS) == builder_fields
    assert set(ModelSettingsBuilder._SETTINGS_TO_REQUEST_FIELDS.values()) <= set(ChatCompletionRequest.model_fields)

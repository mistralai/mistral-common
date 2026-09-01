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
)
from mistral_common.tokens.tokenizers.model_settings_builder import (
    EnumBuilder,
    JSONSchemaBuilder,
    ModelSettingsBuilder,
)


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
                    reasoning_effort=EnumBuilder[ReasoningEffort](
                        values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=False, default=None
                    )
                ),
                ChatCompletionRequest(messages=[], reasoning_effort=ReasoningEffort.none),
                ModelSettings(reasoning_effort=ReasoningEffort.none),
            ),
            (
                ModelSettingsBuilder(
                    reasoning_effort=EnumBuilder[ReasoningEffort](
                        values=[ReasoningEffort.high], accepts_none=False, default=None
                    )
                ),
                ChatCompletionRequest(messages=[], reasoning_effort=ReasoningEffort.high),
                ModelSettings(reasoning_effort=ReasoningEffort.high),
            ),
            (
                ModelSettingsBuilder(
                    reasoning_effort=EnumBuilder[ReasoningEffort](
                        values=[ReasoningEffort.high], accepts_none=True, default=None
                    ),
                ),
                ChatCompletionRequest(messages=[], reasoning_effort=None),
                ModelSettings(reasoning_effort=None),
            ),
        ],
    )
    def test_encode_successful_validation_scenarios(
        self, builder: ModelSettingsBuilder, chat_completion_request: ChatCompletionRequest, settings: ModelSettings
    ) -> None:
        assert settings == builder.build_settings(chat_completion_request)

    def test_encode_invalid_enum_value_raises_error(self) -> None:
        builder_high_only = ModelSettingsBuilder(
            reasoning_effort=EnumBuilder[ReasoningEffort](
                values=[ReasoningEffort.high], accepts_none=False, default=None
            )
        )
        chat_completion_request = ChatCompletionRequest[UATS](messages=[], reasoning_effort=ReasoningEffort.none)
        with pytest.raises(
            InvalidRequestException,
            match=r"reasoning_effort should be one of \['high'\], got none.",
        ):
            builder_high_only.build_settings(chat_completion_request)

    def test_encode_missing_required_field_raises_error(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=EnumBuilder[ReasoningEffort](
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
            reasoning_effort=EnumBuilder[ReasoningEffort](values=[], accepts_none=True, default=None)
        )
        chat_completion_request = ChatCompletionRequest[UATS](messages=[], reasoning_effort=ReasoningEffort.high)

        with pytest.raises(InvalidRequestException, match=r"reasoning_effort not supported for this model."):
            builder.build_settings(chat_completion_request)

    def test_ignore_field(self) -> None:
        builder = ModelSettingsBuilder(reasoning_effort=None)
        chat_completion_request = ChatCompletionRequest[UATS](messages=[], reasoning_effort=ReasoningEffort.high)

        assert builder.build_settings(chat_completion_request) == ModelSettings.none()

    def test_reasoning_effort_does_not_exist(self) -> None:
        with pytest.raises(ValidationError, match=r"[type=enum, input_value='blabla', input_type=str]"):
            ModelSettingsBuilder(
                reasoning_effort=EnumBuilder[ReasoningEffort](
                    values=[ReasoningEffort.none, "blabla"],  # type: ignore[list-item]
                    accepts_none=False,
                    default=None,
                )
            )

    def test_compatibility_accept_none_default(self) -> None:
        with pytest.raises(ValidationError, match=r"Default values can only be defined for accepts_none fields"):
            ModelSettingsBuilder(
                reasoning_effort=EnumBuilder[ReasoningEffort](
                    values=[ReasoningEffort.none], accepts_none=False, default=ReasoningEffort.none
                )
            )

    def test_default_in_values(self) -> None:
        with pytest.raises(ValidationError, match=r"Default value self.default='high' is not in self.values"):
            ModelSettingsBuilder(
                reasoning_effort=EnumBuilder[ReasoningEffort](
                    values=[ReasoningEffort.none], accepts_none=True, default=ReasoningEffort.high
                )
            )

    def test_default_works(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=EnumBuilder[ReasoningEffort](
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
            reasoning_effort=EnumBuilder[ReasoningEffort](
                values=[ReasoningEffort.none], accepts_none=True, default=ReasoningEffort.none
            )
        )
        settings = ModelSettings(reasoning_effort=reasoning_effort)
        if reasoning_effort != ReasoningEffort.none:
            with pytest.raises(InvalidRequestException, match=r"reasoning_effort should be"):
                builder.validate_settings(settings)
        else:
            builder.validate_settings(settings)

    @pytest.mark.parametrize(("accepts_none"), ([False, True]))
    def test_validator_accepts_none(self, accepts_none: bool) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=EnumBuilder[ReasoningEffort](
                values=[ReasoningEffort.none], accepts_none=accepts_none, default=None
            )
        )
        settings = ModelSettings(reasoning_effort=None)
        if accepts_none:
            builder.validate_settings(settings)
        else:
            with pytest.raises(InvalidRequestException, match=r"reasoning_effort should be set for this model"):
                builder.validate_settings(settings)


JSON_SCHEMA_DICT = {"type": "object", "properties": {"a": {"type": "string"}}}


def _req(rf: ResponseFormat) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[UserMessage(content="hi")], response_format=rf)


def test_builder_text_yields_none() -> None:
    b = ModelSettingsBuilder(json_schema=JSONSchemaBuilder(accepts_none=False, default=None))
    assert b.build_settings(_req(ResponseFormat(type=ResponseFormats.text))).json_schema is None


def test_builder_json_yields_anyof() -> None:
    b = ModelSettingsBuilder(json_schema=JSONSchemaBuilder(accepts_none=False, default=None))
    assert b.build_settings(_req(ResponseFormat(type=ResponseFormats.json))).json_schema == {
        "anyOf": [{"type": "object"}, {"type": "array"}]
    }


def test_builder_json_schema_yields_custom_schema() -> None:
    b = ModelSettingsBuilder(json_schema=JSONSchemaBuilder(accepts_none=False, default=None))
    rf = ResponseFormat(type=ResponseFormats.json_schema, json_schema=JsonSchema(name="x", schema=JSON_SCHEMA_DICT))
    assert b.build_settings(_req(rf)).json_schema == JSON_SCHEMA_DICT


def test_json_schema_missing_schema_raises() -> None:
    with pytest.raises(InvalidRequestException, match="must define the schema"):
        ResponseFormat(type=ResponseFormats.json_schema).get_schema()


def test_builder_without_json_schema_ignores_response_format() -> None:
    assert ModelSettingsBuilder().build_settings(_req(ResponseFormat(type=ResponseFormats.json))).json_schema is None


def test_model_settings_json_schema_field() -> None:
    s = ModelSettings(json_schema={"type": "object"})
    assert s.json_schema == {"type": "object"}
    assert s.model_dump(exclude_none=True) == {"json_schema": {"type": "object"}}


def test_model_settings_none_excludes_json_schema() -> None:
    assert ModelSettings.none().model_dump(exclude_none=True) == {}


def test_all_model_settings_and_builder_fields_match() -> None:
    model_fields = set(ModelSettings.model_fields.keys())
    builder_fields = set(ModelSettingsBuilder.model_fields.keys())

    assert model_fields == builder_fields

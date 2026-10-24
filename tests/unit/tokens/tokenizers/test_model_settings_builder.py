import pytest
from pydantic import ValidationError

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ModelSettings, ReasoningEffort
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder


def _request(reasoning_effort: ReasoningEffort | None) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], reasoning_effort=reasoning_effort)


def _remapping_builder() -> ModelSettingsBuilder:
    r"""Builder exposing low, high and max, remapped onto the low, medium and high the model supports.

    The default is an exposed value and is mapped like any request value.
    """
    return ModelSettingsBuilder(
        reasoning_effort=EnumBuilder[ReasoningEffort](
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


class TestEnumBuilderMapping:
    def test_mapping_is_none_by_default(self) -> None:
        builder = EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=False, default=None
        )
        assert builder.mapping is None

    def test_mapping_keys_must_match_values(self) -> None:
        with pytest.raises(ValidationError, match="must match"):
            EnumBuilder[ReasoningEffort](
                values=[ReasoningEffort.none, ReasoningEffort.high],
                mapping={ReasoningEffort.high: ReasoningEffort.high},
                accepts_none=False,
                default=None,
            )
        with pytest.raises(ValidationError, match="must match"):
            EnumBuilder[ReasoningEffort](
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
        EnumBuilder[ReasoningEffort](values=list(mapping), mapping=mapping, accepts_none=False, default=None)

    def test_mapping_must_preserve_reasoning_effort_order(self) -> None:
        with pytest.raises(ValidationError, match="must preserve the order of reasoning efforts"):
            EnumBuilder[ReasoningEffort](
                values=[ReasoningEffort.none, ReasoningEffort.low],
                mapping={ReasoningEffort.none: ReasoningEffort.low, ReasoningEffort.low: ReasoningEffort.none},
                accepts_none=False,
                default=None,
            )

    def test_mapping_preserving_reasoning_effort_order_is_authorized(self) -> None:
        EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.none, ReasoningEffort.low, ReasoningEffort.medium],
            mapping={
                ReasoningEffort.none: ReasoningEffort.low,
                ReasoningEffort.low: ReasoningEffort.low,
                ReasoningEffort.medium: ReasoningEffort.medium,
            },
            accepts_none=False,
            default=None,
        )

    def test_default_must_be_one_of_the_values(self) -> None:
        with pytest.raises(ValidationError, match="is not in"):
            EnumBuilder[ReasoningEffort](
                values=[ReasoningEffort.high],
                mapping={ReasoningEffort.high: ReasoningEffort.high},
                accepts_none=True,
                default=ReasoningEffort.none,
            )

    def test_build_value_maps_exposed_values(self) -> None:
        builder = EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.low, ReasoningEffort.high],
            mapping={ReasoningEffort.low: ReasoningEffort.low, ReasoningEffort.high: ReasoningEffort.medium},
            accepts_none=False,
            default=None,
        )
        assert builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.low) == ReasoningEffort.low
        assert builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.high) == ReasoningEffort.medium

    def test_build_value_maps_default(self) -> None:
        builder = EnumBuilder[ReasoningEffort](
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
        builder = EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.low, ReasoningEffort.high, ReasoningEffort.max],
            mapping=mapping,
            accepts_none=False,
            default=None,
        )
        for exposed, model_side in mapping.items():
            assert builder.build_value(field_name="reasoning_effort", value=exposed) == model_side

    def test_build_value_maps_default_to_model_side_value(self) -> None:
        # The default is the exposed "high", which the model receives as "medium".
        builder = EnumBuilder[ReasoningEffort](
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

    def test_build_value_rejects_unexposed_value(self) -> None:
        builder = EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.high],
            mapping={ReasoningEffort.high: ReasoningEffort.high},
            accepts_none=False,
            default=None,
        )
        with pytest.raises(InvalidRequestException, match="reasoning_effort should be one of"):
            builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.none)

    def test_build_value_requires_value_when_not_accepted(self) -> None:
        builder = EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.none, ReasoningEffort.high],
            mapping={ReasoningEffort.none: ReasoningEffort.none, ReasoningEffort.high: ReasoningEffort.high},
            accepts_none=False,
            default=None,
        )
        with pytest.raises(InvalidRequestException, match="reasoning_effort should be set for this model"):
            builder.build_value(field_name="reasoning_effort", value=None)

    def test_build_value_empty_values_not_supported(self) -> None:
        builder = EnumBuilder[ReasoningEffort](values=[], mapping={}, accepts_none=True, default=None)
        with pytest.raises(InvalidRequestException, match="reasoning_effort not supported for this model"):
            builder.build_value(field_name="reasoning_effort", value=ReasoningEffort.high)


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
            reasoning_effort=EnumBuilder[ReasoningEffort](
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
            reasoning_effort=EnumBuilder[ReasoningEffort](
                values=[ReasoningEffort.none, ReasoningEffort.high],
                mapping={ReasoningEffort.none: ReasoningEffort.none, ReasoningEffort.high: ReasoningEffort.high},
                accepts_none=False,
                default=None,
            )
        )
        assert builder.build_settings(_request(ReasoningEffort.none)) == ModelSettings(
            reasoning_effort=ReasoningEffort.none
        )

    def test_build_settings_mapping_to_unexposed_value(self) -> None:
        # The model-side value is not part of the exposed values: built values
        # are validated against the mapped values, not the exposed ones.
        builder = ModelSettingsBuilder(
            reasoning_effort=EnumBuilder[ReasoningEffort](
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
            reasoning_effort=EnumBuilder[ReasoningEffort](
                values=[ReasoningEffort.high],
                mapping={ReasoningEffort.high: ReasoningEffort.high},
                accepts_none=False,
                default=None,
            )
        )
        with pytest.raises(InvalidRequestException, match="reasoning_effort should be one of"):
            builder.build_settings(_request(ReasoningEffort.none))

    def test_build_settings_missing_effort_raises(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=EnumBuilder[ReasoningEffort](
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
            match=r"reasoning_effort should be one of \['low', 'medium', 'high'\], got none\.",
        ):
            builder.validate_settings(ModelSettings(reasoning_effort=ReasoningEffort.none))

    def test_validate_settings_rejects_none_when_default_set(self) -> None:
        builder = _remapping_builder()
        with pytest.raises(InvalidRequestException, match="reasoning_effort should be set for this model"):
            builder.validate_settings(ModelSettings(reasoning_effort=None))

    def test_validate_settings_pass_through(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=EnumBuilder[ReasoningEffort](
                values=[ReasoningEffort.none, ReasoningEffort.high], accepts_none=True, default=None
            )
        )
        builder.validate_settings(ModelSettings(reasoning_effort=ReasoningEffort.none))
        builder.validate_settings(ModelSettings(reasoning_effort=None))

    def test_validate_settings_empty_mapping(self) -> None:
        builder = ModelSettingsBuilder(
            reasoning_effort=EnumBuilder[ReasoningEffort](values=[], mapping={}, accepts_none=True, default=None)
        )
        with pytest.raises(InvalidRequestException, match="reasoning_effort not supported for this model"):
            builder.validate_settings(ModelSettings(reasoning_effort=ReasoningEffort.high))

    def test_builder_validated_from_dict(self) -> None:
        builder = ModelSettingsBuilder.model_validate(
            {
                "reasoning_effort": {
                    "type": "enum",
                    "accepts_none": True,
                    "default": None,
                    "values": ["low", "high", "max"],
                    "mapping": {"low": "low", "high": "medium", "max": "high"},
                }
            }
        )
        assert isinstance(builder.reasoning_effort, EnumBuilder)
        assert builder.build_settings(_request(ReasoningEffort.max)) == ModelSettings(
            reasoning_effort=ReasoningEffort.high
        )
        assert builder.build_settings(_request(None)) == ModelSettings(reasoning_effort=None)

    def test_model_dump_round_trip(self) -> None:
        builder = _remapping_builder()
        assert ModelSettingsBuilder.model_validate(builder.model_dump()) == builder

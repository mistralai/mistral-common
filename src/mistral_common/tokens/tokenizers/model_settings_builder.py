import itertools
from collections.abc import Mapping
from enum import Enum, StrEnum, auto
from typing import Any, ClassVar, Generic, Literal, TypeAlias, TypeVar, final

from pydantic import model_validator

from mistral_common.base import MistralBase
from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    ModelSettings,
    ReasoningEffort,
    ResponseFormat,
    SchemaRenderingMode,
)
from mistral_common.utils.json_utils import validate_json_schema_by_draft7


class ValidatorType(StrEnum):
    r"""Enumeration of validator types.

    Attributes:
        ENUM: Indicates that the validator is for enum values.
        JSON_SCHEMA: Indicates that the validator is for JSON schema values.
    """

    ENUM = auto()
    JSON_SCHEMA = auto()


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
JSONSchemaDict: TypeAlias = dict[str, Any] | None


class FieldBuilder(MistralBase, Generic[InputT, OutputT]):
    r"""Base class for field builders.

    `InputT` is the request field type, `OutputT` is the converted `ModelSettings` field type.

    Attributes:
        type: The type of validator (e.g., ENUM).
        accepts_none: Whether the field accepts `None` as a valid value in the request.
        default: The default value to use when the field is `None`, if `accepts_none` is `True`.
    """

    type: ValidatorType
    accepts_none: bool
    default: OutputT | None

    @model_validator(mode="after")
    def validate_default_accept_none(self) -> "FieldBuilder":
        r"""Ensure a default value is only set when `accepts_none` is `True`."""
        if not self.accepts_none and self.default is not None:
            raise ValueError(
                f"Default values can only be defined for accepts_none fields {self.accepts_none=} {self.default=}"
            )
        return self

    def _validate_built_value(self, field_name: str, value: OutputT) -> None:
        r"""Validate a non-`None` built value. Must be implemented by subclasses.

        Args:
            field_name: Name of the field being validated, used in errors.
            value: The built value to validate.

        Raises:
            NotImplementedError: Always, as this is an abstract method.
        """
        raise NotImplementedError(f"{field_name} is not supported")

    def _convert(self, field_name: str, input_value: InputT) -> OutputT:
        r"""Convert an input value into its built value.

        Args:
            field_name: Name of the field being converted, used in errors.
            input_value: The request-side value to convert.

        Returns:
            The converted value.

        Raises:
            NotImplementedError: Always, as this is an abstract method.
        """
        raise NotImplementedError

    def _build_from_optional(self, field_name: str, value: InputT | None) -> OutputT | None:
        r"""Resolve an optional value, substituting the default if value is `None`.

        Returns:
            The converted value, or the field's default when the value is `None`
            and the field accepts `None`.

        Raises:
            InvalidRequestException: If value is `None` and the field does not accept `None`.
        """
        if value is None:
            if not self.accepts_none:
                raise InvalidRequestException(f"{field_name} should be set for this model.")
            return self.default
        return self._convert(field_name=field_name, input_value=value)

    def validate_built_value(self, field_name: str, value: OutputT | None) -> None:
        r"""Validate a fully built value, including `None` checks.

        Raises:
            InvalidRequestException: If value is `None` when not permitted, or fails subclass validation.
        """
        if value is None:
            if not (self.accepts_none and self.default is None):
                raise InvalidRequestException(f"{field_name} should be set for this model.")
        else:
            self._validate_built_value(field_name, value)

    @final
    def build_value(self, field_name: str, value: InputT | None) -> OutputT | None:
        r"""Resolve and validate a field value, returning the final built result.

        Returns:
            The resolved, validated value, or None when the field accepts None and
            the input was None.

        Raises:
            InvalidRequestException: If the value is invalid or missing when required.
        """
        built_value = self._build_from_optional(field_name, value)
        self.validate_built_value(field_name, built_value)
        return built_value


E = TypeVar("E", bound=Enum)
EffortT = TypeVar("EffortT")


def validate_reasoning_effort_mapping_order(mapping: Mapping[EffortT, EffortT]) -> None:
    r"""Ensure a reasoning effort mapping preserves the order of reasoning efforts.

    When the exposed values are sorted by increasing effort, their model-side
    values must not decrease: `none -> low, low -> low, medium -> medium` is
    authorized, but `none -> low, low -> none` is not. Keys and values may be
    `ReasoningEffort` members or their string values: the enum constructor
    accepts both forms, so the mapping is normalized to members first.

    Args:
        mapping: Mapping from exposed reasoning efforts to model-side efforts.

    Raises:
        ValueError: If the mapping does not preserve the order of reasoning efforts.
    """
    efforts = {ReasoningEffort(key): ReasoningEffort(value) for key, value in mapping.items()}
    rank = list(ReasoningEffort).index
    model_ranks = [rank(efforts[exposed]) for exposed in sorted(efforts, key=rank)]
    if any(lower > higher for lower, higher in itertools.pairwise(model_ranks)):
        raise ValueError(
            f"Mapping {mapping} must preserve the order of reasoning efforts: "
            "model-side values must not decrease as exposed values increase."
        )


class EnumBuilder(FieldBuilder[E, E]):
    r"""Builder for enum fields.

    This class validates that enum fields contain only authorized values.
    It rejects duplicate values during initialization and ensures the
    allowed values list is non-empty when `None` is not accepted.

    Attributes:
        type: The type of validator (always ENUM for this class).
        values: List of allowed values.
    """

    type: ValidatorType = ValidatorType.ENUM
    values: list[E]

    @property
    def _exposed_values(self) -> list[E]:
        r"""Values exposed on requests."""
        return self.values

    def _validate_allowed_value(self, value: E) -> None:
        r"""Check that a value is exposed, with a specific error when nothing is supported.

        Args:
            value: The value to check.

        Raises:
            InvalidRequestException: If no value is supported, or the value is not exposed.
        """
        exposed = self._exposed_values
        if len(exposed) == 0:
            raise InvalidRequestException("not supported for this model.")
        if value not in exposed:
            raise InvalidRequestException(f"should be one of {exposed}, got {value}.")

    def _convert(self, field_name: str, input_value: E) -> E:
        r"""Validate an exposed value and return it as the built value.

        Args:
            field_name: Name of the field being converted, used in errors.
            input_value: The value to validate.

        Returns:
            The built value.

        Raises:
            InvalidRequestException: If the value is not one of the exposed values.
        """
        self._validate_allowed_value(value=input_value)
        return input_value

    def _build_from_optional(self, field_name: str, value: E | None) -> E | None:
        r"""Resolve an optional value, resolving the default like any exposed value.

        Unlike the base implementation, the default is not returned as-is:
        it goes through `_convert`, so subclasses translating values translate
        the default too.
        """
        if value is None and self.default is not None:
            return self._convert(field_name=field_name, input_value=self.default)
        return super()._build_from_optional(field_name=field_name, value=value)

    @model_validator(mode="after")
    def validate_unique_values(self) -> "EnumBuilder":
        r"""Ensure no duplicate values are present in the allowed values list."""
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"Duplicate values in {self.values=}")
        return self

    @model_validator(mode="after")
    def validate_empty_list(self) -> "EnumBuilder":
        r"""Ensure the allowed values list is non-empty when `None` is not accepted."""
        if len(self.values) == 0 and not self.accepts_none:
            raise ValueError(f"Empty list of values for {self.values=} while not accepts_none.")
        return self

    @model_validator(mode="after")
    def validate_default(self) -> "EnumBuilder":
        r"""Ensure the default value, if set, is among the allowed values."""
        if self.default is not None and self.default not in self.values:
            raise ValueError(f"Default value {self.default=} is not in {self.values=}.")
        return self

    def _validate_built_value(self, field_name: str, value: E) -> None:
        r"""Check that the built value is one the builder can produce.

        Raises:
            InvalidRequestException: If the value is unsupported or not allowed.
        """
        self._validate_allowed_value(value=value)


class ReasoningEffortBuilder(EnumBuilder[ReasoningEffort]):
    r"""Builder for the reasoning effort field.

    Carries the reasoning-effort-specific remapping and invariants so they
    hold wherever the builder is constructed, standalone or from tokenizer
    JSON:

    - When `mapping` is configured, the allowed values are the ones exposed
      on requests and the built value is their translation through the
      mapping, so `ModelSettings` holds model-side values. Requests and the
      default are validated against the exposed `values`.
    - The mapping must preserve the order of reasoning efforts.
    - A builder that accepts `None` declares a default: `None` is an allowed
      request value, never the default itself.

    Attributes:
        mapping: Mapping from exposed values to model-side values, or `None`
            to use the exposed values as-is.
    """

    mapping: dict[ReasoningEffort, ReasoningEffort] | None = None

    @property
    def _exposed_values(self) -> list[ReasoningEffort]:
        r"""Values exposed on requests: the mapping keys when a mapping is configured."""
        if self.mapping is None:
            return super()._exposed_values
        return list(dict.fromkeys(self.mapping))

    @property
    def _built_values(self) -> list[ReasoningEffort]:
        r"""Built values: the mapped values when a mapping is configured, the exposed values otherwise."""
        if self.mapping is None:
            return super()._exposed_values
        return list(dict.fromkeys(self.mapping.values()))

    def _convert(self, field_name: str, input_value: ReasoningEffort) -> ReasoningEffort:
        r"""Translate a validated exposed value into its model-side value.

        Args:
            field_name: Name of the field being converted, used in errors.
            input_value: The exposed value to validate and translate.

        Returns:
            The model-side value.

        Raises:
            InvalidRequestException: If the value is not exposed, or has no translation.
        """
        exposed = super()._convert(field_name=field_name, input_value=input_value)
        if self.mapping is None:
            return exposed
        try:
            return self.mapping[exposed]
        except KeyError:
            raise InvalidRequestException(f"should be one of {list(self.mapping)}, got {input_value}.") from None

    def _validate_built_value(self, field_name: str, value: ReasoningEffort) -> None:
        r"""Check that the built value is one the builder can produce.

        The built value is a model-side value when a mapping is configured,
        an exposed value otherwise.

        Raises:
            InvalidRequestException: If the value is unsupported or not allowed.
        """
        allowed = self._built_values
        if len(allowed) == 0:
            raise InvalidRequestException("not supported for this model.")
        if value not in allowed:
            raise InvalidRequestException(f"should be one of {allowed}, got {value}.")

    @model_validator(mode="after")
    def validate_mapping(self) -> "ReasoningEffortBuilder":
        r"""Ensure the mapping translates exactly the allowed values."""
        if self.mapping is not None and set(self.mapping) != set(self.values):
            raise ValueError(f"Mapping keys {list(self.mapping)=} must match {self.values=}.")
        return self

    @model_validator(mode="after")
    def validate_mapping_order(self) -> "ReasoningEffortBuilder":
        r"""Ensure the mapping preserves the order of reasoning efforts.

        The reasoning effort ladder is ordered: model-side values may not
        decrease as exposed values increase.
        """
        if self.mapping is not None:
            validate_reasoning_effort_mapping_order(self.mapping)
        return self

    @model_validator(mode="after")
    def validate_default_required(self) -> "ReasoningEffortBuilder":
        r"""Ensure the builder declares a default when it accepts `None`.

        A request may leave the effort unset, but the builder must resolve it
        to a declared default: `None` is an allowed request value, never the
        default itself.
        """
        if self.accepts_none and self.default is None:
            raise ValueError("The reasoning effort builder must declare a default when it accepts None.")
        return self


class JSONSchemaBuilder(FieldBuilder[ResponseFormat, JSONSchemaDict]):
    r"""Builder for response-format JSON schema fields.

    Converts a `ResponseFormat` into the JSON schema encoded in model settings.

    Attributes:
        type: The type of validator (always JSON_SCHEMA for this class).
        accepts_none: Always `False` because the response format is always present on requests.
        default: Always `None`, no default schema is supported.
    """

    type: ValidatorType = ValidatorType.JSON_SCHEMA
    accepts_none: Literal[False] = False
    default: None = None

    def validate_built_value(self, field_name: str, value: JSONSchemaDict) -> None:
        r"""Validate a built schema value, allowing `None`.

        Unlike the base implementation, `None` skips validation: a response
        format without a schema (e.g. `text`) legitimately builds to `None`.

        Raises:
            InvalidRequestException: If the built schema is not a valid Draft 7 JSON schema.
        """
        self._validate_built_value(field_name=field_name, value=value)

    def _convert(self, field_name: str, input_value: ResponseFormat) -> JSONSchemaDict:
        r"""Render the response format's schema for model-settings encoding."""
        return input_value.get_schema(purpose=SchemaRenderingMode.model_settings)

    def _validate_built_value(self, field_name: str, value: JSONSchemaDict) -> None:
        r"""Validate the built schema against Draft 7 when present.

        Raises:
            InvalidRequestException: If the schema is not a valid Draft 7 JSON schema.
        """
        if value is None:
            return
        try:
            validate_json_schema_by_draft7(value=value)
        except ValueError as e:
            raise InvalidRequestException(f"{field_name} is not a valid JSON schema: {e}") from e


class ModelSettingsBuilder(MistralBase):
    r"""Builder for ModelSettings to ensure only authorized values are used.

    This class validates that model settings contain only authorized values
    for each field. It enforces a strict field matching approach where:
    - All fields in model settings must have corresponding builder fields.
    - All builder fields must have corresponding model settings fields.
    - Validation is performed only on matching fields.
    - Clear error messages are provided for field mismatches.

    Attributes:
        reasoning_effort: Builder for the allowed ReasoningEffort values, or `None` if unsupported.
        json_schema: Builder for the response-format JSON schema, or `None` if unsupported.
    """

    _SETTINGS_TO_REQUEST_FIELDS: ClassVar[dict[str, str]] = {
        "reasoning_effort": "reasoning_effort",
        "json_schema": "response_format",
    }

    reasoning_effort: ReasoningEffortBuilder | None = None
    json_schema: JSONSchemaBuilder | None = None

    @staticmethod
    def none() -> "ModelSettingsBuilder":
        r"""Return a ModelSettingsBuilder with no field builders configured.

        Any model settings field set on a request will be rejected by a builder
        returned from this method.

        Returns:
            A builder with all field builders set to `None`.
        """
        return ModelSettingsBuilder()

    def build_settings(self, request: ChatCompletionRequest) -> ModelSettings:
        r"""Build and validate a ModelSettings instance from a raw request.

        Iterates over all known fields, applies the corresponding builder, and
        constructs a validated ModelSettings object.

        Args:
            request: The incoming chat completion request.

        Returns:
            A validated ModelSettings instance.

        Raises:
            InvalidRequestException: If any field value is invalid or unsupported.
        """
        dict_settings = {}
        for field_name in ModelSettingsBuilder.model_fields:
            # test_all_model_settings_and_builder_fields_match keeps these fields and the request mapping in sync.
            value = getattr(request, self._SETTINGS_TO_REQUEST_FIELDS[field_name])
            field_builder: FieldBuilder | None = getattr(self, field_name)
            if field_builder is not None:
                dict_settings[field_name] = field_builder.build_value(field_name, value)

        return ModelSettings.model_validate(dict_settings)

    def validate_settings(self, settings: ModelSettings) -> None:
        r"""Validate that all fields in a ModelSettings instance match the configured builders.

        Ensures that fields without a builder are unset, and fields with a builder
        hold a value that passes validation.

        Args:
            settings: The ModelSettings instance to validate.

        Raises:
            InvalidRequestException: If a field is set but has no builder, or fails its builder.
        """
        for field_name in ModelSettingsBuilder.model_fields:
            value = getattr(settings, field_name)
            field_builder: FieldBuilder | None = getattr(self, field_name)
            if field_builder is None:
                if value is not None:
                    raise InvalidRequestException(f"{field_name} not supported for this model")
            else:
                field_builder.validate_built_value(field_name, value)

from enum import Enum
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


class ValidatorType(str, Enum):
    r"""Enumeration of validator types.

    Attributes:
        ENUM: Indicates that the validator is for enum values.
        JSON_SCHEMA: Indicates that the validator is for JSON schema values.
    """

    ENUM = "enum"
    JSON_SCHEMA = "json_schema"


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
JSONSchemaDict: TypeAlias = dict[str, Any] | None


class FieldBuilder(MistralBase, Generic[InputT, OutputT]):
    r"""Base class for field builders.

    `InputT` is the request field type, `OutputT` is the converted `ModelSettings` field type.

    Attributes:
        type: The type of validator (e.g., ENUM).
        accepts_none: Whether the field accepts None as a valid value in the request.
        default: The default value to use when the field is None, if accepts_none is True.
    """

    type: ValidatorType
    accepts_none: bool
    default: OutputT | None

    @model_validator(mode="after")
    def validate_default_accept_none(self) -> "FieldBuilder":
        r"""Ensure a default value is only set when accepts_none is True."""
        if not self.accepts_none and self.default is not None:
            raise ValueError(
                f"Default values can only be defined for accepts_none fields {self.accepts_none=} {self.default=}"
            )
        return self

    def _convert(self, input_value: InputT) -> OutputT:
        r"""Convert a request value to the model-settings value."""
        raise NotImplementedError

    def _build_from_optional(self, field_name: str, input_value: InputT | None) -> OutputT | None:
        r"""Resolve an optional value, substituting the default if input is None.

        Raises:
            InvalidRequestException: If input is None and the field does not accept None.
        """
        if input_value is None:
            if not self.accepts_none:
                raise InvalidRequestException(f"{field_name} should be set for this model.")
            return self.default
        return self._convert(input_value=input_value)

    def validate_built_value(self, field_name: str, built_value: OutputT | None) -> None:
        r"""Validate a fully built value. Must be implemented by subclasses."""
        raise NotImplementedError

    @final
    def build_value(self, field_name: str, input_value: InputT | None) -> OutputT | None:
        r"""Resolve and validate a field value, returning the final built result.

        Raises:
            InvalidRequestException: If the value is invalid or missing when required.
        """
        built_value = self._build_from_optional(field_name, input_value)
        self.validate_built_value(field_name, built_value)
        return built_value


E = TypeVar("E", bound=Enum)


class EnumBuilder(FieldBuilder[E, E]):
    r"""Builder for enum fields.

    This class validates that enum fields contain only authorized values.
    It rejects duplicate values during initialization and ensures the
    allowed values list is non-empty when None is not accepted.

    Attributes:
        type: The type of validator (always ENUM for this class).
        values: List of allowed enum values.
    """

    type: ValidatorType = ValidatorType.ENUM
    values: list[E]

    def _convert(self, input_value: E) -> E:
        r"""Enum builders pass values through unchanged."""
        return input_value

    @model_validator(mode="after")
    def validate_unique_values(self) -> "EnumBuilder":
        r"""Ensure no duplicate values are present in the allowed values list."""
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"Duplicate values in {self.values=}")
        return self

    @model_validator(mode="after")
    def validate_empty_list(self) -> "EnumBuilder":
        r"""Ensure the allowed values list is non-empty when None is not accepted."""
        if len(self.values) == 0 and not self.accepts_none:
            raise ValueError(f"Empty list of values for {self.values=} while not accepts_none.")
        return self

    @model_validator(mode="after")
    def validate_default(self) -> "EnumBuilder":
        r"""Ensure the default value, if set, is among the allowed values."""
        if self.default is not None and self.default not in self.values:
            raise ValueError(f"Default value {self.default=} is not in {self.values=}.")
        return self

    def validate_built_value(self, field_name: str, built_value: E | None) -> None:
        r"""Check that the built value is one of the allowed enum values.

        Raises:
            InvalidRequestException: If unset when required, unsupported, or not allowed.
        """
        if built_value is None:
            if not (self.accepts_none and self.default is None):
                raise InvalidRequestException(f"{field_name} should be set for this model.")
        elif len(self.values) == 0:
            raise InvalidRequestException(f"{field_name} not supported for this model.")
        elif built_value not in self.values:
            raise InvalidRequestException(f"{field_name} should be one of {self.values}, got {built_value}.")


class JSONSchemaBuilder(FieldBuilder[ResponseFormat, JSONSchemaDict]):
    r"""Converts a `ResponseFormat` into a JSON-schema dict for model settings.

    Attributes:
        type: The type of validator (always JSON_SCHEMA for this class).
        accepts_none: Always False, as a response format is always present on the request.
        default: Always None, no default schema is supported.
    """

    type: ValidatorType = ValidatorType.JSON_SCHEMA
    accepts_none: Literal[False]
    default: None

    def _convert(self, input_value: ResponseFormat) -> JSONSchemaDict:
        r"""Render the response format's schema for model-settings encoding."""
        return input_value.get_schema(purpose=SchemaRenderingMode.model_settings)

    def validate_built_value(self, field_name: str, built_value: JSONSchemaDict) -> None:
        r"""Validate the built schema against Draft 7 when present."""
        if built_value is not None:
            validate_json_schema_by_draft7(built_value)


class ModelSettingsBuilder(MistralBase):
    r"""Builder for ModelSettings to ensure only authorized values are used.

    This class validates that model settings contain only authorized values
    for each field. It enforces a strict field matching approach where:
    - All fields in model settings must have corresponding builder fields.
    - All builder fields must have corresponding model settings fields.
    - Validation is performed only on matching fields.
    - Clear error messages are provided for field mismatches.

    Attributes:
        reasoning_effort: Builder for the allowed ReasoningEffort values, or None if unsupported.
        json_schema: Builder for the response-format JSON schema, or None if unsupported.
    """

    _SETTINGS_TO_CONV_FIELDS_MAP: ClassVar[dict[str, str]] = {
        "reasoning_effort": "reasoning_effort",
        "json_schema": "response_format",
    }

    reasoning_effort: EnumBuilder[ReasoningEffort] | None = None
    json_schema: JSONSchemaBuilder | None = None

    @staticmethod
    def none() -> "ModelSettingsBuilder":
        r"""Return a ModelSettingsBuilder with no field builders configured."""
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
            # We have a CI test to ensure all fields match between ModelSettings and ModelSettingsBuilder.
            value = getattr(request, self._SETTINGS_TO_CONV_FIELDS_MAP[field_name])
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

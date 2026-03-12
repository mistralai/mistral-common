from enum import Enum
from typing import Any, Generic, TypeVar, final

from pydantic import model_validator

from mistral_common.base import MistralBase
from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ModelSettings, ReasoningEffort


class ValidatorType(str, Enum):
    r"""Enumeration of validator types.

    Attributes:
        ENUM: Indicates that the validator is for enum values.
    """

    ENUM = "enum"


T = TypeVar("T")


class FieldBuilder(MistralBase, Generic[T]):
    r"""Base class for field builders.

    This class serves as the base for all field builders in the validation framework.
    It ensures that all builders have a type attribute that specifies the kind of
    validation being performed.

    Attributes:
        type: The type of validator (e.g., ENUM).
        accepts_none: Whether the field accepts None as a valid value in the request.
        default: The default value to use when the field is None, if accepts_none is True.
    """

    type: ValidatorType
    accepts_none: bool
    default: T | None

    @model_validator(mode="after")
    def validate_default_accept_none(self) -> "FieldBuilder":
        r"""Ensure a default value is only set when accepts_none is True."""
        if not self.accepts_none and self.default is not None:
            raise ValueError(
                f"Default values can only be defined for accepts_none fields {self.accepts_none=} {self.default=}"
            )
        return self

    def _validate_built_value(self, field_name: str, value: Any) -> None:
        r"""Validate a non-None built value. Must be implemented by subclasses."""
        raise NotImplementedError(f"{field_name} is not supported")

    def _build_from_optional(self, field_name: str, value: T | None) -> T | None:
        r"""Resolve an optional value, substituting the default if value is None.

        Raises:
            InvalidRequestException: If value is None and the field does not accept None.
        """
        if value is None:
            if not self.accepts_none:
                raise InvalidRequestException(f"{field_name} should be set for this model.")
            return self.default
        return value

    @final
    def validate_built_value(self, field_name: str, value: Any) -> None:
        r"""Validate a fully built value, including None checks.

        Raises:
            InvalidRequestException: If value is None when not permitted, or fails subclass validation.
        """
        if value is None:
            if not (self.accepts_none and self.default is None):
                raise InvalidRequestException(f"{field_name} should be set for this model.")
        else:
            self._validate_built_value(field_name, value)

    @final
    def build_value(self, field_name: str, value: T | None) -> T | None:
        r"""Resolve and validate a field value, returning the final built result.

        Raises:
            InvalidRequestException: If the value is invalid or missing when required.
        """
        value = self._build_from_optional(field_name, value)
        self.validate_built_value(field_name, value)
        return value


E = TypeVar("E", bound=Enum)


class EnumBuilder(FieldBuilder[E]):
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

    def _validate_built_value(self, field_name: str, value: Any) -> None:
        r"""Check that value is one of the allowed enum values.

        Raises:
            InvalidRequestException: If no values are allowed, or value is not in the allowed list.
        """
        if len(self.values) == 0:
            raise InvalidRequestException(f"{field_name} not supported for this model.")
        if value not in self.values:
            raise InvalidRequestException(f"{field_name} should be one of {self.values}, got {value}.")


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
    """

    reasoning_effort: EnumBuilder[ReasoningEffort] | None = None

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
            # We have a CI test to ensure all fields match between ModelSettings and ModelSettingsEncoder.
            value = getattr(request, field_name)
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

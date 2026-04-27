from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class MistralBase(BaseModel):
    r"""Base class for all Mistral Pydantic models.

    Forbids extra attributes, validates default values and use enum values.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True, use_enum_values=True)

    @classmethod
    def _filter_cls_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        r"""Filter a dictionary to only include keys that are valid model fields."""
        return {k: v for k, v in data.items() if k in cls.model_fields}

    @classmethod
    def model_validate_ignore_extra(cls, data: dict[str, Any]) -> Self:
        r"""Build the model from the data after filtering out keys not in the model fields."""
        return cls.model_validate(cls._filter_cls_fields(data))

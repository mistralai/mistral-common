from typing import Any

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

from pydantic import BaseModel, ConfigDict


class MistralBase(BaseModel):
    r"""Base class for all Mistral Pydantic models.

    Forbids extra attributes, validates default values and use enum values.

    Methods converting from external formats (e.g. `from_openai`) tolerate
    unknown fields by passing `extra="ignore"` to `model_validate`.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True, use_enum_values=True)

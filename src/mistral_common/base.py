from pydantic import BaseModel, ConfigDict


class MistralBase(BaseModel):
    r"""Base class for all Mistral Pydantic models.

    Forbids extra attributes, validates default values and use enum values.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True, use_enum_values=True)

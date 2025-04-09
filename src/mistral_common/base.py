from pydantic import BaseModel, ConfigDict


class MistralBase(BaseModel):
    """
    Base class for all Mistral Pydantic models.
    """

    model_config = ConfigDict(validate_default=True, use_enum_values=True)

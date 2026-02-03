from pydantic import Field

from mistral_common.base import MistralBase


class BaseCompletionRequest(MistralBase):
    """Base class for completion requests.

    Attributes:
        temperature: Sampling temperature to use, between 0 and 1. Higher values like 0.8 will make the output more
            random, while lower values like 0.2 will make it more focused and deterministic.
        top_p: Nucleus sampling parameter, top-p probability mass, between 0 and 1.
        max_tokens: Maximum number of tokens to generate.
        random_seed: Random seed for reproducibility.

    Examples:
        >>> request = BaseCompletionRequest(temperature=0.7, top_p=0.9, max_tokens=100, random_seed=42)
    """

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=0)
    random_seed: int | None = Field(default=None, ge=0)

from pydantic import Field

from mistral_common.base import MistralBase


class BaseCompletionRequest(MistralBase):
    r"""Base class for completion requests.

    Holds sampling parameters shared by all completion request types
    (chat, FIM, transcription, speech).

    Attributes:
        temperature: Sampling temperature in [0, 1]. Higher values (e.g., 0.8) make
            the output more random; lower values (e.g., 0.2) make it more focused
            and deterministic. Defaults to 0.7.
        top_p: Nucleus sampling probability mass in [0, 1]. The model samples from
            the smallest set of tokens whose cumulative probability exceeds `top_p`.
            Defaults to 1.0 (no truncation).
        max_tokens: Maximum number of tokens to generate. If `None`, the model
            generates until a stop condition (e.g., EOS) is reached.
        random_seed: Seed for reproducible sampling. If `None`, sampling is not
            reproducible.

    Examples:
        >>> request = BaseCompletionRequest(temperature=0.7, top_p=0.9, max_tokens=100, random_seed=42)
    """

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=0)
    random_seed: int | None = Field(default=None, ge=0)

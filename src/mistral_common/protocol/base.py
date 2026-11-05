from collections.abc import Iterable
from typing import Any, ClassVar

from pydantic import Field, ValidationInfo, model_validator

from mistral_common.base import MistralBase
from mistral_common.deprecation import warn_once

# Validation context key set by `from_openai` methods: OpenAI requests carry sampling
# parameters, so converting them must not warn about the deprecated fields.
_FROM_OPENAI = "from_openai"

_SAMPLING_FIELDS = ("temperature", "top_p", "max_tokens", "random_seed")
_UNUSED_FIELD_HINT = "mistral-common does not use it; pass it to your inference engine instead."


def _deprecations(owner: str, fields: Iterable[str], hint: str = _UNUSED_FIELD_HINT) -> dict[str, str]:
    r"""Map each field to its deprecation message."""
    return {field: f"`{owner}.{field}` is deprecated and will be removed in 1.15.0. {hint}" for field in fields}


class BaseCompletionRequest(MistralBase):
    r"""Base class for completion requests.

    Holds sampling parameters shared by all completion request types
    (chat, FIM, transcription, speech). mistral-common does not use them: the
    subclasses deprecate them for removal in 1.15.0.

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

    # Deprecated fields and their warning messages. Passing one explicitly warns once,
    # except through `from_openai`.
    _deprecated_fields: ClassVar[dict[str, str]] = {}

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=0)
    random_seed: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated_fields(cls, values: Any, info: ValidationInfo) -> Any:
        r"""Warn when a deprecated field is passed explicitly."""
        if isinstance(values, dict) and not (info.context or {}).get(_FROM_OPENAI):
            for field, message in cls._deprecated_fields.items():
                if field in values:
                    warn_once(key=message, message=message, category=DeprecationWarning, stacklevel=4)
        return values

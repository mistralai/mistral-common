from typing import Any, ClassVar

from pydantic import Field, ValidationInfo, model_validator

from mistral_common.base import MistralBase
from mistral_common.deprecation import warn_once

# Validation context key set by `from_openai` methods: OpenAI requests carry sampling
# parameters, so converting them must not warn about the deprecated fields.
_FROM_OPENAI = "from_openai"

_SAMPLING_FIELDS = ("temperature", "top_p", "max_tokens", "random_seed")

# Request fields deprecated for removal in 1.15.0, by class. Each class warns for its
# own entry, and the single warning message lists them all.
_DEPRECATED_FIELDS: dict[str, tuple[str, ...]] = {
    "ChatCompletionRequest": (*_SAMPLING_FIELDS, "truncate_for_context_length"),
    "FIMRequest": _SAMPLING_FIELDS,
    "InstructRequest": ("truncate_at_max_tokens",),
    "SpeechRequest": ("id", "model", *_SAMPLING_FIELDS),
    "TranscriptionRequest": ("id", "model", *_SAMPLING_FIELDS, "strict_audio_validation"),
}

_DEPRECATED_FIELDS_MESSAGE = (
    "The following mistral-common request fields are deprecated and will be removed in 1.15.0:\n"
    + "".join(f"- {owner}: {', '.join(fields)}\n" for owner, fields in _DEPRECATED_FIELDS.items())
    + "mistral-common does not use id, model, temperature, top_p, max_tokens or random_seed: pass them to your "
    "inference engine instead. Instead of truncate_for_context_length and truncate_at_max_tokens, truncate the "
    "conversation before encoding. strict_audio_validation has no effect.\n"
    "If you do not set these fields through mistral-common, you can ignore this warning."
)


def _warn_once_for_deprecated_fields(fields: tuple[str, ...], values: Any, stacklevel: int) -> None:
    r"""Warn, once per process for all fields, if `values` sets one of the deprecated `fields`.

    Args:
        fields: The deprecated fields of the caller's class.
        values: The raw input being validated.
        stacklevel: Stack level as for `warnings.warn` called in the caller.
    """
    if isinstance(values, dict) and any(field in values for field in fields):
        warn_once(
            key=_DEPRECATED_FIELDS_MESSAGE,
            message=_DEPRECATED_FIELDS_MESSAGE,
            category=DeprecationWarning,
            stacklevel=stacklevel + 2,
        )


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

    # Deprecated fields. Passing one explicitly warns, once per process for all
    # deprecated request fields, except through `from_openai`.
    _deprecated_fields: ClassVar[tuple[str, ...]] = ()

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=0)
    random_seed: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated_fields(cls, values: Any, info: ValidationInfo) -> Any:
        r"""Warn when a deprecated field is passed explicitly."""
        if not (info.context or {}).get(_FROM_OPENAI):
            _warn_once_for_deprecated_fields(cls._deprecated_fields, values, stacklevel=3)
        return values

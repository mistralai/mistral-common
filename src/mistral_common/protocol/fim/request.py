from typing import ClassVar

from mistral_common.protocol.base import _DEPRECATED_FIELDS, BaseCompletionRequest


class FIMRequest(BaseCompletionRequest):
    r"""A valid Fill in the Middle completion request to be tokenized.

    FIM requests ask the model to complete text between a prefix and a suffix,
    e.g., for code completion in an editor.

    Attributes:
        temperature: Deprecated and will be removed in 1.15.0: mistral-common does not use it.
            See `BaseCompletionRequest`.
        top_p: Deprecated and will be removed in 1.15.0: mistral-common does not use it.
            See `BaseCompletionRequest`.
        max_tokens: Deprecated and will be removed in 1.15.0: mistral-common does not use it.
            See `BaseCompletionRequest`.
        random_seed: Deprecated and will be removed in 1.15.0: mistral-common does not use it.
            See `BaseCompletionRequest`.
        prompt: The prefix text the model continues from.
        suffix: The text that follows the model's completion. If `None`, the model
            generates text after prompt only (plain completion); if provided, the
            model generates text that logically fits between prompt and suffix.

    Examples:
        >>> request = FIMRequest(prompt="Hello, my name is", suffix=" and I live in New York.")
    """

    _deprecated_fields: ClassVar[tuple[str, ...]] = _DEPRECATED_FIELDS["FIMRequest"]

    prompt: str
    suffix: str | None = None

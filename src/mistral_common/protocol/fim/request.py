from mistral_common.protocol.base import BaseCompletionRequest


class FIMRequest(BaseCompletionRequest):
    r"""A valid Fill in the Middle completion request to be tokenized.

    FIM requests ask the model to complete text between a prefix and a suffix,
    e.g., for code completion in an editor.

    Attributes:
        prompt: The prefix text the model continues from.
        suffix: The text that follows the model's completion. If `None`, the model
            generates text after prompt only (plain completion); if provided, the
            model generates text that logically fits between prompt and suffix.

    Examples:
        >>> request = FIMRequest(prompt="Hello, my name is", suffix=" and I live in New York.")
    """

    prompt: str
    suffix: str | None = None

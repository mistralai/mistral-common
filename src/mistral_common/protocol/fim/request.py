from typing import Optional

from mistral_common.protocol.base import BaseCompletionRequest


class FIMRequest(BaseCompletionRequest):
    r"""A valid Fill in the Middle completion request to be tokenized.

    Attributes:
        prompt: The prompt to be completed.
        suffix: The suffix of the prompt. If provided, the model will generate text between the prompt and the suffix.

    Examples:
        >>> request = FIMRequest(prompt="Hello, my name is", suffix=" and I live in New York.")
    """

    prompt: str
    suffix: Optional[str] = None

from typing import Any, Dict, Generic, List, Optional, Union

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.converters import (
    _check_openai_fields_names,
    _is_openai_field_name,
    convert_openai_messages,
    convert_openai_tools,
)
from mistral_common.protocol.instruct.messages import ChatMessage, ChatMessageType
from mistral_common.protocol.instruct.tool_calls import ToolType


class FIMRequest(MistralBase):
    r"""A valid Fill in the Middle completion request to be tokenized.

    Attributes:
        prompt: The prompt to be completed.
        suffix: The suffix of the prompt. If provided, the model will generate text between the prompt and the suffix.

    Examples:
        >>> request = FIMRequest(prompt="Hello, my name is", suffix=" and I live in New York.")
    """

    prompt: str
    suffix: Optional[str] = None

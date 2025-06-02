from typing import Generic, List, Optional

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.messages import ChatMessageType
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


class InstructRequest(MistralBase, Generic[ChatMessageType, ToolType]):
    """A valid Instruct request to be tokenized.

    Attributes:
        messages: The history of the conversation.
        system_prompt: The system prompt to be used for the conversation.
        available_tools: The tools available to the assistant.
        truncate_at_max_tokens: The maximum number of tokens to truncate the conversation at.

    Examples:
        >>> from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
        >>> request = InstructRequest(
        ...     messages=[UserMessage(content="Hello, how are you?")], system_prompt="You are a helpful assistant."
        ... )
    """

    messages: List[ChatMessageType]
    system_prompt: Optional[str] = None
    available_tools: Optional[List[ToolType]] = None
    truncate_at_max_tokens: Optional[int] = None

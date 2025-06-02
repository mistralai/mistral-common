from enum import Enum
from typing import Generic, List, Optional

from pydantic import Field

from mistral_common.base import MistralBase
from mistral_common.protocol.base import BaseCompletionRequest
from mistral_common.protocol.instruct.messages import ChatMessageType
from mistral_common.protocol.instruct.tool_calls import Tool, ToolChoice


class ResponseFormats(str, Enum):
    r"""Enum of the different formats of an instruct response.

    Attributes:
        text: The response is a plain text.
        json: The response is a JSON object.

    Examples:
        >>> response_format = ResponseFormats.text
    """

    text = "text"
    json = "json_object"


class ResponseFormat(MistralBase):
    r"""The format of the response.

    Attributes:
        type: The type of the response.

    Examples:
        >>> response_format = ResponseFormat(type=ResponseFormats.text)
    """

    type: ResponseFormats = ResponseFormats.text


class ChatCompletionRequest(BaseCompletionRequest, Generic[ChatMessageType]):
    r"""Request for a chat completion.

    Attributes:
        model: The model to use for the chat completion.
        messages: The messages to use for the chat completion.
        response_format: The format of the response.
        tools: The tools to use for the chat completion.
        tool_choice: The tool choice to use for the chat completion.
        truncate_for_context_length: Whether to truncate the messages for the context length.

    Examples:
        >>> from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
        >>> from mistral_common.protocol.instruct.tool_calls import ToolTypes, Function
        >>> request = ChatCompletionRequest(
        ...     messages=[
        ...         UserMessage(content="Hello!"),
        ...         AssistantMessage(content="Hi! How can I help you?"),
        ...     ],
        ...     response_format=ResponseFormat(type=ResponseFormats.text),
        ...     tools=[Tool(type=ToolTypes.function, function=Function(name="get_weather", parameters={}))],
        ...     tool_choice=ToolChoice.auto,
        ...     truncate_for_context_length=True,
        ... )
    """

    model: Optional[str] = None
    messages: List[ChatMessageType]
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    tools: Optional[List[Tool]] = None
    tool_choice: ToolChoice = ToolChoice.auto
    truncate_for_context_length: bool = False

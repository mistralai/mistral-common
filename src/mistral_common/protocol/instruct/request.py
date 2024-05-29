from enum import Enum
from typing import Generic, List, Optional

from pydantic import Field

from mistral_common.base import MistralBase
from mistral_common.protocol.base import BaseCompletionRequest
from mistral_common.protocol.instruct.messages import ChatMessageType
from mistral_common.protocol.instruct.tool_calls import Tool, ToolChoice


class ResponseFormats(str, Enum):
    text: str = "text"
    json: str = "json_object"


class ResponseFormat(MistralBase):
    type: ResponseFormats = ResponseFormats.text


class ChatCompletionRequest(BaseCompletionRequest, Generic[ChatMessageType]):
    model: Optional[str] = None
    messages: List[ChatMessageType]
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    tools: Optional[List[Tool]] = None
    tool_choice: ToolChoice = ToolChoice.auto

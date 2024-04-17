from enum import Enum
from typing import List, Optional

from pydantic import Field

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.messages import ChatMessage
from mistral_common.protocol.instruct.tool_calls import Tool, ToolChoice


class ResponseFormats(str, Enum):
    text: str = "text"
    json: str = "json_object"


class ResponseFormat(MistralBase):
    type: ResponseFormats = ResponseFormats.text


class ChatCompletionRequest(MistralBase):
    model: Optional[str] = None
    messages: List[ChatMessage]
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    tools: List[Tool] = Field(default_factory=list)
    tool_choice: ToolChoice = ToolChoice.auto
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=0)
    random_seed: Optional[int] = Field(default=None, ge=0)

from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import Field
from typing_extensions import Annotated  # compatibility with 3.8

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.tool_calls import ToolCall


class Roles(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ChunkTypes(str, Enum):
    text = "text"


class ContentChunk(MistralBase):
    type: ChunkTypes = ChunkTypes.text
    text: str


class BaseMessage(MistralBase):
    role: Literal[Roles.system, Roles.user, Roles.assistant, Roles.tool]


class UserMessage(BaseMessage):
    role: Literal[Roles.user] = Roles.user
    content: Union[str, List[ContentChunk]]


class SystemMessage(BaseMessage):
    role: Literal[Roles.system] = Roles.system
    content: Union[str, List[ContentChunk]]


class AssistantMessage(BaseMessage):
    role: Literal[Roles.assistant] = Roles.assistant
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ToolMessage(BaseMessage):
    content: str
    role: Literal[Roles.tool] = Roles.tool
    tool_call_id: Optional[str] = None

    # Deprecated in V3 tokenization
    name: Optional[str] = None


ChatMessage = Annotated[Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage], Field(discriminator="role")]

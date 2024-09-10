from enum import Enum
from typing import List, Literal, Optional, TypeVar, Union

from pydantic import ConfigDict, Field
from typing_extensions import Annotated, TypeAlias

from mistral_common.base import MistralBase
from mistral_common.multimodal import SerializableImage
from mistral_common.protocol.instruct.tool_calls import ToolCall


class ChunkTypes(str, Enum):
    text = "text"
    image = "image"
    image_url = "image_url"


class BaseContentChunk(MistralBase):
    type: Literal[ChunkTypes.text, ChunkTypes.image, ChunkTypes.image_url]


class ImageChunk(BaseContentChunk):
    type: Literal[ChunkTypes.image] = ChunkTypes.image
    image: SerializableImage
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ImageURL(MistralBase):
    url: str
    detail: Optional[str] = None


class ImageURLChunk(BaseContentChunk):
    """
    {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0
    """

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: Union[ImageURL, str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url


class TextChunk(BaseContentChunk):
    type: Literal[ChunkTypes.text] = ChunkTypes.text
    text: str


ContentChunk = Annotated[Union[TextChunk, ImageChunk, ImageURLChunk], Field(discriminator="type")]


class Roles(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


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
    prefix: bool = False


class FinetuningAssistantMessage(AssistantMessage):
    weight: Optional[float] = None


class ToolMessage(BaseMessage):
    content: str
    role: Literal[Roles.tool] = Roles.tool
    tool_call_id: Optional[str] = None

    # Deprecated in V3 tokenization
    name: Optional[str] = None


ChatMessage = Annotated[Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage], Field(discriminator="role")]

FinetuningMessage = Annotated[
    Union[SystemMessage, UserMessage, FinetuningAssistantMessage, ToolMessage],
    Field(discriminator="role"),
]

ChatMessageType = TypeVar("ChatMessageType", bound=ChatMessage)

# Used for type hinting in generic classes where we might override the message types
UserMessageType = TypeVar("UserMessageType", bound=UserMessage)
AssistantMessageType = TypeVar("AssistantMessageType", bound=AssistantMessage)
ToolMessageType = TypeVar("ToolMessageType", bound=ToolMessage)
SystemMessageType = TypeVar("SystemMessageType", bound=SystemMessage)

UATS: TypeAlias = Union[UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType]

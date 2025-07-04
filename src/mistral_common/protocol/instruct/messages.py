import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypeVar, Union

from pydantic import ConfigDict, Field
from typing_extensions import Annotated, TypeAlias

from mistral_common.base import MistralBase
from mistral_common.image import SerializableImage
from mistral_common.protocol.instruct.tool_calls import ToolCall


class ChunkTypes(str, Enum):
    r"""Enum for the types of chunks that can be sent to the model.

    Attributes:
       text: A text chunk.
       image: An image chunk.
       image_url: An image url chunk.

    Examples:
        >>> from mistral_common.protocol.instruct.messages import ChunkTypes
        >>> chunk_type = ChunkTypes.text
    """

    text = "text"
    image = "image"
    image_url = "image_url"


class BaseContentChunk(MistralBase):
    r"""Base class for all content chunks.

    Content chunks are used to send different types of content to the model.

    Attributes:
       type: The type of the chunk.
    """

    type: Literal[ChunkTypes.text, ChunkTypes.image, ChunkTypes.image_url]

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_chunk: Dict[str, Union[str, Dict[str, str]]]) -> "BaseContentChunk":
        r"""Converts the OpenAI chunk to the Mistral format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}")


class ImageChunk(BaseContentChunk):
    r"""Image chunk.

    Attributes:
       image: The image to be sent to the model.

    Examples:
        >>> from PIL import Image
        >>> image_chunk = ImageChunk(image=Image.new('RGB', (200, 200), color='blue'))
    """

    type: Literal[ChunkTypes.image] = ChunkTypes.image
    image: SerializableImage
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format."""
        base64_image = self.model_dump(include={"image"}, context={"add_format_prefix": True})["image"]
        return {"type": "image_url", "image_url": {"url": base64_image}}

    @classmethod
    def from_openai(cls, openai_chunk: Dict[str, Union[str, Dict[str, str]]]) -> "ImageChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        assert openai_chunk.get("type") == "image_url", openai_chunk

        image_url_dict = openai_chunk["image_url"]
        assert isinstance(image_url_dict, dict) and "url" in image_url_dict, image_url_dict

        if re.match(r"^data:image/\w+;base64,", image_url_dict["url"]):  # Remove the prefix if it exists
            image_url_dict["url"] = image_url_dict["url"].split(",")[1]

        return cls.model_validate({"image": image_url_dict["url"]})


class ImageURL(MistralBase):
    r"""Image URL or a base64 encoded image.

    Attributes:
       url: The URL of the image.
       detail: The detail of the image.

    Examples:
       >>> image_url = ImageURL(url="https://example.com/image.png")
    """

    url: str
    detail: Optional[str] = None


class ImageURLChunk(BaseContentChunk):
    r"""Image URL chunk.

    Attributes:
       image_url: The URL of the image or a base64 encoded image to be sent to the model.

    Examples:
        >>> image_url_chunk = ImageURLChunk(image_url="data:image/png;base64,iVBORw0")
    """

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: Union[ImageURL, str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format."""
        image_url_dict = {"url": self.get_url()}
        if isinstance(self.image_url, ImageURL) and self.image_url.detail is not None:
            image_url_dict["detail"] = self.image_url.detail

        out_dict: Dict[str, Union[str, Dict[str, str]]] = {
            "type": "image_url",
            "image_url": image_url_dict,
        }
        return out_dict

    @classmethod
    def from_openai(cls, openai_chunk: Dict[str, Union[str, Dict[str, str]]]) -> "ImageURLChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        return cls.model_validate({"image_url": openai_chunk["image_url"]})


class TextChunk(BaseContentChunk):
    r"""Text chunk.

    Attributes:
      text: The text to be sent to the model.

    Examples:
        >>> text_chunk = TextChunk(text="Hello, how can I help you?")
    """

    type: Literal[ChunkTypes.text] = ChunkTypes.text
    text: str

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format."""
        return self.model_dump()

    @classmethod
    def from_openai(cls, messages: Dict[str, Union[str, Dict[str, str]]]) -> "TextChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        return cls.model_validate(messages)


ContentChunk = Annotated[Union[TextChunk, ImageChunk, ImageURLChunk], Field(discriminator="type")]


def _convert_openai_content_chunks(openai_content_chunks: Dict[str, Union[str, Dict[str, str]]]) -> ContentChunk:
    content_type_str = openai_content_chunks.get("type")

    if content_type_str is None:
        raise ValueError("Content chunk must have a type field.")

    content_type = ChunkTypes(content_type_str)

    if content_type == ChunkTypes.text:
        return TextChunk.from_openai(openai_content_chunks)
    elif content_type == ChunkTypes.image_url:
        return ImageURLChunk.from_openai(openai_content_chunks)
    elif content_type == ChunkTypes.image:
        return ImageChunk.from_openai(openai_content_chunks)
    else:
        raise ValueError(f"Unknown content chunk type: {content_type}")


class Roles(str, Enum):
    r"""Enum for the roles of the messages.

    Attributes:
       system: The system role.
       user: The user role.
       assistant: The assistant role.
       tool: The tool role.

    Examples:
        >>> role = Roles.user
    """

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BaseMessage(MistralBase):
    r"""Base class for all messages.

    Attributes:
       role: The role of the message.
    """

    role: Literal[Roles.system, Roles.user, Roles.assistant, Roles.tool]

    def to_openai(self) -> Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]:
        r"""Converts the message to the OpenAI format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(
        cls, openai_message: Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]
    ) -> "BaseMessage":
        r"""Converts the OpenAI message to the Mistral format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}.")


class UserMessage(BaseMessage):
    r"""User message.

    Attributes:
        content: The content of the message.

    Examples:
        >>> message = UserMessage(content="Can you help me to write a poem?")
    """

    role: Literal[Roles.user] = Roles.user
    content: Union[str, List[ContentChunk]]

    def to_openai(self) -> Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]:
        r"""Converts the message to the OpenAI format."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": [chunk.to_openai() for chunk in self.content]}

    @classmethod
    def from_openai(
        cls, openai_message: Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]
    ) -> "UserMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        if isinstance(openai_message["content"], str):
            return cls.model_validate(openai_message)
        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": [_convert_openai_content_chunks(chunk) for chunk in openai_message["content"]],
            },
        )


class SystemMessage(BaseMessage):
    r"""System message.

    Attributes:
        content: The content of the message.

    Examples:
        >>> message = SystemMessage(content="You are a helpful assistant.")
    """

    role: Literal[Roles.system] = Roles.system
    content: str

    def to_openai(self) -> Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]:
        r"""Converts the message to the OpenAI format."""
        return self.model_dump()

    @classmethod
    def from_openai(
        cls, openai_message: Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]
    ) -> "SystemMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        return cls.model_validate(openai_message)


class AssistantMessage(BaseMessage):
    r"""Assistant message.

    Attributes:
        role: The role of the message.
        content: The content of the message.
        tool_calls: The tool calls of the message.
        prefix: Whether the message is a prefix.

    Examples:
        >>> message = AssistantMessage(content="Hello, how can I help you?")
    """

    role: Literal[Roles.assistant] = Roles.assistant
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    prefix: bool = False

    def to_openai(self) -> Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]:
        r"""Converts the message to the OpenAI format."""
        out_dict: dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]] = {
            "role": self.role,
        }
        if self.content is not None:
            out_dict["content"] = self.content
        if self.tool_calls is not None:
            out_dict["tool_calls"] = [tool_call.to_openai() for tool_call in self.tool_calls]

        return out_dict

    @classmethod
    def from_openai(
        cls, openai_message: Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]
    ) -> "AssistantMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        openai_tool_calls = openai_message.get("tool_calls", None)
        tools_calls = (
            [
                ToolCall.from_openai(openai_tool_call)  # type: ignore[arg-type]
                for openai_tool_call in openai_tool_calls
            ]
            if openai_tool_calls is not None
            else None
        )

        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": openai_message.get("content"),
                "tool_calls": tools_calls,
            }
        )


class FinetuningAssistantMessage(AssistantMessage):
    r"""Assistant message for finetuning.

    Attributes:
        weight: The weight of the message to train on.

    Examples:
        >>> message = FinetuningAssistantMessage(content="Hello, how can I help you?", weight=0.5)
    """

    weight: Optional[float] = None


class ToolMessage(BaseMessage):
    r"""Tool message.

    Attributes:
        content: The content of the message.
        tool_call_id: The tool call id of the message.
        name: The name of the tool. (Deprecated in V3 tokenization)

    Examples:
       >>> message = ToolMessage(content="Hello, how can I help you?", tool_call_id="123")
    """

    content: str
    role: Literal[Roles.tool] = Roles.tool
    tool_call_id: Optional[str] = None

    # Deprecated in V3 tokenization
    name: Optional[str] = None

    def to_openai(self) -> Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]:
        r"""Converts the message to the OpenAI format."""
        assert self.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return self.model_dump(exclude={"name"})

    @classmethod
    def from_openai(cls, messages: Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]) -> "ToolMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        tool_message = cls.model_validate(messages)
        assert tool_message.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return tool_message


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

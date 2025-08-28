from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypeVar, Union

from pydantic import Field
from typing_extensions import Annotated, TypeAlias

from mistral_common.base import MistralBase
from mistral_common.imports import create_deprecate_cls_import
from mistral_common.protocol.instruct.chunk import (
    AudioChunk as ToDeprecateAudioChunk,
)
from mistral_common.protocol.instruct.chunk import (
    AudioURL as ToDeprecateAudioURL,
)
from mistral_common.protocol.instruct.chunk import (
    AudioURLChunk as ToDeprecateAudioURLChunk,
)
from mistral_common.protocol.instruct.chunk import (
    AudioURLType as ToDeprecateAudioURLType,
)
from mistral_common.protocol.instruct.chunk import (
    BaseContentChunk as ToDeprecateBaseContentChunk,
)
from mistral_common.protocol.instruct.chunk import (
    ChunkTypes as ToDeprecateChunkTypes,
)
from mistral_common.protocol.instruct.chunk import (
    ContentChunk,
    TextChunk,
    ThinkChunk,
    UserContentChunk,
    _convert_openai_content_chunks,
)
from mistral_common.protocol.instruct.chunk import (
    ImageChunk as ToDeprecateImageChunk,
)
from mistral_common.protocol.instruct.chunk import (
    ImageURL as ToDeprecateImageURL,
)
from mistral_common.protocol.instruct.chunk import (
    ImageURLChunk as ToDeprecateImageURLChunk,
)
from mistral_common.protocol.instruct.chunk import (
    RawAudio as ToDeprecateRawAudio,
)
from mistral_common.protocol.instruct.tool_calls import ToolCall

AudioChunk = create_deprecate_cls_import(ToDeprecateAudioChunk, __name__, ToDeprecateAudioChunk.__module__, "1.10.0")
AudioURL = create_deprecate_cls_import(ToDeprecateAudioURL, __name__, ToDeprecateAudioURL.__module__, "1.10.0")
AudioURLChunk = create_deprecate_cls_import(
    ToDeprecateAudioURLChunk, __name__, ToDeprecateAudioURLChunk.__module__, "1.10.0"
)
AudioURLType = create_deprecate_cls_import(
    ToDeprecateAudioURLType, __name__, ToDeprecateAudioURLType.__module__, "1.10.0"
)
BaseContentChunk = create_deprecate_cls_import(
    ToDeprecateBaseContentChunk, __name__, ToDeprecateBaseContentChunk.__module__, "1.10.0"
)
ChunkTypes = create_deprecate_cls_import(ToDeprecateChunkTypes, __name__, ToDeprecateChunkTypes.__module__, "1.10.0")
ImageURL = create_deprecate_cls_import(ToDeprecateImageURL, __name__, ToDeprecateImageURL.__module__, "1.10.0")
ImageURLChunk = create_deprecate_cls_import(
    ToDeprecateImageURLChunk, __name__, ToDeprecateImageURLChunk.__module__, "1.10.0"
)
ImageChunk = create_deprecate_cls_import(ToDeprecateImageChunk, __name__, ToDeprecateImageChunk.__module__, "1.10.0")
RawAudio = create_deprecate_cls_import(ToDeprecateRawAudio, __name__, ToDeprecateRawAudio.__module__, "1.10.0")


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
    content: Union[str, List[UserContentChunk]]

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
    content: Union[str, List[Union[TextChunk, ThinkChunk]]]

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
    content: Optional[Union[str, List[Union[TextChunk, ThinkChunk]]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    prefix: bool = False

    def to_openai(self) -> Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]:
        r"""Converts the message to the OpenAI format."""
        out_dict: dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]] = {
            "role": self.role,
        }
        if self.content is None:
            pass
        elif isinstance(self.content, str):
            out_dict["content"] = self.content
        else:
            out_dict["content"] = [chunk.to_openai() for chunk in self.content]
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
        openai_content = openai_message.get("content", None)
        content: Optional[Union[str, List[ContentChunk]]] = None
        if openai_content is None or isinstance(openai_content, str):
            content = openai_content
        elif isinstance(openai_content, list):
            content = [_convert_openai_content_chunks(chunk) for chunk in openai_content]
        else:
            raise ValueError(f"Unknown content type: {type(openai_content)}")

        return cls.model_validate(
            {
                "role": openai_message["role"],
                "content": content,
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

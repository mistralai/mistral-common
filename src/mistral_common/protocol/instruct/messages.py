import warnings
from enum import Enum
from typing import Any, Literal, TypeVar

from pydantic import Field, field_validator
from typing_extensions import Annotated, TypeAlias

from mistral_common.base import MistralBase
from mistral_common.exceptions import InvalidAssistantMessageException
from mistral_common.protocol.instruct.chunk import (
    ContentChunk,
    TextChunk,
    ThinkChunk,
    UserContentChunk,
    _convert_openai_content_chunks,
)
from mistral_common.protocol.instruct.tool_calls import ToolCall

warnings.filterwarnings(
    action="once",
    category=FutureWarning,
    message=r".*`convert_thinking_format` defaults to 'thinking_chunks'.*",
)


class OpenAIReasoningField(str, Enum):
    r"""How to serialize leading `ThinkChunk` in `AssistantMessage.to_openai()`.

    Attributes:
        thinking_chunks: Use think chunks (Mistral convention).
        reasoning: Flat `reasoning` string (vLLM convention).
        reasoning_content: Flat `reasoning_content` string (SGLang convention).
    """

    thinking_chunks = "thinking_chunks"
    reasoning = "reasoning"
    reasoning_content = "reasoning_content"


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

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the message to the OpenAI format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "BaseMessage":
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
    content: str | list[UserContentChunk]

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the message to the OpenAI format."""
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        return {"role": self.role, "content": [chunk.to_openai() for chunk in self.content]}

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "UserMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        if isinstance(openai_message["content"], str):
            return cls.model_validate_ignore_extra(openai_message)
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
    content: str | list[TextChunk | ThinkChunk]

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the message to the OpenAI format."""
        return self.model_dump()

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "SystemMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        return cls.model_validate_ignore_extra(openai_message)


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
    content: str | list[TextChunk | ThinkChunk] | None = None
    tool_calls: list[ToolCall] | None = None
    prefix: bool = False

    @field_validator("content")
    @classmethod
    def _validate_thinking_chunks_are_leading(
        cls, content: str | list[TextChunk | ThinkChunk] | None
    ) -> str | list[TextChunk | ThinkChunk] | None:
        """Validates that all ThinkChunks are contiguous and at the start of the content list."""
        if not isinstance(content, list):
            return content
        seen_non_think = False
        for chunk in content:
            if isinstance(chunk, ThinkChunk):
                if seen_non_think:
                    raise InvalidAssistantMessageException(
                        "ThinkChunks must be leading: all ThinkChunks must appear before any other content chunk."
                    )
            else:
                seen_non_think = True
        return content

    def to_openai(
        self,
        convert_thinking_format: OpenAIReasoningField | None = None,
    ) -> dict[str, Any]:
        r"""Converts the message to the OpenAI format.

        Args:
            convert_thinking_format: Conversion strategy for think chunks. When ``None``, defaults to
                `OpenAIReasoningField.thinking_chunks` (chunks kept inline) but emits a `FutureWarning` if
                the content contains `ThinkChunk`.
        """
        out_dict: dict[str, Any] = {
            "role": self.role,
        }
        if self.content is None:
            pass
        elif isinstance(self.content, str):
            out_dict["content"] = self.content
        else:
            if convert_thinking_format is None and any(isinstance(c, ThinkChunk) for c in self.content):
                warnings.warn(
                    "`convert_thinking_format` defaults to 'thinking_chunks' but will change to 'reasoning' "
                    "in 1.13.0. Pass `convert_thinking_format` explicitly to silence this warning.",
                    FutureWarning,
                    stacklevel=2,
                )

            effective_format = convert_thinking_format or OpenAIReasoningField.thinking_chunks

            if effective_format == OpenAIReasoningField.thinking_chunks:
                out_dict["content"] = [chunk.to_openai() for chunk in self.content]
            else:
                split_idx = 0
                for chunk in self.content:
                    if isinstance(chunk, ThinkChunk):
                        split_idx += 1
                    else:
                        break

                leading_thinks = self.content[:split_idx]
                remaining = self.content[split_idx:]

                if leading_thinks:
                    match effective_format:
                        case OpenAIReasoningField.reasoning | OpenAIReasoningField.reasoning_content:
                            combined = "\n".join(tc.thinking for tc in leading_thinks if isinstance(tc, ThinkChunk))
                            out_dict[effective_format.value] = combined
                        case _:
                            raise ValueError(f"{effective_format=} is not supported.")

                if len(remaining) == 1 and isinstance(remaining[0], TextChunk):
                    out_dict["content"] = remaining[0].text
                elif remaining:
                    out_dict["content"] = [chunk.to_openai() for chunk in remaining]

        if self.tool_calls is not None:
            out_dict["tool_calls"] = [tool_call.to_openai() for tool_call in self.tool_calls]

        return out_dict

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "AssistantMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        openai_tool_calls = openai_message.get("tool_calls", None)
        if openai_tool_calls is None:
            tools_calls: list[ToolCall] | None = None
        elif isinstance(openai_tool_calls, list):
            tools_calls = []
            for openai_tool_call in openai_tool_calls or []:
                tools_calls.append(ToolCall.from_openai(openai_tool_call))
        else:
            raise ValueError(f"tool_calls must be a list, got {type(openai_tool_calls)}")
        openai_content = openai_message.get("content", None)
        content: str | list[ContentChunk] | None = None
        if openai_content is None or isinstance(openai_content, str):
            content = openai_content
        elif isinstance(openai_content, list):
            content = [_convert_openai_content_chunks(chunk) for chunk in openai_content]
        else:
            raise ValueError(f"Unknown content type: {type(openai_content)}")

        reasoning_content: str | None = openai_message.get("reasoning_content")
        reasoning: str | None = openai_message.get("reasoning")

        match reasoning_content, reasoning:
            case None, None:
                openai_thinking = None
            case None, _:
                openai_thinking = reasoning
            case _, None:
                openai_thinking = reasoning_content
            case _, _:
                if reasoning_content != reasoning:
                    raise ValueError("`reasoning_content` and `reasoning` should be equal.")
                openai_thinking = reasoning

        if openai_thinking is not None:
            has_thinking_chunks = isinstance(content, list) and any(isinstance(chunk, ThinkChunk) for chunk in content)
            if has_thinking_chunks:
                raise InvalidAssistantMessageException(
                    "Message cannot have both thinking chunks in content and a top-level"
                    " `reasoning` or `reasoning_content` field."
                )

            reasoning_chunk = ThinkChunk(thinking=openai_thinking, closed=True)
            if isinstance(content, str):
                content = [reasoning_chunk, TextChunk(text=content)]
            elif content is None:
                content = [reasoning_chunk]
            else:
                content.insert(0, reasoning_chunk)

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

    weight: float | None = None


class ToolMessage(BaseMessage):
    r"""Tool message.

    Attributes:
        content: The content of the message.
        tool_call_id: The tool call id of the message.
        name: The name of the tool. (Deprecated in V3 tokenization)

    Examples:
       >>> message = ToolMessage(content="Hello, how can I help you?", tool_call_id="123")
    """

    content: str | list[TextChunk]
    role: Literal[Roles.tool] = Roles.tool
    tool_call_id: str | None = None

    # Deprecated in V3 tokenization
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the message to the OpenAI format."""
        assert self.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return self.model_dump(exclude={"name"})

    @classmethod
    def from_openai(cls, messages: dict[str, str | list[dict[str, str | dict[str, Any]]]]) -> "ToolMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        tool_message = cls.model_validate_ignore_extra(messages)
        assert tool_message.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return tool_message


ChatMessage = Annotated[SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")]

FinetuningMessage = Annotated[
    SystemMessage | UserMessage | FinetuningAssistantMessage | ToolMessage,
    Field(discriminator="role"),
]

ChatMessageType = TypeVar("ChatMessageType", bound=ChatMessage)

# Used for type hinting in generic classes where we might override the message types
UserMessageType = TypeVar("UserMessageType", bound=UserMessage)
AssistantMessageType = TypeVar("AssistantMessageType", bound=AssistantMessage)
ToolMessageType = TypeVar("ToolMessageType", bound=ToolMessage)
SystemMessageType = TypeVar("SystemMessageType", bound=SystemMessage)

UATS: TypeAlias = UserMessageType | AssistantMessageType | ToolMessageType | SystemMessageType

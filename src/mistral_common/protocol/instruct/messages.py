import warnings
from collections.abc import Sequence
from enum import Enum
from typing import Any, ClassVar, Literal, TypeVar

from pydantic import Field, model_validator
from typing_extensions import Annotated, TypeAlias, TypeGuard

from mistral_common.base import MistralBase
from mistral_common.exceptions import InvalidAssistantMessageException
from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURLChunk,
    BaseContentChunk,
    ContentChunk,
    ImageChunk,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
    _convert_openai_content_chunks,
)
from mistral_common.protocol.instruct.tool_calls import ToolCall

warnings.filterwarnings(
    action="once",
    category=FutureWarning,
    message=r".*`convert_thinking_format` defaults to 'thinking_chunks'.*",
)


def _are_think_chunks(chunks: Sequence[ContentChunk]) -> TypeGuard[list[ThinkChunk]]:
    r"""Narrow a chunk list to ThinkChunk list."""
    return all(isinstance(c, ThinkChunk) for c in chunks)


def _are_text_chunks(chunks: Sequence[ContentChunk]) -> TypeGuard[list[TextChunk]]:
    r"""Narrow a chunk list to TextChunk list."""
    return all(isinstance(c, TextChunk) for c in chunks)


class ReasoningFieldFormat(str, Enum):
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

    # Allow-list of content chunk types accepted by this message. Must be set by each subclass.
    _allowed_content_chunks: ClassVar[tuple[type[BaseContentChunk], ...]]

    @model_validator(mode="after")
    def _validate_allowed_content_chunks(self) -> "BaseMessage":
        r"""Enforce the per-message content chunk allow-list."""
        content = getattr(self, "content", None)
        if isinstance(content, list):
            for chunk in content:
                if not isinstance(chunk, self._allowed_content_chunks):
                    raise ValueError(f"{type(chunk).__name__} cannot be used in {self.role} message.")
        return self

    @staticmethod
    def _content_to_openai(
        content: str | Sequence[ContentChunk] | None,
    ) -> str | list[dict[str, Any]] | None:
        r"""Serialize message content to OpenAI format.

        Args:
            content: String, list of content chunks, or None.

        Returns:
            String content as-is, list of chunks serialized via each chunk's
            to_openai(), or None.
        """
        if content is None or isinstance(content, str):
            return content
        return [chunk.to_openai() for chunk in content]

    @staticmethod
    def _content_from_openai(
        raw: str | list[dict[str, Any]] | None,
    ) -> str | list[ContentChunk] | None:
        r"""Deserialize content from OpenAI format.

        Args:
            raw: Raw content from OpenAI message dict.

        Returns:
            String content as-is, list of deserialized content chunks, or None.

        Raises:
            ValueError: If content type is unrecognized.
        """
        if raw is None or isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            return [_convert_openai_content_chunks(chunk) for chunk in raw]
        raise ValueError(f"Unknown content type: {type(raw)}")

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
    content: str | list[ContentChunk]
    _allowed_content_chunks: ClassVar[tuple[type[BaseContentChunk], ...]] = (
        TextChunk,
        ImageChunk,
        ImageURLChunk,
        AudioChunk,
        AudioURLChunk,
    )

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the message to the OpenAI format."""
        return {"role": self.role, "content": self._content_to_openai(self.content)}

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "UserMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        return cls.model_validate(
            {"role": openai_message["role"], "content": cls._content_from_openai(openai_message["content"])}
        )


class SystemMessage(BaseMessage):
    r"""System message.

    Attributes:
        content: The content of the message.

    Examples:
        >>> message = SystemMessage(content="You are a helpful assistant.")
    """

    role: Literal[Roles.system] = Roles.system
    content: str | list[ContentChunk]
    _allowed_content_chunks: ClassVar[tuple[type[BaseContentChunk], ...]] = (TextChunk, AudioChunk, ThinkChunk)

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the message to the OpenAI format."""
        return {"role": self.role, "content": self._content_to_openai(self.content)}

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "SystemMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        return cls.model_validate(
            {"role": openai_message["role"], "content": cls._content_from_openai(openai_message["content"])}
        )


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
    content: str | list[ContentChunk] | None = None
    _allowed_content_chunks: ClassVar[tuple[type[BaseContentChunk], ...]] = (TextChunk, ThinkChunk)
    tool_calls: list[ToolCall] | None = None
    prefix: bool = False

    def to_openai(
        self,
        reasoning_field_format: ReasoningFieldFormat | None = None,
    ) -> dict[str, Any]:
        r"""Converts the message to the OpenAI format.

        Args:
            reasoning_field_format: Format for converting thinking chunks. When `None`, defaults to
                `ReasoningFieldFormat.thinking_chunks` (chunks kept inline) but emits a `FutureWarning` if
                the content contains `ThinkChunk`.
        """
        out_dict: dict[str, Any] = {
            "role": self.role,
        }
        if self.tool_calls is not None:
            out_dict["tool_calls"] = [tool_call.to_openai() for tool_call in self.tool_calls]

        if self.content is None:
            return out_dict

        if isinstance(self.content, str):
            out_dict["content"] = self.content
            return out_dict

        last_think_idx: int = -1
        for i, chunk in enumerate(self.content):
            if isinstance(chunk, ThinkChunk):
                if (i - last_think_idx) > 1:
                    raise InvalidAssistantMessageException(
                        "ThinkChunks must be leading: all ThinkChunks must appear before any other content chunk."
                    )
                last_think_idx = i

        if reasoning_field_format is None and last_think_idx >= 0:
            warnings.warn(
                "`convert_thinking_format` defaults to 'thinking_chunks' but will change to 'reasoning' "
                "in 1.13.0. Pass `reasoning_field_format` explicitly to silence this warning.",
                FutureWarning,
                stacklevel=2,
            )

        match reasoning_field_format:
            case None | ReasoningFieldFormat.thinking_chunks:
                out_dict["content"] = self._content_to_openai(self.content)
            case ReasoningFieldFormat.reasoning | ReasoningFieldFormat.reasoning_content:
                think_chunks, content_chunks = self.content[: last_think_idx + 1], self.content[last_think_idx + 1 :]
                if not _are_think_chunks(think_chunks) or not _are_text_chunks(content_chunks):
                    raise RuntimeError("Impossible, only think or content chunks should have been present.")
                if len(think_chunks) > 0:
                    out_dict[reasoning_field_format.value] = "\n".join(tc.thinking for tc in think_chunks)

                if len(content_chunks) == 1:
                    out_dict["content"] = content_chunks[0].text
                elif content_chunks:
                    out_dict["content"] = self._content_to_openai(content_chunks)
            case _:
                raise ValueError(f"{reasoning_field_format=} is not supported.")

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
        content = cls._content_from_openai(openai_message.get("content"))

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

    content: str | list[ContentChunk]
    role: Literal[Roles.tool] = Roles.tool
    tool_call_id: str | None = None

    # Deprecated in V3 tokenization
    name: str | None = None

    # Tool messages do not accept thinking chunks.
    _allowed_content_chunks: ClassVar[tuple[type[BaseContentChunk], ...]] = (
        TextChunk,
        ImageChunk,
        ImageURLChunk,
        AudioChunk,
        AudioURLChunk,
    )

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the message to the OpenAI format."""
        assert self.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "content": self._content_to_openai(self.content),
        }

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "ToolMessage":
        r"""Converts the OpenAI message to the Mistral format."""
        content = cls._content_from_openai(openai_message["content"])
        tool_message = cls.model_validate(
            {
                "role": openai_message["role"],
                "tool_call_id": openai_message["tool_call_id"],
                "content": content,
                "name": openai_message.get("name"),
            }
        )
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

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
    r"""Format options for serializing thinking content in `AssistantMessage.to_openai()`.

    Controls how leading ThinkChunk content is represented in the OpenAI output.

    Attributes:
        thinking_chunks: Keep thinking content as inline chunks (Mistral convention).
            This preserves the chunk structure in the output.
        reasoning: Use a flat "reasoning" string field (vLLM convention).
            All thinking content is concatenated into a single string.
        reasoning_content: Use a flat `reasoning_content` string field (SGLang convention).
            Similar to reasoning but with a different field name.
    """

    thinking_chunks = "thinking_chunks"
    reasoning = "reasoning"
    reasoning_content = "reasoning_content"


class Roles(str, Enum):
    r"""Enum of valid message roles in a conversation.

    Attributes:
        system: System message providing context or instructions to the assistant.
        user: User message containing the user's input or query.
        assistant: Assistant message containing the model's response.
        tool: Tool message containing output from a tool/function call.

    Examples:
        >>> role = Roles.user
    """

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BaseMessage(MistralBase):
    r"""Abstract base class for all chat message types.

    Provides common functionality for message serialization and validation.
    Subclasses must implement `to_openai()` and `from_openai()`.

    Attributes:
        role: The role of this message (system, user, assistant, or tool).
    """

    role: Literal[Roles.system, Roles.user, Roles.assistant, Roles.tool]

    # Allow-list of content chunk types accepted by this message. Must be set by each subclass.
    _allowed_content_chunks: ClassVar[tuple[type[BaseContentChunk], ...]]

    @model_validator(mode="after")
    def _validate_allowed_content_chunks(self) -> "BaseMessage":
        r"""Validate that all content chunks are allowed for this message type.

        Each message subclass defines `_allowed_content_chunks` specifying which
        chunk types are valid. This validator raises ValueError if any chunk
        is not in the allowed list.

        Returns:
            Self, after validation.
        """
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
            content: Message content to serialize. Can be:
                - `None`: Returns `None`
                - str: Returns the string as-is
                - list of ContentChunk: Returns list of each chunk's `to_openai()` result

        Returns:
            Serialized content matching OpenAI's format:
                - `None` for `None` input
                - str for string input
                - list of dict for chunk list input
        """
        if content is None or isinstance(content, str):
            return content
        return [chunk.to_openai() for chunk in content]

    @staticmethod
    def _content_from_openai(
        raw: str | list[dict[str, Any]] | None,
    ) -> str | list[ContentChunk] | None:
        r"""Deserialize content from OpenAI format to Mistral format.

        Args:
            raw: Raw content from an OpenAI message dictionary. Can be:
                - `None`: Returns `None`
                - str: Returns the string as-is
                - list of dict: Each dict is converted to a ContentChunk

        Returns:
            Deserialized content in Mistral format:
                - `None` for `None` input
                - str for string input
                - list of ContentChunk for list input

        Raises:
            ValueError: If raw is not `None`, str, or list.
        """
        if raw is None or isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            return [_convert_openai_content_chunks(chunk) for chunk in raw]
        raise ValueError(f"Unknown content type: {type(raw)}")

    def to_openai(self) -> dict[str, Any]:
        r"""Convert this message to OpenAI format.

        Must be implemented by concrete subclasses.

        Returns:
            Dictionary matching OpenAI's message schema.

        Raises:
            NotImplementedError: Always, as this is an abstract method.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "BaseMessage":
        r"""Create a message instance from OpenAI format.

        Must be implemented by concrete subclasses.

        Args:
            openai_message: Dictionary matching OpenAI's message schema.

        Returns:
            Message instance of the appropriate subclass.

        Raises:
            NotImplementedError: Always, as this is an abstract method.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}.")


class UserMessage(BaseMessage):
    r"""A message from the user in a conversation.

    User messages can contain text, images, audio, or combinations thereof.

    Attributes:
        content: The message content. Can be:
            - str: Plain text message
            - list of ContentChunk: Multimodal content (text, images, audio)
            Valid chunk types: TextChunk, ImageChunk, ImageURLChunk, AudioChunk, AudioURLChunk

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
        r"""Convert this user message to OpenAI format.

        Returns:
            Dictionary with "role" set to "user" and "content" serialized.
        """
        return {"role": self.role, "content": self._content_to_openai(self.content)}

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "UserMessage":
        r"""Create a UserMessage from OpenAI format.

        Args:
            openai_message: Dictionary with "role" and "content" keys.

        Returns:
            UserMessage instance with content deserialized from OpenAI format.
        """
        return cls.model_validate(
            {"role": openai_message["role"], "content": cls._content_from_openai(openai_message["content"])}
        )


class SystemMessage(BaseMessage):
    r"""A system message providing context or instructions to the assistant.

    System messages set the behavior, persona, or context for the assistant.
    They are typically provided at the start of a conversation.

    Attributes:
        content: The message content. Can be:
            - str: Plain text instructions
            - list of ContentChunk: Text, audio, or thinking content
            Valid chunk types: TextChunk, AudioChunk, ThinkChunk

    Examples:
        >>> message = SystemMessage(content="You are a helpful assistant.")
    """

    role: Literal[Roles.system] = Roles.system
    content: str | list[ContentChunk]
    _allowed_content_chunks: ClassVar[tuple[type[BaseContentChunk], ...]] = (TextChunk, AudioChunk, ThinkChunk)

    def to_openai(self) -> dict[str, Any]:
        r"""Convert this system message to OpenAI format.

        Returns:
            Dictionary with "role" set to "system" and "content" serialized.
        """
        return {"role": self.role, "content": self._content_to_openai(self.content)}

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "SystemMessage":
        r"""Create a SystemMessage from OpenAI format.

        Args:
            openai_message: Dictionary with "role" and "content" keys.

        Returns:
            SystemMessage instance with content deserialized from OpenAI format.
        """
        return cls.model_validate(
            {"role": openai_message["role"], "content": cls._content_from_openai(openai_message["content"])}
        )


class AssistantMessage(BaseMessage):
    r"""A message from the assistant in a conversation.

    Assistant messages can contain the model's response text and/or tool calls.

    Attributes:
        content: The message content. Can be:
            - `None`: Empty message (only valid with `tool_calls`)
            - str: Plain text response
            - list of ContentChunk: Text and/or thinking content
            Valid chunk types: TextChunk, ThinkChunk
        tool_calls: List of ToolCall objects if the assistant called tools.
            If `None`, no tools were called.
        prefix: If `True`, this message is a prefix/partial message that will
            be continued. Used for streaming and continuation scenarios.

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
        r"""Convert this assistant message to OpenAI format.

        Handles conversion of thinking chunks to various OpenAI-compatible formats.

        Args:
            reasoning_field_format: Format for serializing thinking content:
                - `None`: Defaults to `thinking_chunks` but emits FutureWarning if
                  content contains ThinkChunk (will change to reasoning in 1.13.0)
                - `thinking_chunks`: Keep thinking as inline chunks
                - reasoning: Use flat "reasoning" field (vLLM convention)
                - `reasoning_content`: Use flat `reasoning_content` field (SGLang)

        Returns:
            Dictionary with "role" set to "assistant", and `content`/`tool_calls` serialized.

        Raises:
            InvalidAssistantMessageException: If ThinkChunks are not leading (must
                appear before any other content chunks).
            RuntimeError: If content chunks are not all ThinkChunk or TextChunk
                when using `reasoning`/`reasoning_content` formats.
            ValueError: If `reasoning_field_format` is not supported.
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
        r"""Create an AssistantMessage from OpenAI format.

        Handles conversion of OpenAI's `reasoning`/`reasoning_content` fields to
        Mistral's ThinkChunk format.

        Args:
            openai_message: Dictionary matching OpenAI's assistant message schema.
                Can contain "content", `tool_calls`, "reasoning", or `reasoning_content`.

        Returns:
            AssistantMessage instance with thinking content converted to ThinkChunk
            and `tool_calls` parsed from OpenAI format.

        Raises:
            InvalidAssistantMessageException: If message has both thinking chunks
                in `content` and a top-level `reasoning`/`reasoning_content` field.
            ValueError: If both `reasoning` and `reasoning_content` are present but unequal.
        """
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
                    "Message cannot have both thinking chunks in `content` and a top-level"
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
    r"""Assistant message with a weight for finetuning.

    Extends AssistantMessage with a weight parameter used during finetuning
    to control how much this message contributes to the training loss.

    Attributes:
        weight: The weight for this message during finetuning. If `None`, the message
            is treated with default weight. Must be >= 0 if provided.

    Examples:
        >>> message = FinetuningAssistantMessage(content="Hello, how can I help you?", weight=1)
    """

    weight: float | None = None


class ToolMessage(BaseMessage):
    r"""A message from a tool containing the result of a function call.

    Tool messages are sent by tools in response to ToolCall objects from
    the assistant. They contain the tool's output.

    Attributes:
        content: The tool's output content. Can be:
            - str: Plain text output
            - list of ContentChunk: Multimodal output (text, images, audio)
            Valid chunk types: TextChunk, ImageChunk, ImageURLChunk, AudioChunk, AudioURLChunk
        tool_call_id: The unique ID of the tool call this message responds to.
            Required for proper association with the assistant's tool call.
        name: The name of the tool (deprecated in V3 tokenization).
            Kept for backwards compatibility but not used in newer tokenizers.

    Examples:
       >>> message = ToolMessage(content="The weather is sunny", tool_call_id="123")
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
        r"""Convert this tool message to OpenAI format.

        Returns:
            Dictionary with "role" set to "tool", `tool_call_id`, and "content".

        Raises:
            AssertionError: If `tool_call_id` is `None` (required for tool messages).
        """
        assert self.tool_call_id is not None, "tool_call_id must be provided for tool messages."
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "content": self._content_to_openai(self.content),
        }

    @classmethod
    def from_openai(cls, openai_message: dict[str, Any]) -> "ToolMessage":
        r"""Create a ToolMessage from OpenAI format.

        Args:
            openai_message: Dictionary with "role", `tool_call_id`, "content",
                and optionally "name" keys.

        Returns:
            ToolMessage instance with content and `tool_call_id` parsed from OpenAI format.
        """
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

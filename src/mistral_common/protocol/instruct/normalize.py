import json
import warnings
from typing import Generic, Sequence

from typing_extensions import TypeGuard, assert_never

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.chunk import (
    AssistantContentChunk,
    AudioChunk,
    AudioURLChunk,
    ContentChunk,
    ImageChunk,
    ImageURLChunk,
    SystemContentChunk,
    TextChunk,
    ThinkChunk,
    UserContentChunk,
)
from mistral_common.protocol.instruct.messages import (
    UATS,
    AssistantMessage,
    AssistantMessageType,
    FinetuningAssistantMessage,
    Roles,
    SystemMessage,
    SystemMessageType,
    ToolMessage,
    ToolMessageType,
    UserMessage,
    UserMessageType,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest, ModelSettings
from mistral_common.protocol.instruct.tool_calls import FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.base import InstructRequestType, TokenizerVersion
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder

_DEFAULT_JOIN_STR = "\n\n"


def _is_user_content(
    chunks: list[ContentChunk],
) -> TypeGuard[list[UserContentChunk]]:
    r"""Narrow ContentChunk list to user-compatible types."""
    return all(isinstance(c, (TextChunk, ImageChunk, ImageURLChunk, AudioChunk, AudioURLChunk)) for c in chunks)


def _is_assistant_content(
    chunks: list[ContentChunk],
) -> TypeGuard[list[AssistantContentChunk]]:
    r"""Narrow ContentChunk list to assistant-compatible types."""
    return all(isinstance(c, (TextChunk, ThinkChunk)) for c in chunks)


def _is_system_content(
    chunks: list[ContentChunk],
) -> TypeGuard[list[SystemContentChunk]]:
    r"""Narrow ContentChunk list to system-compatible types."""
    return all(isinstance(c, (TextChunk, AudioChunk, ThinkChunk)) for c in chunks)


def _aggregate_content_chunks_impl(
    contents: list[list[ContentChunk] | str | None],
    msg_join_str: str,
    chunk_join_str: str,
) -> list[ContentChunk] | str:
    r"""Coalesce TextChunks within the same message and across different messages.

    Adjacent TextChunks within the same message are joined with `chunk_join_str`.
    Text from different messages is joined with `msg_join_str`.

    Args:
        contents: A list of message contents, where each element is either a string,
            a list of ContentChunks, or None. This is typically
            `[message.content for message in messages]`.
        msg_join_str: Separator inserted between text from different messages.
        chunk_join_str: Separator inserted between adjacent text chunks within
            the same message.

    Returns:
        A plain string if only text chunks were present, otherwise a list of
        ContentChunks with adjacent text coalesced.
    """
    all_content: list[ContentChunk] = []
    cur_text_parts: list[str] = []

    def _flush_text() -> None:
        if cur_text_parts:
            all_content.append(TextChunk(text="".join(cur_text_parts)))
            cur_text_parts.clear()

    for content in contents:
        needs_new_msg_sep = bool(cur_text_parts)
        if not content:  # skip None or empty string
            continue
        elif isinstance(content, str):
            join = msg_join_str if needs_new_msg_sep else ""
            cur_text_parts.append(join + content)
        else:  # list[ContentChunk]
            for chunk in content:
                if isinstance(chunk, TextChunk):
                    if not chunk.text:  # skip empty text chunks
                        continue
                    if not cur_text_parts:
                        join = ""
                    elif needs_new_msg_sep:
                        join = msg_join_str
                    else:
                        join = chunk_join_str
                    cur_text_parts.append(join + chunk.text)
                    needs_new_msg_sep = False
                else:
                    _flush_text()
                    all_content.append(chunk)

    if not all_content:
        # Only text encountered: return as str
        return "".join(cur_text_parts)

    _flush_text()

    return all_content


class InstructRequestNormalizer(
    Generic[UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType, InstructRequestType]
):
    r"""Takes a [ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest] and normalizes
    it into an [InstructRequest][mistral_common.protocol.instruct.request.InstructRequest].

    The normalization process does several things such as:
    - Aggregate consecutive messages of the same role
    - Aggregate system prompts
    - Normalize json content
    - Normalize tool calls

    Examples:
        >>> normalizer = InstructRequestNormalizer.normalizer()
    """

    _system_prompt_in_begin: bool = False
    _allow_tool_call_and_content: bool = False
    _chunk_join_str: str = _DEFAULT_JOIN_STR
    _msg_join_str: str = _DEFAULT_JOIN_STR

    def __init__(
        self,
        user_message_class: type[UserMessageType],
        assistant_message_class: type[AssistantMessageType],
        tool_message_class: type[ToolMessageType],
        system_message_class: type[SystemMessageType],
        instruct_request_class: type[InstructRequestType],
        model_settings_builder: ModelSettingsBuilder | None,
    ):
        r"""Initializes the normalizer with the appropriate message classes.

        Args:
           user_message_class: The class for user messages.
           assistant_message_class: The class for assistant messages.
           tool_message_class: The class for tool messages.
           system_message_class: The class for system messages.
           instruct_request_class: The class for instruct requests.
           model_settings_builder: The builder for model settings, or None if unsupported.
        """
        self._user_message_class = user_message_class
        self._assistant_message_class = assistant_message_class
        self._tool_message_class = tool_message_class
        self._instruct_request_class = instruct_request_class
        # this is unused but makes creation nicer
        self._system_message_class = system_message_class
        self._model_settings_builder = model_settings_builder

    @staticmethod
    def normalizer(model_settings_builder: ModelSettingsBuilder | None = None) -> "InstructRequestNormalizer":
        r"""Returns a normalizer for the default instruct request.

        Args:
            model_settings_builder: Must be None for this normalizer version.

        Returns:
            A normalizer for the default instruct request.

        Raises:
            ValueError: If model_settings_builder is not None.

        Examples:
            >>> normalizer = InstructRequestNormalizer.normalizer()
        """
        if model_settings_builder is not None:
            raise ValueError(
                f"model_settings_builder must be None for InstructRequestNormalizer, got {model_settings_builder}"
            )
        return InstructRequestNormalizer(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest[UATS, Tool], None
        )

    def build_settings(self, request: ChatCompletionRequest) -> ModelSettings:
        r"""Build model settings from a chat completion request.

        For pre-v15 normalizers, model settings are all `None`.

        Args:
            request: The chat completion request.

        Returns:
            Returns `ModelSettings.none()`.
        """
        if self._model_settings_builder is not None:
            raise InvalidRequestException(
                f"model_settings_builder should be None for {type(self).__name__}, got {self._model_settings_builder}"
            )
        return ModelSettings.none()

    def _normalize_json_content(self, content: str | None) -> str:
        if content is None or len(content) == 0:
            return "{}"

        try:
            parsed_json = json.loads(content)
            normalized_content = json.dumps(parsed_json, ensure_ascii=False)
        except json.JSONDecodeError:
            normalized_content = content
        return normalized_content

    def _aggregate_content_chunks(self, messages: list[UATS]) -> list[ContentChunk] | str:
        """Coalesce neighboring blocks of ContentChunks across messages."""
        return _aggregate_content_chunks_impl(
            [message.content for message in messages],
            msg_join_str=self._msg_join_str,
            chunk_join_str=self._chunk_join_str,
        )

    def _aggregate_content_chunks_to_str_same_message(self, message: UATS) -> str:
        """Aggregate a single message's content chunks to a string.

        Args:
            message: A single message with role system or tool.

        Returns:
            The aggregated content as a string.
        """
        assert message.role in (Roles.system, Roles.tool), message.role
        aggregated = self._aggregate_content_chunks([message])
        assert isinstance(aggregated, str), aggregated
        return aggregated

    def _aggregate_system_prompts(self, messages: list[UATS]) -> str | None:
        system_prompt: list[str] = []

        for message in messages:
            if message.role == Roles.system and message.content:
                system_prompt.append(self._aggregate_content_chunks_to_str_same_message(message))

        return self._msg_join_str.join(system_prompt) if len(system_prompt) else None

    def _aggregate_tool_messages(self, messages: list[UATS], latest_call_ids: list[str]) -> list[ToolMessageType]:
        """Normalize tool messages without aggregation across messages.

        Each tool message's content is validated and JSON-normalized.
        """
        tool_messages: list[ToolMessageType] = []
        for message in messages:
            assert isinstance(message, self._tool_message_class), "Expected tool message"
            content = self._aggregate_content_chunks([message])
            if not isinstance(content, str):
                raise InvalidRequestException(
                    f"Unexpected content chunk types in tool message: {[type(c).__name__ for c in content]}"
                )
            normalized_content = self._normalize_json_content(content)

            tool_messages.append(
                self._tool_message_class(
                    content=normalized_content, tool_call_id=message.tool_call_id, name=message.name
                )
            )

        return tool_messages

    def _normalize_tool_call(self, tool_call: ToolCall) -> ToolCall:
        normalized_function_aruments = self._normalize_json_content(tool_call.function.arguments)
        return ToolCall(
            function=FunctionCall(name=tool_call.function.name, arguments=normalized_function_aruments),
            id=tool_call.id,
        )

    def _narrow_assistant_content(self, content: list[ContentChunk] | str) -> str | list[AssistantContentChunk]:
        r"""Validate and narrow content chunks for assistant messages.

        Only TextChunk and ThinkChunk are allowed.

        Args:
            content: The aggregated content chunks.

        Returns:
            The validated and narrowed content.

        Raises:
            InvalidRequestException: If unsupported chunk types are found.
        """
        if isinstance(content, str) or _is_assistant_content(content):
            return content
        raise InvalidRequestException(
            f"Unexpected content chunk types in assistant message: {[type(c).__name__ for c in content]}"
        )

    def _aggregate_system_messages(self, messages: list[UATS]) -> list[SystemMessageType]:
        return []

    def _aggregate_assistant_messages(self, messages: list[UATS]) -> AssistantMessageType:
        tool_calls: list[ToolCall] = []
        prefix: bool = False
        weight: float | None = None

        content = self._aggregate_content_chunks(messages)

        for message in messages:
            assert isinstance(message, self._assistant_message_class), "Expected assistant message"

            if not self._allow_tool_call_and_content and (message.tool_calls and message.content):
                raise ValueError(f"Tool calls and content cannot be used together in the same message. {message}")

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    normalized_tool_call = self._normalize_tool_call(tool_call)
                    tool_calls.append(normalized_tool_call)

            prefix |= message.prefix

            if isinstance(message, FinetuningAssistantMessage):
                if weight is not None:
                    assert weight == message.weight, (
                        "Expected weights of aggregated FinetuningAssistantMessage to be equal"
                    )
                weight = message.weight

        validated_content = self._narrow_assistant_content(content)

        aggregated_message = self._assistant_message_class(
            content=validated_content,
            tool_calls=tool_calls or None,
            prefix=prefix,
        )

        if weight is not None and hasattr(aggregated_message, "weight"):
            aggregated_message.weight = weight
        return aggregated_message

    def _aggregate_user_messages(self, messages: list[UATS]) -> UserMessageType:
        """Coalesce neighboring blocks of ContentChunks in user messages."""
        content = self._aggregate_content_chunks(messages)
        if isinstance(content, str) or _is_user_content(content):
            return self._user_message_class(content=content)
        raise InvalidRequestException(
            f"Unexpected content chunk types in user message: {[type(c).__name__ for c in content]}"
        )

    def _aggregate_role(self, messages: list[UATS], role: Roles | None, latest_call_ids: list[str]) -> Sequence[UATS]:
        if role == Roles.tool:
            return self._aggregate_tool_messages(messages, latest_call_ids)
        elif role == Roles.assistant:
            return [self._aggregate_assistant_messages(messages)]
        elif role == Roles.user:
            return [self._aggregate_user_messages(messages)]
        else:  # System messages are ignored
            return self._aggregate_system_messages(messages)

    def _aggregate_messages(self, messages: list[UATS]) -> list[UATS]:
        aggregated_messages: list[UATS] = []
        messages_to_aggregate: list[UATS] = []
        current_role: Roles | None = None
        current_weight: float | None = None
        latest_call_ids: list[str] = []

        # Collect consecutive lists of messages with the same role and weight
        for message in messages:
            new_weight = getattr(message, "weight", None)
            if current_role != message.role or (new_weight != current_weight):
                aggregated_messages.extend(self._aggregate_role(messages_to_aggregate, current_role, latest_call_ids))

                if current_role == Roles.assistant:
                    assistant_message = aggregated_messages[-1]
                    assert isinstance(aggregated_messages[-1], AssistantMessage)
                    if assistant_message.tool_calls is not None:
                        for tool_call in assistant_message.tool_calls:
                            latest_call_ids.append(tool_call.id)

                elif current_role == Roles.tool:
                    # reordering only depends on the latest aggregated assistant message
                    latest_call_ids.clear()

                messages_to_aggregate.clear()
            current_weight = new_weight
            current_role = message.role
            messages_to_aggregate.append(message)

        # Add the last set of messages
        aggregated_messages.extend(self._aggregate_role(messages_to_aggregate, current_role, latest_call_ids))

        # If the first message is not a user message, or we didn't aggregate
        # anything (all system messages) for example, add an empty user message
        if len(aggregated_messages) == 0 or (
            not self._system_prompt_in_begin and aggregated_messages[0].role != Roles.user
        ):
            aggregated_messages.insert(0, self._user_message_class(content=""))

        return aggregated_messages

    def from_chat_completion_request(self, request: ChatCompletionRequest[UATS]) -> InstructRequestType:
        r"""Converts a chat completion request to an instruct request.

        Args:
            request: The chat completion request to convert.

        Returns:
            The converted instruct request.

        Examples:
            >>> from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
            >>> request = ChatCompletionRequest(
            ...     messages=[
            ...         UserMessage(content="Hello"),
            ...         AssistantMessage(content="Hi"),
            ...     ],
            ... )
            >>> normalizer = InstructRequestNormalizer.normalizer()
            >>> instruct_request = normalizer.from_chat_completion_request(request)
        """
        system_prompt = self._aggregate_system_prompts(request.messages)
        messages = self._aggregate_messages(request.messages)

        settings = self.build_settings(request)
        if settings != ModelSettings.none():
            raise InvalidRequestException(f"Model settings are not supported for {type(self).__name__}, got {settings}")

        return self._instruct_request_class(
            messages=messages,
            system_prompt=system_prompt,
            available_tools=request.tools,
            continue_final_message=request.continue_final_message,
            settings=settings,
        )


class InstructRequestNormalizerV7(InstructRequestNormalizer):
    r"""Normalizer for the v7 tokenizer.

    Examples:
        >>> normalizer = InstructRequestNormalizerV7.normalizer()
    """

    _system_prompt_in_begin: bool = True
    _allow_tool_call_and_content: bool = True

    @staticmethod
    def normalizer(model_settings_builder: ModelSettingsBuilder | None = None) -> "InstructRequestNormalizerV7":
        r"""Returns a normalizer for the default instruct request.

        Args:
            model_settings_builder: Must be None for this normalizer version.

        Returns:
            A normalizer for the V7 instruct request.

        Raises:
            ValueError: If model_settings_builder is not None.

        Examples:
            >>> normalizer = InstructRequestNormalizerV7.normalizer()
        """
        if model_settings_builder is not None:
            raise ValueError(
                f"model_settings_builder must be None for InstructRequestNormalizerV7, got {model_settings_builder}"
            )
        return InstructRequestNormalizerV7(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest[UATS, Tool], None
        )

    def _narrow_system_content(self, content: list[ContentChunk] | str) -> str | list[SystemContentChunk]:
        r"""Validate content chunks for system messages.

        V7+ accepts all SystemContentChunk types (text, audio, thinking).
        V15 overrides to reject ThinkChunk.

        Args:
            content: The aggregated content chunks.

        Returns:
            The validated content.

        Raises:
            InvalidRequestException: If unsupported chunk types are found.
        """
        if isinstance(content, str) or _is_system_content(content):
            return content
        raise InvalidRequestException(
            f"Unexpected content chunk types in system message: {[type(c).__name__ for c in content]}"
        )

    def _aggregate_tool_messages(self, messages: list[UATS], latest_call_ids: list[str]) -> list[ToolMessageType]:
        """Normalize tool messages without JSON normalization.

        V7+ normalizers skip JSON content normalization for tool messages but still
        reject non-text content chunks.
        """
        tool_messages: list[ToolMessageType] = []
        for message in messages:
            assert isinstance(message, self._tool_message_class), "Expected tool message"
            content = self._aggregate_content_chunks([message])
            if not isinstance(content, str):
                raise InvalidRequestException(
                    f"Unexpected content chunk types in tool message: {[type(c).__name__ for c in content]}"
                )
            tool_messages.append(
                self._tool_message_class(content=content, tool_call_id=message.tool_call_id, name=message.name)
            )
        return tool_messages

    def _aggregate_system_messages(self, messages: list[UATS]) -> list[SystemMessageType]:
        aggregated: list[SystemMessageType] = []
        for message in messages:
            if isinstance(message, self._system_message_class):
                content = self._aggregate_content_chunks([message])
                validated = self._narrow_system_content(content)
                aggregated.append(self._system_message_class(content=validated))
        return aggregated

    def _aggregate_role(self, messages: list[UATS], role: Roles | None, latest_call_ids: list[str]) -> Sequence[UATS]:
        if role == Roles.tool:
            return self._aggregate_tool_messages(messages, latest_call_ids)
        elif role == Roles.assistant:
            return [self._aggregate_assistant_messages(messages)]
        elif role == Roles.user:
            return [self._aggregate_user_messages(messages)]
        elif role == Roles.system:
            return self._aggregate_system_messages(messages)
        else:
            assert role is None and len(messages) == 0
            return []

    def _aggregate_system_prompts(self, messages: list[UATS]) -> str | None:
        raise NotImplementedError("We should not aggregate system prompts")

    def from_chat_completion_request(self, request: ChatCompletionRequest[UATS]) -> InstructRequestType:  # type: ignore[type-var, misc]
        r"""Converts a chat completion request to an instruct request.

        Args:
            request: The chat completion request to convert.

        Returns:
            The converted instruct request.

        Examples:
            >>> from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
            >>> request = ChatCompletionRequest(
            ...     messages=[
            ...         UserMessage(content="Hello"),
            ...         AssistantMessage(content="Hi"),
            ...     ],
            ... )
            >>> normalizer = InstructRequestNormalizerV7.normalizer()
            >>> instruct_request = normalizer.from_chat_completion_request(request)
        """
        messages = self._aggregate_messages(request.messages)
        settings = self.build_settings(request)
        if settings != ModelSettings.none():
            raise InvalidRequestException(f"Model settings are not supported for {type(self).__name__}, got {settings}")
        return self._instruct_request_class(  # type: ignore[no-any-return]
            messages=messages,
            system_prompt=None,
            available_tools=request.tools,
            continue_final_message=request.continue_final_message,
            settings=settings,
        )


class InstructRequestNormalizerV13(InstructRequestNormalizerV7):
    r"""Normalizer for the v13 tokenizer.

    It reorders tool messages based on the tool call order.

    Examples:
        >>> normalizer = InstructRequestNormalizerV13.normalizer()
    """

    @staticmethod
    def normalizer(model_settings_builder: ModelSettingsBuilder | None = None) -> "InstructRequestNormalizerV13":
        r"""Returns a normalizer for the default instruct request.

        Args:
            model_settings_builder: Must be None for this normalizer version.

        Returns:
            A normalizer for the V13 instruct request.

        Raises:
            ValueError: If model_settings_builder is not None.
        """
        if model_settings_builder is not None:
            raise ValueError(
                f"model_settings_builder must be None for InstructRequestNormalizerV13, got {model_settings_builder}"
            )
        return InstructRequestNormalizerV13(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest[UATS, Tool], None
        )

    @staticmethod
    def _inplace_sort_tool_messages(tool_messages: list[ToolMessageType], latest_call_ids: list[str]) -> None:
        id_to_tool_call_idx = {call_id: idx for idx, call_id in enumerate(latest_call_ids)}
        id_to_tool_result_idx = {message.tool_call_id: idx for idx, message in enumerate(tool_messages)}
        # First order by tool call idx and then by tool result idx
        tool_messages.sort(
            key=lambda msg: (
                id_to_tool_call_idx.get(msg.tool_call_id or "null", float("inf")),
                id_to_tool_result_idx[msg.tool_call_id],
            ),
        )

    def _aggregate_tool_messages(self, messages: list[UATS], latest_call_ids: list[str]) -> list[ToolMessageType]:
        tool_messages: list[ToolMessageType] = super()._aggregate_tool_messages(messages, latest_call_ids)
        self._inplace_sort_tool_messages(tool_messages=tool_messages, latest_call_ids=latest_call_ids)
        return tool_messages


class InstructRequestNormalizerV15(InstructRequestNormalizerV13):
    r"""Normalizer for the v15 tokenizer.

    It reorders tool messages based on the tool call order and builds model settings.

    Examples:
        >>> normalizer = InstructRequestNormalizerV15.normalizer()
    """

    _chunk_join_str: str = ""

    def _aggregate_tool_messages(self, messages: list[UATS], latest_call_ids: list[str]) -> list[ToolMessageType]:
        r"""V15 accepts all ContentChunk types in tool messages."""
        tool_messages: list[ToolMessageType] = []
        for message in messages:
            assert isinstance(message, self._tool_message_class), "Expected tool message"
            content = self._aggregate_content_chunks([message])
            tool_messages.append(
                self._tool_message_class(content=content, tool_call_id=message.tool_call_id, name=message.name)
            )
        self._inplace_sort_tool_messages(tool_messages=tool_messages, latest_call_ids=latest_call_ids)
        return tool_messages

    def _narrow_system_content(self, content: list[ContentChunk] | str) -> str | list[SystemContentChunk]:
        r"""V15 system messages allow TextChunk and AudioChunk but reject ThinkChunk."""
        validated = super()._narrow_system_content(content)
        if isinstance(validated, str):
            return validated
        if any(isinstance(c, ThinkChunk) for c in validated):
            raise InvalidRequestException("ThinkChunk in system message is not supported for V15")
        return validated

    @staticmethod
    def normalizer(model_settings_builder: ModelSettingsBuilder | None = None) -> "InstructRequestNormalizerV15":
        r"""Returns a normalizer for the V15 instruct request.

        Args:
            model_settings_builder: The builder for model settings.

        Returns:
            A normalizer for the V15 instruct request.
        """
        return InstructRequestNormalizerV15(
            UserMessage,
            AssistantMessage,
            ToolMessage,
            SystemMessage,
            InstructRequest,
            model_settings_builder,
        )

    def build_settings(self, request: ChatCompletionRequest) -> ModelSettings:
        r"""Build model settings using the configured model settings builder.

        Args:
            request: The chat completion request.

        Returns:
            The built model settings.

        Raises:
            InvalidRequestException: If no model settings builder is configured.
        """
        if self._model_settings_builder is None:
            raise InvalidRequestException(f"model_settings_builder must not be None for {type(self).__name__}")
        return self._model_settings_builder.build_settings(request)

    def from_chat_completion_request(self, request: ChatCompletionRequest[UATS]) -> InstructRequestType:  # type: ignore[type-var, misc]
        r"""Converts a chat completion request to an instruct request.

        Args:
            request: The chat completion request to convert.

        Returns:
            The converted instruct request.
        """
        messages = self._aggregate_messages(request.messages)
        settings = self.build_settings(request)
        return self._instruct_request_class(  # type: ignore[no-any-return]
            messages=messages,
            system_prompt=None,
            available_tools=request.tools,
            continue_final_message=request.continue_final_message,
            settings=settings,
        )


def normalizer_for_tokenizer_version(
    version: TokenizerVersion, model_settings_builder: ModelSettingsBuilder | None = None
) -> InstructRequestNormalizer:
    r"""Deprecated in favor to `get_normalizer`, will be removed in 1.12.0."""
    warnings.warn(
        "`normalizer_for_tokenizer_version` is deprecated and will be removed in 1.12.0. "
        "Please call `get_normalizer` instead.",
        FutureWarning,
    )
    return get_normalizer(version=version, model_settings_builder=model_settings_builder)


def get_normalizer(
    version: TokenizerVersion, model_settings_builder: ModelSettingsBuilder | None = None
) -> InstructRequestNormalizer:
    r"""Gets the appropriate normalizer for the given tokenizer version.

    Args:
        version: The tokenizer version to get the normalizer for.
        model_settings_builder: The builder for model settings, or None if unsupported.

    Returns:
        The appropriate normalizer for the given tokenizer version.

    Examples:
        >>> normalizer = get_normalizer(TokenizerVersion.v1)
    """
    match version:
        case TokenizerVersion.v1 | TokenizerVersion.v2 | TokenizerVersion.v3:
            normalizer_cls = InstructRequestNormalizer
        case TokenizerVersion.v7:
            normalizer_cls = InstructRequestNormalizerV7
        case TokenizerVersion.v11 | TokenizerVersion.v13:
            normalizer_cls = InstructRequestNormalizerV13
        case TokenizerVersion.v15:
            normalizer_cls = InstructRequestNormalizerV15
        case _:
            assert_never(version)

    return normalizer_cls.normalizer(model_settings_builder=model_settings_builder)

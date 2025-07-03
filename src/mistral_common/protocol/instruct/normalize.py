import json
from typing import Generic, List, Optional, Sequence, Type, Union

from mistral_common.protocol.instruct.messages import (
    UATS,
    AssistantMessage,
    AssistantMessageType,
    ContentChunk,
    FinetuningAssistantMessage,
    Roles,
    SystemMessage,
    SystemMessageType,
    TextChunk,
    ToolMessage,
    ToolMessageType,
    UserMessage,
    UserMessageType,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, Tool, ToolCall
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import InstructRequestType, TokenizerVersion


class InstructRequestNormalizer(
    Generic[UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType, InstructRequestType]
):
    r"""Takes a [ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest] and normalizes
    it into an [InstructRequest][mistral_common.tokens.instruct.request.InstructRequest].

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

    def __init__(
        self,
        user_message_class: Type[UserMessageType],
        assistant_message_class: Type[AssistantMessageType],
        tool_message_class: Type[ToolMessageType],
        system_message_class: Type[SystemMessageType],
        instruct_request_class: Type[InstructRequestType],
    ):
        r"""Initializes the normalizer with the appropriate message classes.

        Args:
           user_message_class: The class for user messages.
           assistant_message_class: The class for assistant messages.
           tool_message_class: The class for tool messages.
           system_message_class: The class for system messages.
           instruct_request_class: The class for instruct requests.
        """
        self._user_message_class = user_message_class
        self._assistant_message_class = assistant_message_class
        self._tool_message_class = tool_message_class
        self._instruct_request_class = instruct_request_class
        # this is unused but makes creation nicer
        self._system_message_class = system_message_class

    @staticmethod
    def normalizer() -> "InstructRequestNormalizer":
        r"""Returns a normalizer for the default instruct request.

        Examples:
            >>> normalizer = InstructRequestNormalizer.normalizer()
        """
        return InstructRequestNormalizer(
            UserMessage,
            AssistantMessage,
            ToolMessage,
            SystemMessage,
            InstructRequest[UATS, Tool],
        )

    def _normalize_json_content(self, content: Optional[str]) -> str:
        if content is None or len(content) == 0:
            return "{}"

        try:
            parsed_json = json.loads(content)
            normalized_content = json.dumps(parsed_json, ensure_ascii=False)
        except json.JSONDecodeError:
            normalized_content = content
        return normalized_content

    def _aggregate_content_chunks(self, content: Union[str, List[TextChunk]], chunk_join_str: str = "\n\n") -> str:
        if isinstance(content, list):
            return chunk_join_str.join([chunk.text for chunk in content])
        else:
            return content

    def _aggregate_system_prompts(self, request: ChatCompletionRequest[UATS]) -> Optional[str]:
        system_prompt: List[str] = []

        for message in request.messages:
            if message.role == Roles.system and message.content:
                system_prompt.append(self._aggregate_content_chunks(message.content))

        return "\n\n".join(system_prompt) if len(system_prompt) else None

    def _aggregate_tool_messages(self, messages: List[UATS], latest_call_ids: List[str]) -> List[ToolMessageType]:
        r"""
        We currently do not do any aggregation for tool messages, but we normalize the json content
        """
        tool_messages: List[ToolMessageType] = []
        for message in messages:
            assert isinstance(message, self._tool_message_class), "Expected tool message"
            content = self._aggregate_content_chunks(message.content)
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

    def _aggregate_assistant_messages(self, messages: List[UATS]) -> AssistantMessageType:
        aggregated_content: List[str] = []
        tool_calls: List[ToolCall] = []
        prefix: bool = False
        weight: Optional[float] = None
        for message in messages:
            assert isinstance(message, self._assistant_message_class), "Expected assistant message"

            if not self._allow_tool_call_and_content and (message.tool_calls and message.content):
                raise ValueError(f"Tool calls and content cannot be used together in the same message. {message}")

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    normalized_tool_call = self._normalize_tool_call(tool_call)
                    tool_calls.append(normalized_tool_call)

            if message.content:
                aggregated_content.append(self._aggregate_content_chunks(message.content))

            prefix |= message.prefix
            if isinstance(message, FinetuningAssistantMessage):
                # Only FinetuningAssistantMessage can be weighted
                if weight is not None:
                    assert weight == message.weight, (
                        "Expected weights of aggregated FinetuningAssistantMessage to be equal"
                    )
                weight = message.weight

        aggregated_message = self._assistant_message_class(
            content="\n\n".join(aggregated_content) if len(aggregated_content) else None,
            tool_calls=tool_calls or None,
            prefix=prefix,
        )

        if weight is not None and hasattr(aggregated_message, "weight"):
            aggregated_message.weight = weight
        return aggregated_message

    def _aggregate_user_messages(self, messages: List[UATS]) -> UserMessageType:
        """
        Just coalesce neighboring blocks of text
        """
        all_content: List[ContentChunk] = []
        text_chunks: List[str] = []
        for message in messages:
            assert isinstance(message, self._user_message_class), f"Expected user message got {type(message)}"
            if isinstance(message.content, str):
                text_chunks.append(message.content)
            else:  # it's a List[ContentChunk]
                for chunk in message.content:
                    if isinstance(chunk, TextChunk):
                        text_chunks.append(chunk.text)
                    else:
                        if text_chunks:
                            all_content.append(TextChunk(text="\n\n".join(text_chunks)))
                            text_chunks = []
                        all_content.append(chunk)

        text_content = "\n\n".join(text_chunks) if text_chunks else ""

        if not all_content:
            # if no ContentChunk was passed, we return content as a str
            return self._user_message_class(content=text_content)

        if text_content:
            # else we return a List of content chunks
            all_content.append(TextChunk(text=text_content))

        return self._user_message_class(content=all_content)

    def _aggregate_role(
        self, messages: List[UATS], role: Optional[Roles], latest_call_ids: List[str]
    ) -> Sequence[UATS]:
        if role == Roles.tool:
            return self._aggregate_tool_messages(messages, latest_call_ids)
        elif role == Roles.assistant:
            return [self._aggregate_assistant_messages(messages)]
        elif role == Roles.user:
            return [self._aggregate_user_messages(messages)]
        else:  # System messages are ignored
            return []

    def _aggregate_messages(self, request: ChatCompletionRequest[UATS]) -> List[UATS]:
        aggregated_messages: List[UATS] = []
        messages_to_aggregate: List[UATS] = []
        current_role: Optional[Roles] = None
        current_weight: Optional[float] = None
        latest_call_ids: List[str] = []

        # Collect consecutive lists of messages with the same role and weight
        for message in request.messages:
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
        system_prompt = self._aggregate_system_prompts(request)
        messages = self._aggregate_messages(request)

        return self._instruct_request_class(
            messages=messages,
            system_prompt=system_prompt,
            available_tools=request.tools,
            continue_final_message=request.continue_final_message,
        )


class InstructRequestNormalizerV7(InstructRequestNormalizer):
    r"""Normalizer for the v7 tokenizer.

    Examples:
        >>> normalizer = InstructRequestNormalizerV7.normalizer()
    """

    _system_prompt_in_begin: bool = True
    _allow_tool_call_and_content: bool = True

    @staticmethod
    def normalizer() -> "InstructRequestNormalizerV7":
        r"""Returns a normalizer for the default instruct request

        Examples:
            >>> normalizer = InstructRequestNormalizerV7.normalizer()
        """
        return InstructRequestNormalizerV7(
            UserMessage,
            AssistantMessage,
            ToolMessage,
            SystemMessage,
            InstructRequest[UATS, Tool],
        )

    def _aggregate_role(
        self, messages: List[UATS], role: Optional[Roles], latest_call_ids: list[str]
    ) -> Sequence[UATS]:
        if role == Roles.tool:
            return self._aggregate_tool_messages(messages, latest_call_ids)
        elif role == Roles.assistant:
            return [self._aggregate_assistant_messages(messages)]
        elif role == Roles.user:
            return [self._aggregate_user_messages(messages)]
        elif role == Roles.system:
            return messages
        else:
            assert role is None and len(messages) == 0
            return []

    def _aggregate_system_prompts(self, request: ChatCompletionRequest[UATS]) -> Optional[str]:
        raise NotImplementedError("We should not aggregate system prompts")

    def from_chat_completion_request(self, request: ChatCompletionRequest[UATS]) -> InstructRequestType:  # type: ignore[type-var]
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
        messages = self._aggregate_messages(request)
        return self._instruct_request_class(messages=messages, system_prompt=None, available_tools=request.tools)  # type: ignore[no-any-return]


class InstructRequestNormalizerV13(InstructRequestNormalizerV7):
    r"""Normalizer for the v13 tokenizer.

    It reorders tool messages based on the tool call order.

    Examples:
        >>> normalizer = InstructRequestNormalizerV13.normalizer()
    """

    @staticmethod
    def normalizer() -> "InstructRequestNormalizerV13":
        r"""Returns a normalizer for the default instruct request."""
        return InstructRequestNormalizerV13(
            UserMessage,
            AssistantMessage,
            ToolMessage,
            SystemMessage,
            InstructRequest[UATS, Tool],
        )

    def _aggregate_tool_messages(self, messages: List[UATS], latest_call_ids: List[str]) -> List[ToolMessageType]:
        tool_messages: List[ToolMessageType] = super()._aggregate_tool_messages(messages, latest_call_ids)
        id_to_tool_call_idx = {call_id: idx for idx, call_id in enumerate(latest_call_ids)}
        id_to_tool_result_idx = {message.tool_call_id: idx for idx, message in enumerate(tool_messages)}
        # First order by tool call idx and then by tool result idx
        tool_messages.sort(
            key=lambda msg: (
                id_to_tool_call_idx.get(msg.tool_call_id or "null", float("inf")),
                id_to_tool_result_idx[msg.tool_call_id],
            ),
        )
        return tool_messages


def normalizer_for_tokenizer_version(version: TokenizerVersion) -> InstructRequestNormalizer:
    r"""Gets the appropriate normalizer for the given tokenizer version.

    Args:
        version: The tokenizer version to get the normalizer for.

    Returns:
        The appropriate normalizer for the given tokenizer version.

    Examples:
        >>> normalizer = normalizer_for_tokenizer_version(TokenizerVersion.v1)
    """
    if version in {TokenizerVersion.v1, TokenizerVersion.v2, TokenizerVersion.v3}:
        return InstructRequestNormalizer.normalizer()
    elif version in {TokenizerVersion.v7, TokenizerVersion.v11}:
        return InstructRequestNormalizerV7.normalizer()
    elif version == TokenizerVersion.v13:
        return InstructRequestNormalizerV13.normalizer()
    raise ValueError(f"Unknown tokenizer version {version}")

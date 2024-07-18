import json
from typing import Generic, List, Optional, Sequence, Type, Union

from mistral_common.protocol.instruct.messages import (
    UATS,
    AssistantMessage,
    AssistantMessageType,
    ContentChunk,
    Roles,
    SystemMessage,
    SystemMessageType,
    ToolMessage,
    ToolMessageType,
    UserMessage,
    UserMessageType,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, Tool, ToolCall
from mistral_common.tokens.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import InstructRequestType


class InstructRequestNormalizer(
    Generic[UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType, InstructRequestType]
):
    """
    Takes a ChatCompletionRequest and normalizes it into an InstructRequest.

    The normalization process does several things such as:
    - Aggregate consecutive messages of the same role
    - Aggregate system prompts
    - Normalize json content
    - Normalize tool calls
    """

    def __init__(
        self,
        user_message_class: Type[UserMessageType],
        assistant_message_class: Type[AssistantMessageType],
        tool_message_class: Type[ToolMessageType],
        system_message_class: Type[SystemMessageType],
        instruct_request_class: Type[InstructRequestType],
    ):
        self._user_message_class = user_message_class
        self._assistant_message_class = assistant_message_class
        self._tool_message_class = tool_message_class
        self._instruct_request_class = instruct_request_class
        # this is unused but makes creation nicer
        self._system_message_class = system_message_class

    @staticmethod
    def normalizer() -> "InstructRequestNormalizer":
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

    def _aggregate_content_chunks(self, content: Union[str, List[ContentChunk]], chunk_join_str: str = "\n\n") -> str:
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

    def _aggregate_tool_messages(self, messages: List[UATS]) -> List[ToolMessageType]:
        """
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
        for message in messages:
            assert isinstance(message, self._assistant_message_class), "Expected assistant message"
            if message.tool_calls is not None:
                for tool_call in message.tool_calls:
                    normalized_tool_call = self._normalize_tool_call(tool_call)
                    tool_calls.append(normalized_tool_call)
            elif message.content:
                aggregated_content.append(self._aggregate_content_chunks(message.content))
            prefix |= message.prefix

        return self._assistant_message_class(
            content="\n\n".join(aggregated_content) if len(aggregated_content) else None,
            tool_calls=tool_calls or None,
            prefix=prefix,
        )

    def _aggregate_user_messages(self, messages: List[UATS]) -> UserMessageType:
        aggregated_content: List[str] = []
        for message in messages:
            assert isinstance(message, self._user_message_class), "Expected user message"
            content = self._aggregate_content_chunks(message.content)
            if content:
                aggregated_content.append(content)

        aggregated_content_str = "\n\n".join(aggregated_content)
        return self._user_message_class(content=aggregated_content_str)

    def _aggregate_role(self, messages: List[UATS], role: Optional[Roles]) -> Sequence[UATS]:
        if role == Roles.tool:
            return self._aggregate_tool_messages(messages)
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

        # Collect consecutive lists of messages with the same role
        for message in request.messages:
            if current_role != message.role:
                aggregated_messages.extend(self._aggregate_role(messages_to_aggregate, current_role))
                messages_to_aggregate.clear()

            current_role = message.role
            messages_to_aggregate.append(message)

        # Add the last set of messages
        aggregated_messages.extend(self._aggregate_role(messages_to_aggregate, current_role))

        # If the first message is not a user message, or we didnt aggregate
        # anything (all system messages) for example, add an empty user message
        if len(aggregated_messages) == 0 or aggregated_messages[0].role != Roles.user:
            aggregated_messages.insert(0, self._user_message_class(content=""))

        return aggregated_messages

    def from_chat_completion_request(self, request: ChatCompletionRequest[UATS]) -> InstructRequestType:
        system_prompt = self._aggregate_system_prompts(request)
        messages = self._aggregate_messages(request)

        return self._instruct_request_class(
            messages=messages, system_prompt=system_prompt, available_tools=request.tools
        )

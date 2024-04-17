import json
from typing import List, Optional, Sequence, Union

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ContentChunk,
    Roles,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ChatMessage
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.instruct.request import InstructRequest


class InstructRequestNormalizer:
    """
    Takes a ChatCompletionRequest and normalizes it into an InstructRequest.

    The normalization process does several things such as:
    - Aggregate consecutive messages of the same role
    - Aggregate system prompts
    - Normalize json content
    - Normalize tool calls
    """

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

    def _aggregate_system_prompts(self, request: ChatCompletionRequest) -> Optional[str]:
        system_prompt: List[str] = []

        for message in request.messages:
            if message.role == Roles.system and message.content:
                system_prompt.append(self._aggregate_content_chunks(message.content))

        return "\n\n".join(system_prompt) if len(system_prompt) else None

    def _aggregate_tool_messages(self, messages: List[ChatMessage]) -> List[ToolMessage]:
        """
        We currently do not do any aggregation for tool messages, but we normalize the json content
        """
        tool_messages: List[ToolMessage] = []
        for message in messages:
            assert isinstance(message, ToolMessage), "Expected tool message"
            content = self._aggregate_content_chunks(message.content)
            normalized_content = self._normalize_json_content(content)
            tool_messages.append(
                ToolMessage(content=normalized_content, tool_call_id=message.tool_call_id, name=message.name)
            )

        return tool_messages

    def _normalize_tool_call(self, tool_call: ToolCall) -> ToolCall:
        normalized_function_aruments = self._normalize_json_content(tool_call.function.arguments)
        return ToolCall(
            function=FunctionCall(name=tool_call.function.name, arguments=normalized_function_aruments),
            id=tool_call.id,
        )

    def _aggregate_assistant_messages(self, messages: List[ChatMessage]) -> AssistantMessage:
        aggregated_content: List[str] = []
        tool_calls: List[ToolCall] = []
        for message in messages:
            assert isinstance(message, AssistantMessage), "Expected assistant message"
            if message.tool_calls is not None:
                for tool_call in message.tool_calls:
                    normalized_tool_call = self._normalize_tool_call(tool_call)
                    tool_calls.append(normalized_tool_call)
            elif message.content:
                aggregated_content.append(self._aggregate_content_chunks(message.content))

        return AssistantMessage(
            content="\n\n".join(aggregated_content) if len(aggregated_content) else None,
            tool_calls=tool_calls or None,
        )

    def _aggregate_user_messages(self, messages: List[ChatMessage]) -> UserMessage:
        aggregated_content: List[str] = []
        for message in messages:
            assert isinstance(message, UserMessage), "Expected user message"
            content = self._aggregate_content_chunks(message.content)
            if content:
                aggregated_content.append(content)

        aggregated_content_str = "\n\n".join(aggregated_content)
        return UserMessage(content=aggregated_content_str)

    def _aggregate_role(self, messages: List[ChatMessage], role: Optional[Roles]) -> Sequence[ChatMessage]:
        if role == Roles.tool:
            return self._aggregate_tool_messages(messages)
        elif role == Roles.assistant:
            return [self._aggregate_assistant_messages(messages)]
        elif role == Roles.user:
            return [self._aggregate_user_messages(messages)]
        else: # System messages are ignored
            return []

    def _aggregate_messages(self, request: ChatCompletionRequest) -> List[ChatMessage]:
        aggregated_messages: List[ChatMessage] = []
        messages_to_aggregate: List[ChatMessage] = []
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
            aggregated_messages.insert(0, UserMessage(content=""))

        return aggregated_messages

    def from_chat_completion_request(self, request: ChatCompletionRequest) -> InstructRequest:
        system_prompt = self._aggregate_system_prompts(request)
        messages = self._aggregate_messages(request)

        return InstructRequest(messages=messages, system_prompt=system_prompt, available_tools=request.tools)

from typing import Any

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import Tool


def convert_openai_messages(
    messages: list[dict[str, str | list[dict[str, str | dict[str, Any]]]]],
) -> list[ChatMessage]:
    r"""Convert OpenAI messages to Mistral messages.

    Dispatches each message to the `from_openai` constructor of the matching
    role class.

    Args:
        messages: Message dicts matching OpenAI's chat schema. Each must have
            a "role" key of "user", "assistant", "tool", or "system".

    Returns:
        The Mistral message instances, one per input dict.

    Raises:
        ValueError: If a message has an unknown role.
    """
    converted_messages: list[ChatMessage] = []
    for openai_message in messages:
        message_role = openai_message.get("role")
        message: ChatMessage
        if message_role == "user":
            message = UserMessage.from_openai(openai_message)
        elif message_role == "assistant":
            message = AssistantMessage.from_openai(openai_message)
        elif message_role == "tool":
            message = ToolMessage.from_openai(openai_message)
        elif message_role == "system":
            message = SystemMessage.from_openai(openai_message)
        else:
            raise ValueError(f"Unknown message role: {message_role}")
        converted_messages.append(message)
    return converted_messages


def convert_openai_tools(
    tools: list[dict[str, Any]],
) -> list[Tool]:
    r"""Convert OpenAI tools to Mistral tools.

    Args:
        tools: Tool dicts matching OpenAI's tool schema, each with "type" and
            "function" keys.

    Returns:
        The Mistral Tool instances, one per input dict.
    """
    converted_tools = [Tool.from_openai(openai_tool) for openai_tool in tools]
    return converted_tools

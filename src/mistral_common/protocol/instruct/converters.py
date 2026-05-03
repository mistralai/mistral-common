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

    Args:
        messages: The OpenAI messages to convert.

    Returns:
        The Mistral messages.
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
        tools: The OpenAI tools to convert.

    Returns:
        The Mistral tools.
    """
    converted_tools = [Tool.from_openai(openai_tool) for openai_tool in tools]
    return converted_tools

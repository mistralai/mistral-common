from typing import Any, Dict, List, Set, Union

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import Tool


def convert_openai_messages(
    messages: List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]],
) -> List[ChatMessage]:
    r"""Convert OpenAI messages to Mistral messages.

    Args:
        messages: The OpenAI messages to convert.

    Returns:
        The Mistral messages.
    """
    converted_messages: List[ChatMessage] = []
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
    tools: List[Dict[str, Any]],
) -> List[Tool]:
    r"""Convert OpenAI tools to Mistral tools.

    Args:
        tools: The OpenAI tools to convert.

    Returns:
        The Mistral tools.
    """
    converted_tools = [Tool.from_openai(openai_tool) for openai_tool in tools]
    return converted_tools


def _check_openai_fields_names(valid_fields_names: Set[str], names: Set[str]) -> None:
    r"""Check if the names are valid field names.

    Names are valid if they are inside the `valid_fields_names` set or chat completion OpenAI fields. If the names are
    not valid field names, raise a ValueError.

    The error message will contain the invalid field names sorted by if they are openAI valid field names or not.

    Args:
        valid_fields_names: The valid field names.
        names: The names to check.

    Raises:
        ValueError: If the names are not valid field names.
    """

    openai_valid_params = set()
    non_valid_params = set()

    for name in names:
        if name in valid_fields_names:
            continue
        elif name in _OPENAI_COMPLETION_FIELDS:
            openai_valid_params.add(name)
        else:
            non_valid_params.add(name)

    if openai_valid_params or non_valid_params:
        raise ValueError(
            "Invalid parameters passed to `ChatCompletionRequest.from_openai`:\n"
            f"OpenAI valid parameters but not in `ChatCompletionRequest`: {openai_valid_params}\n"
            f"Non valid parameters: {non_valid_params}"
        )


def _is_openai_field_name(name: str) -> bool:
    return name in _OPENAI_COMPLETION_FIELDS


_OPENAI_COMPLETION_FIELDS: Set[str] = {
    "messages",
    "model",
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "web_search_options",
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout",
}

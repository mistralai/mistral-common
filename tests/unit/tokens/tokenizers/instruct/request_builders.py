r"""Shared instruct-request builders for versioned tokenizer tests.

These builders construct requests whose shape is identical across tokenizer versions.
Expected token IDs and decoded text stay in each version's test module.
"""

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool


def build_normal_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the multi-turn request without a system prompt."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ]
    )


def build_system_singleturn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the single-turn request with a system prompt."""
    return InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="a")], system_prompt="SYSTEM")


def build_system_multiturn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the multi-turn request with a system prompt."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ],
        system_prompt="SYSTEM",
    )


def build_prefixed_final_message_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request with a prefixed final assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d", prefix=True),
        ],
        system_prompt="SYSTEM",
    )


def build_tools_singleturn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the single-turn request with one available tool."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a")],
        available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
    )


def build_tools_multiturn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the multi-turn request with two available tools."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ],
        available_tools=[
            Tool(function=Function(name="tool1", description="1", parameters={})),
            Tool(function=Function(name="tool2", description="2", parameters={})),
        ],
    )


def build_system_tools_multiturn_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the multi-turn request with a system prompt and tool."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ],
        available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        system_prompt="SYSTEM",
    )


def build_empty_assistant_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request containing an empty assistant message."""
    return InstructRequest[ChatMessage, Tool](messages=[UserMessage(content="a"), AssistantMessage(content="")])


def build_system_message_request() -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request containing a system message."""
    return InstructRequest[ChatMessage, Tool](messages=[SystemMessage(content="s"), UserMessage(content="a")])

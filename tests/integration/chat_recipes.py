r"""Python-authored chat recipes for public chat workflow cases.

Each recipe constructs a fresh ``ChatCompletionRequest`` on every call, so
cases never share mutable request state. Recipes are reusable across cases
and never carry expectations; the case registry binds a recipe to one
tokenizer configuration and one expected outcome.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall


@dataclass(frozen=True)
class ChatRecipe:
    """One Python-authored input recipe identified by a stable recipe id."""

    recipe_id: str
    build: Callable[[], ChatCompletionRequest[ChatMessage]]


_CURRENT_WEATHER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to use. Infer this from the users location.",
        },
    },
    "required": ["location", "format"],
}

_N_DAY_WEATHER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to use. Infer this from the users location.",
        },
        "num_days": {
            "type": "integer",
            "description": "The number of days to forecast",
        },
    },
    "required": ["location", "format", "num_days"],
}

_SEND_EMAIL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "attachment": {
            "type": "string",
            "description": "The path or URL of an attachment (optional)",
        },
        "body": {
            "type": "string",
            "description": "The body/content of the email",
        },
        "subject": {
            "type": "string",
            "description": "The subject of the email",
        },
        "to": {
            "type": "string",
            "description": "The email address of the recipient",
        },
    },
    "required": ["to", "subject", "body"],
}


def _current_weather_tool() -> Tool:
    return Tool(
        function=Function(
            name="get_current_weather",
            description="Get the current weather",
            parameters=deepcopy(_CURRENT_WEATHER_PARAMETERS),
        )
    )


def _n_day_weather_tool() -> Tool:
    return Tool(
        function=Function(
            name="get_n_day_weather_forecast",
            description="Get an N-day weather forecast",
            parameters=deepcopy(_N_DAY_WEATHER_PARAMETERS),
        )
    )


def _send_email_tool() -> Tool:
    return Tool(
        function=Function(
            name="send_email",
            description="Send an email to a recipient",
            parameters=deepcopy(_SEND_EMAIL_PARAMETERS),
        )
    )


def _weather_tools() -> list[Tool]:
    return [_current_weather_tool(), _n_day_weather_tool()]


def _weather_system_messages() -> list[ChatMessage]:
    return [
        SystemMessage(content="Don't make assumptions about what values to plug into functions."),
        SystemMessage(content="Ask for clarification if a user request is ambiguous."),
    ]


def _build_no_tools() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        model="debug",
        messages=[
            UserMessage(content="What's the result of 5 + 5?"),
            AssistantMessage(content="The result of 5 + 5 is 10."),
            UserMessage(content="What is the square root of 64?"),
            AssistantMessage(content="The square root of 64 is 8, because 8 x 8 equals 64."),
            UserMessage(content="Can you multiply the results of the previous two questions?"),
            AssistantMessage(content="Sure! The result of 10 x 8 is 80."),
            UserMessage(content="Thanks"),
        ],
        tools=[],
    )


def _build_weather_full() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        model="debug",
        messages=[
            *_weather_system_messages(),
            UserMessage(content="What's the weather like today"),
            AssistantMessage(content="Sure, can you please provide me with your location?"),
            UserMessage(content="I'm in Paris, France."),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments='{"location":"Paris, France","format":"celsius"}',
                        ),
                    )
                ],
            ),
            ToolMessage(name="get_current_weather", content="22", tool_call_id="123456789"),
            AssistantMessage(content="The current temperature in Paris, France is 22 degrees Celsius."),
            UserMessage(content="what is the weather going to be like in Paris, France over the next x days"),
            AssistantMessage(content="Please tell me the number of days you want the weather forecast for."),
            UserMessage(content="5 days"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="get_n_day_weather_forecast",
                            arguments='{"location":"Paris, France","format":"celsius","num_days":5}',
                        ),
                    )
                ],
            ),
            ToolMessage(
                name="get_n_day_weather_forecast",
                content='{"2024-05-22":"22","2024-05-23":"23","2024-05-24":"24","2024-05-25":"25","2024-05-26":"26"}',
                tool_call_id="123456789",
            ),
        ],
        tools=_weather_tools(),
    )


def _build_weather_no_history() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        model="debug",
        messages=[
            *_weather_system_messages(),
            UserMessage(content="What's the weather like today"),
            AssistantMessage(content="Sure, can you please provide me with your location?"),
            UserMessage(content="I'm in Paris, France."),
            AssistantMessage(content="The current temperature in Paris, France is 22 degrees Celsius."),
            UserMessage(content="what is the weather going to be like in Paris, France over the next x days"),
            AssistantMessage(content="Please tell me the number of days you want the weather forecast for."),
            UserMessage(content="5 days"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="get_n_day_weather_forecast",
                            arguments='{\n  "location": "Paris, France",\n  "format": "celsius",\n  "num_days": 5\n}',
                        ),
                    )
                ],
            ),
            ToolMessage(
                name="get_n_day_weather_forecast",
                content=(
                    '{"2024-05-22": "22", "2024-05-23": "23", "2024-05-24": "24", '
                    '"2024-05-25": "25", "2024-05-26": "26"}'
                ),
                tool_call_id="123456789",
            ),
        ],
        tools=_weather_tools(),
    )


def _build_weather_no_system() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        model="debug",
        messages=[
            UserMessage(content="What's the weather like today"),
            AssistantMessage(content="Sure, can you please provide me with your location?"),
            UserMessage(content="I'm in Paris, France."),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments='{\n  "location": "Paris, France",\n  "format": "celsius"\n}',
                        ),
                    )
                ],
            ),
            ToolMessage(name="get_current_weather", content="22", tool_call_id="123456789"),
            AssistantMessage(content="The current temperature in Paris, France is 22 degrees Celsius."),
            UserMessage(content="what is the weather going to be like in Paris, France over the next x days"),
            AssistantMessage(content="Please tell me the number of days you want the weather forecast for."),
            UserMessage(content="5 days"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="get_n_day_weather_forecast",
                            arguments='{\n  "location": "Paris, France",\n  "format": "celsius",\n  "num_days": 5\n}',
                        ),
                    )
                ],
            ),
            ToolMessage(
                name="get_n_day_weather_forecast",
                content=(
                    '{"2024-05-22": "22", "2024-05-23": "23", "2024-05-24": "24", '
                    '"2024-05-25": "25", "2024-05-26": "26"}'
                ),
                tool_call_id="123456789",
            ),
        ],
        tools=_weather_tools(),
    )


def _build_several_calls() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        model="debug",
        messages=[
            UserMessage(
                content=(
                    "I need to send a report to my manager. The report is saved as a PDF at 'report.pdf'. "
                    'The email address is manager@company.com. The subject should be "Monthly Report" '
                    'and the body should say "Attached is the monthly report for your review."'
                )
            ),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="send_email",
                            arguments=(
                                '{"attachment": "report.pdf", '
                                '"body": "Attached is the monthly report for your review.", '
                                '"subject": "Monthly Report", "to": "manager@company.com"}'
                            ),
                        ),
                    )
                ],
            ),
            ToolMessage(name="send_email", content="Email sent to manager@company.com", tool_call_id="123456789"),
            AssistantMessage(content="I've sent the report to your manager. Anything else?"),
            UserMessage(
                content=(
                    "Great! Can you also send the same report to the finance team? "
                    "Their email addresses are finance1@company.com and finance2@company.com."
                )
            ),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="send_email",
                            arguments=(
                                '{"attachment": "report.pdf", '
                                '"body": "Attached is the monthly report for your review.", '
                                '"subject": "Monthly Report", "to": "finance1@company.com"}'
                            ),
                        ),
                    )
                ],
            ),
            ToolMessage(name="send_email", content="Email sent to finance1@company.com", tool_call_id="123456789"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="send_email",
                            arguments=(
                                '{"attachment": "report.pdf", '
                                '"body": "Attached is the monthly report for your review.", '
                                '"subject": "Monthly Report", "to": "finance2@company.com."}'
                            ),
                        ),
                    )
                ],
            ),
            ToolMessage(name="send_email", content="Email sent to finance2@company.com", tool_call_id="123456789"),
        ],
        tools=[_send_email_tool()],
    )


def _build_parallel_calls() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        model="debug",
        messages=[
            UserMessage(
                content=(
                    "I need to send a report to my manager. The report is saved as a PDF at 'report.pdf'. "
                    'The email address is manager@company.com. The subject should be "Monthly Report" '
                    'and the body should say "Attached is the monthly report for your review."'
                )
            ),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="send_email",
                            arguments=(
                                '{"attachment": "report.pdf", '
                                '"body": "Attached is the monthly report for your review.", '
                                '"subject": "Monthly Report", "to": "manager@company.com"}'
                            ),
                        ),
                    )
                ],
            ),
            ToolMessage(name="send_email", content="Email sent to manager@company.com", tool_call_id="123456789"),
            AssistantMessage(content="I've sent the report to your manager. Anything else?"),
            UserMessage(
                content=(
                    "Great! Can you also send the same report to the finance team? "
                    "Their email addresses are finance1@company.com and finance2@company.com."
                )
            ),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="send_email",
                            arguments=(
                                '{"attachment": "report.pdf", '
                                '"body": "Attached is the monthly report for your review.", '
                                '"subject": "Monthly Report", "to": "finance1@company.com"}'
                            ),
                        ),
                    ),
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(
                            name="send_email",
                            arguments=(
                                '{"attachment": "report.pdf", '
                                '"body": "Attached is the monthly report for your review.", '
                                '"subject": "Monthly Report", "to": "finance2@company.com."}'
                            ),
                        ),
                    ),
                ],
            ),
            ToolMessage(name="send_email", content="Email sent to finance1@company.com", tool_call_id="123456789"),
            ToolMessage(name="send_email", content="Email sent to finance2@company.com", tool_call_id="123456789"),
        ],
        tools=[_send_email_tool()],
    )


def _build_tool_results_request(
    *, call_count: int, result_count: int, terminal_assistant: bool
) -> ChatCompletionRequest[ChatMessage]:
    tool_calls = [
        ToolCall(id=f"call0000{index}", function=FunctionCall(name=f"tool_{index}", arguments="{}"))
        for index in range(1, call_count + 1)
    ]
    messages: list[ChatMessage] = [
        UserMessage(content="Run these tools."),
        AssistantMessage(content=None, tool_calls=tool_calls),
    ]
    messages.extend(
        ToolMessage(name=f"tool_{index}", content=f"result {index}", tool_call_id=f"call0000{index}")
        for index in range(1, result_count + 1)
    )
    if terminal_assistant:
        messages.append(AssistantMessage(content="The tool work is complete."))

    return ChatCompletionRequest[ChatMessage](model="test", messages=messages)


def _build_parallel_tool_results() -> ChatCompletionRequest[ChatMessage]:
    return _build_tool_results_request(call_count=2, result_count=2, terminal_assistant=False)


def _build_parallel_tool_results_finetuning() -> ChatCompletionRequest[ChatMessage]:
    return _build_tool_results_request(call_count=2, result_count=2, terminal_assistant=True)


def _build_mismatched_tool_results() -> ChatCompletionRequest[ChatMessage]:
    return _build_tool_results_request(call_count=1, result_count=2, terminal_assistant=False)


def _build_mismatched_tool_results_finetuning() -> ChatCompletionRequest[ChatMessage]:
    return _build_tool_results_request(call_count=1, result_count=2, terminal_assistant=True)


NO_TOOLS = ChatRecipe(recipe_id="sample-no-tools", build=_build_no_tools)
WEATHER_FULL = ChatRecipe(recipe_id="sample-weather-full", build=_build_weather_full)
WEATHER_NO_HISTORY = ChatRecipe(recipe_id="sample-weather-no-history", build=_build_weather_no_history)
WEATHER_NO_SYSTEM = ChatRecipe(recipe_id="sample-weather-no-system", build=_build_weather_no_system)
SEVERAL_CALLS = ChatRecipe(recipe_id="sample-several-calls", build=_build_several_calls)
PARALLEL_CALLS = ChatRecipe(recipe_id="sample-parallel-calls", build=_build_parallel_calls)
PARALLEL_TOOL_RESULTS = ChatRecipe(recipe_id="parallel-tool-results", build=_build_parallel_tool_results)
PARALLEL_TOOL_RESULTS_FINETUNING = ChatRecipe(
    recipe_id="parallel-tool-results-finetuning", build=_build_parallel_tool_results_finetuning
)
MISMATCHED_TOOL_RESULTS = ChatRecipe(recipe_id="mismatched-tool-results", build=_build_mismatched_tool_results)
MISMATCHED_TOOL_RESULTS_FINETUNING = ChatRecipe(
    recipe_id="mismatched-tool-results-finetuning", build=_build_mismatched_tool_results_finetuning
)

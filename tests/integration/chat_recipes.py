r"""Python-authored chat recipes for public chat workflow cases.

Each recipe constructs a fresh ``ChatCompletionRequest`` on every call, so
cases never share mutable request state. Recipes are reusable across cases
and never carry expectations; the case registry binds a recipe to one
tokenizer configuration and one expected outcome.
"""

import base64
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image

from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURLChunk,
    ContentChunk,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort, ResponseFormat
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from tests.fixtures.audio import get_dummy_audio_chunk


@dataclass(frozen=True)
class ChatRecipe:
    """One Python-authored input recipe identified by a stable recipe id."""

    recipe_id: str
    build: Callable[[], ChatCompletionRequest[ChatMessage]]


def build_red_image() -> Image.Image:
    r"""Create a fresh 4x4 RGB red image for image-bearing recipes."""
    return Image.new(mode="RGB", size=(4, 4), color="red")


def build_image_url_chunk(*, image: Image.Image) -> ImageURLChunk:
    r"""Encode an image as a PNG data URL chunk.

    Args:
        image: Image to encode.

    Returns:
        The image chunk containing the encoded PNG data URL.
    """
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG")
    image_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode(encoding='ascii')}"
    return ImageURLChunk(image_url=image_url)


def build_math_interpreter_tool() -> Tool:
    r"""Create the shared arithmetic-expression tool with fresh schema data."""
    return Tool(
        function=Function(
            name="math_interpreter",
            description="Get the value of an arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression.",
                    }
                },
            },
        )
    )


def build_function_tool(*, name: str, description: str | None, parameters: dict[str, Any]) -> Tool:
    r"""Create a tool with independent schema data for one request.

    Args:
        name: Function name carried by the tool.
        description: Function description, or `None` to use the model default.
        parameters: JSON schema copied into the fresh tool.

    Returns:
        A tool with a deep copy of the supplied parameter schema.
    """
    function_values: dict[str, Any] = {"name": name, "parameters": deepcopy(parameters)}
    if description is not None:
        function_values["description"] = description
    return Tool(function=Function(**function_values))


def build_user_content_request(*, content: list[ContentChunk]) -> ChatCompletionRequest[ChatMessage]:
    r"""Wrap fresh content in a single user message.

    Args:
        content: Ordered chunks belonging to the user message.

    Returns:
        A request containing the user message.
    """
    return ChatCompletionRequest[ChatMessage](messages=[UserMessage(content=content)])


def build_two_tool_call_messages(*, tool_results: tuple[tuple[str, str], tuple[str, str]]) -> list[ChatMessage]:
    r"""Build the shared two-call scaffold, preserving the supplied result order.

    Args:
        tool_results: Ordered pairs of result content and matching tool-call id.

    Returns:
        Fresh system, user, tool-call, tool-result, assistant, and user messages.
    """
    return [
        SystemMessage(content="S"),
        UserMessage(content="U1"),
        AssistantMessage(
            content="A1",
            tool_calls=[
                ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{}")),
                ToolCall(id="999999999", function=FunctionCall(name="F2", arguments="{}")),
            ],
        ),
        *[ToolMessage(content=content, tool_call_id=tool_call_id) for content, tool_call_id in tool_results],
        AssistantMessage(content="A2"),
        UserMessage(content="U2"),
    ]


def build_tool_call_messages(*, call_ids: Sequence[str | None], result_ids: Sequence[str] = ()) -> list[ChatMessage]:
    r"""Build a user message, an assistant message with one tool call per id, and one tool result per result id.

    Args:
        call_ids: Tool call ids for the assistant message's calls; `None` leaves the
            id unset (the model default `"null"`).
        result_ids: Tool call ids answered by tool result messages, in the given order.

    Returns:
        Fresh user, assistant and tool messages.
    """
    return [
        UserMessage(content="U"),
        AssistantMessage(
            tool_calls=[
                ToolCall(function=FunctionCall(name="F", arguments="{}"))
                if call_id is None
                else ToolCall(id=call_id, function=FunctionCall(name="F", arguments="{}"))
                for call_id in call_ids
            ]
        ),
        *[ToolMessage(content="R", tool_call_id=result_id) for result_id in result_ids],
    ]


def build_tool_call(*, tool_call_id: str, name: str, arguments: dict[str, Any] | str) -> ToolCall:
    r"""Create a tool call with the given id, function name and arguments.

    Args:
        tool_call_id: Unique identifier paired with the tool result message.
        name: Function name carried by the call.
        arguments: Call arguments as a JSON string or a dict parsed on construction.

    Returns:
        A fresh tool call.
    """
    return ToolCall(id=tool_call_id, function=FunctionCall(name=name, arguments=arguments))  # type: ignore[arg-type]


def build_tool_message(*, content: str | list[ContentChunk], tool_call_id: str, name: str | None = None) -> ToolMessage:
    r"""Create a tool result message responding to the given tool call.

    Args:
        content: Tool output, plain text or multimodal chunks.
        tool_call_id: Identifier of the assistant tool call this answers.
        name: Optional tool name; omitted when the renderer injects it.

    Returns:
        A fresh tool message.
    """
    if name is None:
        return ToolMessage(content=content, tool_call_id=tool_call_id)
    return ToolMessage(name=name, content=content, tool_call_id=tool_call_id)


def _call_id_recipe(*, recipe_id: str, tool_call_id: str) -> ChatRecipe:
    r"""Bind a call id to a recipe that creates a fresh request each time.

    Args:
        recipe_id: Stable identity for the recipe.
        tool_call_id: Identifier paired between the call and result.

    Returns:
        A recipe that builds a fresh request each time.
    """

    def build() -> ChatCompletionRequest[ChatMessage]:
        return ChatCompletionRequest[ChatMessage](
            messages=[
                UserMessage(content="a"),
                AssistantMessage(
                    tool_calls=[ToolCall(id=tool_call_id, function=FunctionCall(name="f", arguments="{}"))]
                ),
                ToolMessage(content="b", tool_call_id=tool_call_id),
            ]
        )

    return ChatRecipe(recipe_id=recipe_id, build=build)


def _build_prefixed_final_request() -> ChatCompletionRequest[ChatMessage]:
    r"""Build the shared user message followed by a prefixed assistant turn."""
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)]
    )


def _build_system_think() -> ChatCompletionRequest[ChatMessage]:
    r"""Build the shared system-think request."""
    return ChatCompletionRequest[ChatMessage](
        messages=[SystemMessage(content=[ThinkChunk(thinking="Hi")]), UserMessage(content="Hello")]
    )


def build_user_media_request(
    *, prompt: str, content_chunk: AudioChunk | AudioURLChunk | ImageURLChunk
) -> ChatCompletionRequest[ChatMessage]:
    r"""Build a user prompt followed by the supplied media chunk.

    Args:
        prompt: Text preceding the media chunk.
        content_chunk: Fresh audio or image-URL chunk for this request.

    Returns:
        A fresh request containing the prompt and media chunk.
    """
    return ChatCompletionRequest[ChatMessage](messages=[UserMessage(content=[TextChunk(text=prompt), content_chunk])])


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
    return build_function_tool(
        name="get_current_weather",
        description="Get the current weather",
        parameters=_CURRENT_WEATHER_PARAMETERS,
    )


def _n_day_weather_tool() -> Tool:
    return build_function_tool(
        name="get_n_day_weather_forecast",
        description="Get an N-day weather forecast",
        parameters=_N_DAY_WEATHER_PARAMETERS,
    )


def _send_email_tool() -> Tool:
    return build_function_tool(
        name="send_email",
        description="Send an email to a recipient",
        parameters=_SEND_EMAIL_PARAMETERS,
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


def build_tool_results_request(
    *, call_count: int, terminal_assistant: bool, tools: list[Tool] | None = None
) -> ChatCompletionRequest[ChatMessage]:
    r"""Build one assistant turn with `call_count` tool calls followed by two tool results.

    Args:
        call_count: Number of tool calls on the assistant turn.
        terminal_assistant: Whether to append a final assistant message.
        tools: Optional tool declarations attached to the request.

    Returns:
        A fresh request with the parallel tool-call scaffold.
    """
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
        for index in (1, 2)
    )
    if terminal_assistant:
        messages.append(AssistantMessage(content="The tool work is complete."))

    return build_conversation_request(messages=messages, model="test", tools=tools)


def _build_parallel_tool_results() -> ChatCompletionRequest[ChatMessage]:
    return build_tool_results_request(call_count=2, terminal_assistant=False)


def _build_mismatched_tool_results() -> ChatCompletionRequest[ChatMessage]:
    return build_tool_results_request(call_count=1, terminal_assistant=False)


def _build_mismatched_tool_results_finetuning() -> ChatCompletionRequest[ChatMessage]:
    return build_tool_results_request(call_count=1, terminal_assistant=True)


NO_TOOLS = ChatRecipe(recipe_id="sample-no-tools", build=_build_no_tools)
WEATHER_FULL = ChatRecipe(recipe_id="sample-weather-full", build=_build_weather_full)
WEATHER_NO_HISTORY = ChatRecipe(recipe_id="sample-weather-no-history", build=_build_weather_no_history)
WEATHER_NO_SYSTEM = ChatRecipe(recipe_id="sample-weather-no-system", build=_build_weather_no_system)
SEVERAL_CALLS = ChatRecipe(recipe_id="sample-several-calls", build=_build_several_calls)
PARALLEL_CALLS = ChatRecipe(recipe_id="sample-parallel-calls", build=_build_parallel_calls)
PARALLEL_TOOL_RESULTS = ChatRecipe(recipe_id="parallel-tool-results", build=_build_parallel_tool_results)
MISMATCHED_TOOL_RESULTS = ChatRecipe(recipe_id="mismatched-tool-results", build=_build_mismatched_tool_results)
MISMATCHED_TOOL_RESULTS_FINETUNING = ChatRecipe(
    recipe_id="mismatched-tool-results-finetuning", build=_build_mismatched_tool_results_finetuning
)
PREFIXED_FINAL = ChatRecipe(recipe_id="prefixed-final", build=_build_prefixed_final_request)
SYSTEM_THINK = ChatRecipe(recipe_id="released-system-think", build=_build_system_think)


def _build_system_then_audio_user() -> ChatCompletionRequest[ChatMessage]:
    r"""Build a system message followed by a user message carrying an audio chunk."""
    return ChatCompletionRequest[ChatMessage](
        messages=[
            SystemMessage(content="S"),
            UserMessage(content=[TextChunk(text="Transcribe this audio."), get_dummy_audio_chunk()]),
        ]
    )


SYSTEM_THEN_AUDIO_USER = ChatRecipe(recipe_id="system-then-audio-user", build=_build_system_then_audio_user)


def _build_mid_conversation_system() -> ChatCompletionRequest[ChatMessage]:
    r"""Build a conversation with a system message after the first user message."""
    return ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(content="first"),
            SystemMessage(content="mid-conversation system"),
            UserMessage(content="second"),
        ]
    )


def _build_empty_system() -> ChatCompletionRequest[ChatMessage]:
    r"""Build a request whose system message has `None` content.

    `SystemMessage.model_construct` bypasses request-model validation so the
    empty system reaches the validator's own check, which is the
    project-owned rejection under test.
    """
    empty_system = SystemMessage.model_construct(role="system", content=None)
    return ChatCompletionRequest[ChatMessage](messages=[empty_system, UserMessage(content="hi")])


def _build_reasoning_effort_unsupported() -> ChatCompletionRequest[ChatMessage]:
    r"""Build a plain request carrying a reasoning effort, for pre-v15 profiles."""
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content="hi")],
        reasoning_effort=ReasoningEffort.high,
    )


MID_CONVERSATION_SYSTEM = ChatRecipe(recipe_id="mid-conversation-system", build=_build_mid_conversation_system)
EMPTY_SYSTEM = ChatRecipe(recipe_id="empty-system", build=_build_empty_system)
REASONING_EFFORT_UNSUPPORTED = ChatRecipe(
    recipe_id="reasoning-effort-unsupported", build=_build_reasoning_effort_unsupported
)
CALL_ID_X = _call_id_recipe(recipe_id="call-id-x", tool_call_id="x")
CALL_ID_SLASH = _call_id_recipe(recipe_id="call-id-slash", tool_call_id="call/id-1")


def build_conversation_request(
    *,
    messages: list[ChatMessage],
    model: str | None = None,
    tools: list[Tool] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    response_format: ResponseFormat | None = None,
) -> ChatCompletionRequest[ChatMessage]:
    r"""Assemble a fresh request from ready-made messages and optional settings.

    Args:
        messages: Ordered conversation messages.
        model: Optional model name carried by the request.
        tools: Optional tool declarations attached to the request.
        reasoning_effort: Optional reasoning effort setting.
        response_format: Optional response format setting.

    Returns:
        A fresh request carrying the supplied messages and settings.
    """
    request_kwargs: dict[str, Any] = {"messages": messages}
    if model is not None:
        request_kwargs["model"] = model
    if tools is not None:
        request_kwargs["tools"] = tools
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        request_kwargs["response_format"] = response_format
    return ChatCompletionRequest[ChatMessage](**request_kwargs)


def build_user_assistant_request(
    *,
    user_content: str | list[ContentChunk],
    assistant_content: str | list[ContentChunk] | None,
    tools: list[Tool] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    response_format: ResponseFormat | None = None,
) -> ChatCompletionRequest[ChatMessage]:
    r"""Build a two-turn user/assistant conversation.

    Args:
        user_content: User message content (text or ordered chunks).
        assistant_content: Assistant message content.
        tools: Optional tool declarations attached to the request.
        reasoning_effort: Optional reasoning effort setting.
        response_format: Optional response format setting.

    Returns:
        A fresh request with the user and assistant turns.
    """
    return build_conversation_request(
        messages=[UserMessage(content=user_content), AssistantMessage(content=assistant_content)],
        tools=tools,
        reasoning_effort=reasoning_effort,
        response_format=response_format,
    )


def build_tool_call_request(
    *,
    tool_name: str,
    call_id: str,
    arguments: str,
    result_content: str | list[ContentChunk],
    result_name: str | None = None,
    assistant_content: str | None = None,
    user_content: str = "Hello",
    terminal_content: str | None = None,
    follow_up: tuple[str, str] | None = None,
    tools: list[Tool] | None = None,
) -> ChatCompletionRequest[ChatMessage]:
    r"""Build a single tool-call turn: user, assistant call, tool result, optional follow-ups.

    Args:
        tool_name: Function name of the tool call.
        call_id: Tool-call id shared by the call and the result.
        arguments: Raw JSON arguments string of the call.
        result_content: Tool result content (text or ordered chunks).
        result_name: Tool name on the result message; `None` omits it (v2 requires it).
        assistant_content: Text content of the assistant turn carrying the call.
        user_content: User message content.
        terminal_content: Optional final assistant message after the result.
        follow_up: Optional `(user_content, assistant_content)` pair after the result.
        tools: Optional tool declarations attached to the request.

    Returns:
        A fresh request with the tool-call conversation.
    """
    messages: list[ChatMessage] = [
        UserMessage(content=user_content),
        AssistantMessage(
            content=assistant_content,
            tool_calls=[ToolCall(id=call_id, function=FunctionCall(name=tool_name, arguments=arguments))],
        ),
        ToolMessage(content=result_content, tool_call_id=call_id, name=result_name),
    ]
    if terminal_content is not None:
        messages.append(AssistantMessage(content=terminal_content))
    if follow_up is not None:
        messages.append(UserMessage(content=follow_up[0]))
        messages.append(AssistantMessage(content=follow_up[1]))
    return build_conversation_request(messages=messages, tools=tools)


def build_aggregated_tool_calls_request(*, tools: list[Tool] | None = None) -> ChatCompletionRequest[ChatMessage]:
    r"""Build two consecutive assistant tool-call turns whose results are aggregated.

    Args:
        tools: Optional tool declarations attached to the request.

    Returns:
        A fresh request with both calls, both results, and a closing exchange.
    """
    return build_conversation_request(
        messages=[
            UserMessage(content="Weather?"),
            AssistantMessage(
                content="Checking Paris.",
                tool_calls=[
                    ToolCall(id="123456789", function=FunctionCall(name="get_weather", arguments='{"city": "Paris"}'))
                ],
            ),
            AssistantMessage(
                content="And London.",
                tool_calls=[
                    ToolCall(id="023456789", function=FunctionCall(name="get_weather", arguments='{"city": "London"}'))
                ],
            ),
            ToolMessage(name="get_weather", content="22", tool_call_id="123456789"),
            ToolMessage(name="get_weather", content="15", tool_call_id="023456789"),
            AssistantMessage(content="Paris: 22, London: 15"),
            UserMessage(content="Thanks"),
            AssistantMessage(content="Welcome"),
        ],
        tools=tools,
    )


def build_thinking_assistant_request(
    *,
    user_content: str = "Hi",
    thinking: str | None = None,
    text: str | None = None,
    closed: bool = True,
    tool_calls: list[ToolCall] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> ChatCompletionRequest[ChatMessage]:
    r"""Build a user/assistant turn where the assistant optionally carries thinking.

    Args:
        user_content: User message content.
        thinking: Thinking text; `None` omits the ThinkChunk.
        text: Assistant text content; `None` omits the TextChunk.
        closed: Whether the ThinkChunk is closed.
        tool_calls: Optional tool calls on the assistant turn.
        reasoning_effort: Optional reasoning effort setting.

    Returns:
        A fresh request with the assistant turn built from the given parts.
    """
    chunks: list[ContentChunk] = []
    if thinking is not None:
        chunks.append(ThinkChunk(thinking=thinking, closed=closed))
    if text is not None:
        chunks.append(TextChunk(text=text))
    content: str | list[ContentChunk] | None
    if not chunks:
        content = None
    elif thinking is None:
        content = text
    else:
        content = chunks
    return build_conversation_request(
        messages=[UserMessage(content=user_content), AssistantMessage(content=content, tool_calls=tool_calls)],
        reasoning_effort=reasoning_effort,
    )


def build_system_content_request(
    *,
    system_content: str | list[ContentChunk],
    user_content: str = "Hello",
    assistant_content: str = "Hi",
    tools: list[Tool] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> ChatCompletionRequest[ChatMessage]:
    r"""Build a conversation starting with a system message of the given content.

    Args:
        system_content: System message content (text or ordered chunks).
        user_content: User message content.
        assistant_content: Assistant message content.
        tools: Optional tool declarations attached to the request.
        reasoning_effort: Optional reasoning effort setting.

    Returns:
        A fresh request with system, user, and assistant turns.
    """
    return build_conversation_request(
        messages=[
            SystemMessage(content=system_content),
            UserMessage(content=user_content),
            AssistantMessage(content=assistant_content),
        ],
        tools=tools,
        reasoning_effort=reasoning_effort,
    )

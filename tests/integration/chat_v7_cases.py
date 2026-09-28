r"""Public v7 chat recipes and success cases from the legacy v7 selectors."""

from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from tests.integration.chat_cases import PublicChatSuccessCase
from tests.integration.chat_recipes import ChatRecipe
from tests.integration.tokenizer_configurations import (
    BUNDLED_SPM_V7_MM_TEST,
    PINNED_V7_AUDIO_FINETUNING,
    PINNED_V7_AUDIO_SERVING,
    PINNED_V7_IMAGE_FINETUNING,
    PINNED_V7_IMAGE_SERVING,
)


@dataclass(frozen=True)
class V7DirectEqualityCase:
    """Direct instruct recipe paired with its public success case ID."""

    case_id: str
    build_direct_request: Callable[[], InstructRequest]


def _red_4x4() -> Image.Image:
    return Image.new(mode="RGB", size=(4, 4), color="red")


def _build_system_tools_image() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        tools=[
            Tool(
                function=Function(
                    name="t",
                    parameters={
                        "type": "object",
                        "properties": {"g": {"type": "string"}, "h": {"type": "string"}},
                    },
                )
            )
        ],
        messages=[
            SystemMessage(content="a"),
            UserMessage(content=[TextChunk(text="a"), ImageChunk(image=_red_4x4())]),
            AssistantMessage(content="b"),
            ToolMessage(tool_call_id="123456789", content="f"),
        ],
    )


def _available_tools() -> list[Tool]:
    return [
        Tool(function=Function(name="t1", parameters={})),
        Tool(function=Function(name="t2", parameters={})),
    ]


def _tool_content_messages(*, include_results: bool) -> list[ChatMessage]:
    messages: list[ChatMessage] = [
        UserMessage(content="a"),
        AssistantMessage(
            content="b1b2",
            tool_calls=[
                ToolCall(id="000000000", function=FunctionCall(name="t1", arguments="{}")),
                ToolCall(id="111111111", function=FunctionCall(name="t2", arguments="{}")),
            ],
        ),
    ]
    if include_results:
        messages.extend(
            [
                ToolMessage(content="r1", tool_call_id="000000000"),
                ToolMessage(content="r2", tool_call_id="111111111"),
            ]
        )
    return messages


def _build_tool_content_instruct_request() -> InstructRequest:
    return InstructRequest(messages=_tool_content_messages(include_results=False), available_tools=_available_tools())


def _build_tool_results_instruct_request() -> InstructRequest:
    return InstructRequest(messages=_tool_content_messages(include_results=True), available_tools=_available_tools())


def _build_chat_request_from_instruct(
    instruct_request: InstructRequest, *, model: str | None = None
) -> ChatCompletionRequest[ChatMessage]:
    tools = instruct_request.available_tools
    exclude = {"system_prompt", "truncate_at_max_tokens", "available_tools", "settings"}
    return ChatCompletionRequest[ChatMessage](**instruct_request.model_dump(exclude=exclude), model=model, tools=tools)


def _build_tool_content() -> ChatCompletionRequest[ChatMessage]:
    return _build_chat_request_from_instruct(_build_tool_content_instruct_request())


def _build_tool_results() -> ChatCompletionRequest[ChatMessage]:
    return _build_chat_request_from_instruct(_build_tool_results_instruct_request(), model="test-model")


def _build_prefixed_final() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)],
    )


_SYSTEM_TOOLS_IMAGE = ChatRecipe(recipe_id="v7-system-tools-image", build=_build_system_tools_image)
_TOOL_CONTENT = ChatRecipe(recipe_id="v7-tool-content", build=_build_tool_content)
_TOOL_RESULTS = ChatRecipe(recipe_id="v7-tool-results", build=_build_tool_results)
_PREFIXED_FINAL = ChatRecipe(recipe_id="v7-prefixed-final", build=_build_prefixed_final)

V7_SUCCESS_CASES: tuple[PublicChatSuccessCase, ...] = (
    PublicChatSuccessCase(
        case_id="chat-v7-system-tools-image",
        recipe=_SYSTEM_TOOLS_IMAGE,
        configuration=BUNDLED_SPM_V7_MM_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v7-tool-content-no-audio-ft",
        recipe=_TOOL_CONTENT,
        configuration=PINNED_V7_IMAGE_FINETUNING,
    ),
    PublicChatSuccessCase(
        case_id="chat-v7-tool-content-audio-ft",
        recipe=_TOOL_CONTENT,
        configuration=PINNED_V7_AUDIO_FINETUNING,
    ),
    PublicChatSuccessCase(
        case_id="chat-v7-tool-results-no-audio-serving",
        recipe=_TOOL_RESULTS,
        configuration=PINNED_V7_IMAGE_SERVING,
    ),
    PublicChatSuccessCase(
        case_id="chat-v7-tool-results-audio-serving",
        recipe=_TOOL_RESULTS,
        configuration=PINNED_V7_AUDIO_SERVING,
    ),
    PublicChatSuccessCase(
        case_id="chat-v7-prefixed-final",
        recipe=_PREFIXED_FINAL,
        configuration=BUNDLED_SPM_V7_MM_TEST,
    ),
)

V7_DIRECT_EQUALITY_CASES: tuple[V7DirectEqualityCase, ...] = (
    V7DirectEqualityCase(
        case_id="chat-v7-tool-content-no-audio-ft",
        build_direct_request=_build_tool_content_instruct_request,
    ),
    V7DirectEqualityCase(
        case_id="chat-v7-tool-content-audio-ft",
        build_direct_request=_build_tool_content_instruct_request,
    ),
    V7DirectEqualityCase(
        case_id="chat-v7-tool-results-no-audio-serving",
        build_direct_request=_build_tool_results_instruct_request,
    ),
    V7DirectEqualityCase(
        case_id="chat-v7-tool-results-audio-serving",
        build_direct_request=_build_tool_results_instruct_request,
    ),
)

r"""Public v13 chat recipes and cases from the legacy v13 selectors."""

from collections.abc import Callable

from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURLChunk, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from tests.fixtures.audio import get_dummy_audio_chunk, get_dummy_audio_url_chunk
from tests.integration.chat_cases import PublicChatSuccessCase
from tests.integration.chat_recipes import (
    CALL_ID_SLASH,
    CALL_ID_X,
    PREFIXED_FINAL,
    ChatRecipe,
    build_math_interpreter_tool,
    build_two_tool_call_messages,
)
from tests.integration.tokenizer_configurations import PINNED_V13_TEXT_TEST, SYNTHETIC_V13_AUDIO_TEST


def _build_think_order() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        tools=[build_math_interpreter_tool()],
        messages=[
            SystemMessage(content=[TextChunk(text="S1"), ThinkChunk(thinking="TS"), TextChunk(text="S2")]),
            UserMessage(content="U1"),
            AssistantMessage(
                content="A1",
                tool_calls=[
                    ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{}")),
                    ToolCall(id="999999999", function=FunctionCall(name="F2", arguments="{}")),
                ],
            ),
            ToolMessage(content="R1", tool_call_id="123456789"),
            ToolMessage(content="R2", tool_call_id="999999999"),
            AssistantMessage(content=[ThinkChunk(thinking="T1"), TextChunk(text="A2")]),
            UserMessage(content="U2"),
        ],
    )


def _build_reversed_results() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        tools=[build_math_interpreter_tool()],
        messages=build_two_tool_call_messages(tool_results=(("R2", "999999999"), ("R1", "123456789"))),
    )


def _build_system_audio_request(
    *, audio_chunk_factory: Callable[[], AudioChunk | AudioURLChunk]
) -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[SystemMessage(content="hello"), UserMessage(content=[audio_chunk_factory()])]
    )


def _build_system_audio_chunk() -> ChatCompletionRequest[ChatMessage]:
    return _build_system_audio_request(audio_chunk_factory=get_dummy_audio_chunk)


def _build_system_audio_url() -> ChatCompletionRequest[ChatMessage]:
    return _build_system_audio_request(audio_chunk_factory=get_dummy_audio_url_chunk)


_THINK_ORDER = ChatRecipe(recipe_id="v13-think-order", build=_build_think_order)
_REVERSED_RESULTS = ChatRecipe(recipe_id="v13-reversed-results", build=_build_reversed_results)
_SYSTEM_AUDIO_CHUNK = ChatRecipe(recipe_id="v13-system-audio-chunk", build=_build_system_audio_chunk)
_SYSTEM_AUDIO_URL = ChatRecipe(recipe_id="v13-system-audio-url", build=_build_system_audio_url)

V13_SUCCESS_CASES: tuple[PublicChatSuccessCase, ...] = (
    PublicChatSuccessCase(
        case_id="chat-v13-think-order",
        recipe=_THINK_ORDER,
        configuration=PINNED_V13_TEXT_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v13-reversed-results",
        recipe=_REVERSED_RESULTS,
        configuration=PINNED_V13_TEXT_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v13-call-id-x",
        recipe=CALL_ID_X,
        configuration=PINNED_V13_TEXT_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v13-call-id-slash",
        recipe=CALL_ID_SLASH,
        configuration=PINNED_V13_TEXT_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v13-prefixed-final",
        recipe=PREFIXED_FINAL,
        configuration=PINNED_V13_TEXT_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v13-system-audio-chunk",
        recipe=_SYSTEM_AUDIO_CHUNK,
        configuration=SYNTHETIC_V13_AUDIO_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v13-system-audio-url",
        recipe=_SYSTEM_AUDIO_URL,
        configuration=SYNTHETIC_V13_AUDIO_TEST,
    ),
)

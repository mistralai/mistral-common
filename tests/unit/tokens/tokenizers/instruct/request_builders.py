r"""Shared instruct-request builders for versioned tokenizer tests.

These builders construct requests whose shape is identical across tokenizer versions.
Expected token IDs and decoded text stay in each version's test module.
"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURLChunk,
    ContentChunk,
    ImageChunk,
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
from mistral_common.protocol.instruct.request import InstructRequest, ModelSettings
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.tokens.tokenizers.audio import Audio


def build_normal_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the multi-turn request without a system prompt."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_system_singleturn_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the single-turn request with a system prompt."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a")], system_prompt="SYSTEM", settings=settings or ModelSettings.none()
    )


def build_system_multiturn_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the multi-turn request with a system prompt."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ],
        system_prompt="SYSTEM",
        settings=settings or ModelSettings.none(),
    )


def build_prefixed_final_message_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request with a prefixed final assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d", prefix=True),
        ],
        system_prompt="SYSTEM",
        settings=settings or ModelSettings.none(),
    )


def build_tools_singleturn_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the single-turn request with one available tool."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a")],
        available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        settings=settings or ModelSettings.none(),
    )


def build_tools_multiturn_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
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
        settings=settings or ModelSettings.none(),
    )


def build_system_tools_multiturn_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
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
        settings=settings or ModelSettings.none(),
    )


def build_empty_assistant_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request containing an empty assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), AssistantMessage(content="")], settings=settings or ModelSettings.none()
    )


def build_system_message_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request containing a system message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content="s"), UserMessage(content="a")], settings=settings or ModelSettings.none()
    )


def build_system_message_multiturn_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the multi-turn request with a system message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            SystemMessage(content="SYSTEM"),
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_system_message_prefixed_final_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request with a system message and a prefixed final assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            SystemMessage(content="SYSTEM"),
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d", prefix=True),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_system_message_tools_multiturn_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the multi-turn request with a system message and one available tool."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            SystemMessage(content="SYSTEM"),
            UserMessage(content="a"),
            AssistantMessage(content="b"),
            UserMessage(content="c"),
            AssistantMessage(content="d"),
        ],
        available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        settings=settings or ModelSettings.none(),
    )


def build_tool_message_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request containing a tool result without a preceding assistant turn."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), ToolMessage(content="d", tool_call_id="123456789")],
        settings=settings or ModelSettings.none(),
    )


def build_tool_call_request(
    tool_call: ToolCall, *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by an assistant turn with the supplied tool call."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), AssistantMessage(tool_calls=[tool_call])],
        settings=settings or ModelSettings.none(),
    )


def build_assistant_tool_calls_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by an assistant turn with one identified tool call."""
    return build_tool_call_request(
        tool_call=ToolCall(id="123456789", function=FunctionCall(name="b", arguments="{}")), settings=settings
    )


def build_user_then_assistant_request(
    assistant: AssistantMessage, *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by the supplied assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), assistant], settings=settings or ModelSettings.none()
    )


def build_red_image_chunk() -> ImageChunk:
    r"""Build an image chunk holding a red 4x4 image."""
    return ImageChunk(image=Image.new(mode="RGB", size=(4, 4), color="red"))


def build_blue_image_chunk() -> ImageChunk:
    r"""Build an image chunk holding a blue 4x4 image."""
    return ImageChunk(image=Image.new(mode="RGB", size=(4, 4), color="blue"))


def build_silent_audio_chunk(*, num_samples: int = 2400, sampling_rate: int = 24_000) -> AudioChunk:
    r"""Build an audio chunk holding silence."""
    audio = Audio(audio_array=np.zeros(shape=num_samples, dtype=np.float32), sampling_rate=sampling_rate, format="wav")
    return AudioChunk.from_audio(audio=audio)


def build_image_user_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single user message holds a red 4x4 image followed by text."""
    content: list[ContentChunk] = [build_red_image_chunk(), TextChunk(text="a")]
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=content)], settings=settings or ModelSettings.none()
    )


def build_text_then_image_user_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single user message holds text followed by a red 4x4 image."""
    content: list[ContentChunk] = [TextChunk(text="a"), build_red_image_chunk()]
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=content)], settings=settings or ModelSettings.none()
    )


def build_audio_user_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single user message holds a silent audio chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[build_silent_audio_chunk()])], settings=settings or ModelSettings.none()
    )


def build_text_then_audio_user_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single user message holds text followed by a silent audio chunk."""
    content: list[ContentChunk] = [TextChunk(text="a"), build_silent_audio_chunk()]
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=content)], settings=settings or ModelSettings.none()
    )


def build_multi_image_user_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single user message holds two sequential images (red then blue)."""
    content: list[ContentChunk] = [build_red_image_chunk(), build_blue_image_chunk()]
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=content)], settings=settings or ModelSettings.none()
    )


def build_multi_audio_user_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single user message holds two sequential silent audio chunks."""
    content: list[ContentChunk] = [build_silent_audio_chunk(), build_silent_audio_chunk()]
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=content)], settings=settings or ModelSettings.none()
    )


def build_multiturn_image_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a multi-turn request where each user message holds a red 4x4 image."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content=[build_red_image_chunk()]),
            AssistantMessage(content="b"),
            UserMessage(content=[build_red_image_chunk()]),
            AssistantMessage(content="d"),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_multiturn_audio_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a multi-turn request where each user message holds a silent audio chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content=[build_silent_audio_chunk()]),
            AssistantMessage(content="b"),
            UserMessage(content=[build_silent_audio_chunk()]),
            AssistantMessage(content="d"),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_assistant_list_content_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose assistant message holds a single text chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), AssistantMessage(content=[TextChunk(text="b")])],
        settings=settings or ModelSettings.none(),
    )


def build_system_and_user_words_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with multi-word system and user message content."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content="hello world"), UserMessage(content="good day")],
        settings=settings or ModelSettings.none(),
    )


def build_truncation_last_user_at_limit_request(
    *, settings: ModelSettings | None, truncate_at_max_tokens: int = 9
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a one-message request whose tokens, BOS excluded, equal the limit."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a" * 7)],
        truncate_at_max_tokens=truncate_at_max_tokens,
        settings=settings or ModelSettings.none(),
    )


# --- Data builders shared by multiple request builders ---


def build_blue_6x4_image_chunk() -> ImageChunk:
    r"""Build an image chunk holding a blue 6x4 image."""
    return ImageChunk(image=Image.new(mode="RGB", size=(6, 4), color="blue"))


def build_string_properties_tool() -> Tool:
    r"""Build the tool `t` whose parameters are the string properties `g` and `h`."""
    return Tool(
        function=Function(
            name="t",
            parameters={
                "type": "object",
                "properties": {"g": {"type": "string"}, "h": {"type": "string"}},
            },
        )
    )


def build_tool_call_with_argument() -> ToolCall:
    r"""Build a call to the tool `t` with the argument `g="h"`."""
    return ToolCall(function=FunctionCall(name="t", arguments='{"g": "h"}'))


def build_two_tools() -> list[Tool]:
    r"""Build the tools `t1` and `t2`, both without parameters."""
    return [Tool(function=Function(name="t1", parameters={})), Tool(function=Function(name="t2", parameters={}))]


def build_two_calls_assistant_message(*, prefix: bool) -> AssistantMessage:
    r"""Build an assistant message with content `b1b2` and calls to `t1` and `t2`."""
    return AssistantMessage(
        content="b1b2",
        prefix=prefix,
        tool_calls=[
            ToolCall(id="000000000", function=FunctionCall(name="t1", arguments="{}")),
            ToolCall(id="111111111", function=FunctionCall(name="t2", arguments="{}")),
        ],
    )


# --- Prefixed assistant variants ---


def build_none_content_one_call_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of `None` content and one call."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            AssistantMessage(
                content=None,
                tool_calls=[ToolCall(function=FunctionCall(name="test_fn", arguments="{}"))],
                prefix=True,
            )
        ],
        settings=settings or ModelSettings.none(),
    )


def build_empty_content_one_call_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of empty content and one call."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(function=FunctionCall(name="test_fn", arguments="{}"))],
                prefix=True,
            )
        ],
        settings=settings or ModelSettings.none(),
    )


def build_none_content_no_calls_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of `None` content and no calls."""
    return InstructRequest[ChatMessage, Tool](
        messages=[AssistantMessage(content=None, tool_calls=None, prefix=True)],
        settings=settings or ModelSettings.none(),
    )


def build_none_content_empty_calls_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of `None` content and an empty call list."""
    return InstructRequest[ChatMessage, Tool](
        messages=[AssistantMessage(content=None, tool_calls=[], prefix=True)],
        settings=settings or ModelSettings.none(),
    )


def build_empty_content_no_calls_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of empty content and no calls."""
    return InstructRequest[ChatMessage, Tool](
        messages=[AssistantMessage(content="", tool_calls=None, prefix=True)],
        settings=settings or ModelSettings.none(),
    )


def build_empty_content_empty_calls_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a lone prefixed assistant message of empty content and an empty call list."""
    return InstructRequest[ChatMessage, Tool](
        messages=[AssistantMessage(content="", tool_calls=[], prefix=True)],
        settings=settings or ModelSettings.none(),
    )


def build_prefixed_text_request(content: str, *, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by a prefixed assistant turn with the given text content."""
    return build_user_then_assistant_request(
        assistant=AssistantMessage(content=content, prefix=True), settings=settings
    )


# --- Tool call variants ---


def build_system_tools_and_call_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a system message, a tool and an assistant turn calling it."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=[build_string_properties_tool()],
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="a"),
            AssistantMessage(content="b", tool_calls=[build_tool_call_with_argument()]),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_tool_result_after_history_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a tool result after consecutive user turns and a tool call."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=[build_string_properties_tool()],
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="a"),
            UserMessage(content="c"),
            AssistantMessage(content="b", tool_calls=[build_tool_call_with_argument()]),
            ToolMessage(content="b", tool_call_id="1234"),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_tool_call_and_content_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with an assistant turn combining content and two tool calls."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=build_two_tools(),
        messages=[UserMessage(content="a"), build_two_calls_assistant_message(prefix=False)],
        settings=settings or ModelSettings.none(),
    )


def build_prefixed_tool_call_and_content_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose final prefixed assistant turn combines content and two tool calls."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=build_two_tools(),
        messages=[UserMessage(content="a"), build_two_calls_assistant_message(prefix=True)],
        settings=settings or ModelSettings.none(),
    )


def build_tool_calls_with_two_results_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with an assistant turn of two tool calls followed by one result per call."""
    return InstructRequest[ChatMessage, Tool](
        available_tools=build_two_tools(),
        messages=[
            UserMessage(content="a"),
            build_two_calls_assistant_message(prefix=False),
            ToolMessage(content="r1", tool_call_id="000000000"),
            ToolMessage(content="r2", tool_call_id="111111111"),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_one_call_without_id_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by an assistant turn with one tool call lacking an ID (accepted in v11)."""
    return build_user_then_assistant_request(
        assistant=AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"))]),
        settings=settings,
    )


def build_two_calls_without_id_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by an assistant turn with two tool calls lacking IDs (accepted in v11)."""
    return build_user_then_assistant_request(
        assistant=AssistantMessage(
            tool_calls=[
                ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla")),
                ToolCall(function=FunctionCall(name="b", arguments="blu")),
            ]
        ),
        settings=settings,
    )


def build_call_with_id_before_last_user_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a tool call with an ID followed by the final user turn."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="a_a_a", arguments="blabla"), id="ABC")]),
            UserMessage(content="c"),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_tool_result_request(
    content: str | list[ContentChunk], *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with one assistant call and its tool result."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{}"))]),
            ToolMessage(content=content, tool_call_id="123456789"),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_tool_result_text_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose tool result is the text ``R1``."""
    return build_tool_result_request(content="R1", settings=settings)


def build_tool_result_text_chunks_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose tool result has two text chunks."""
    return build_tool_result_request(content=[TextChunk(text="R1"), TextChunk(text="R2")], settings=settings)


def build_tool_result_image_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a tool result containing one red image."""
    return build_tool_result_request(
        content=[ImageChunk(image=Image.new(mode="RGB", size=(4, 4), color="red"))], settings=settings
    )


def build_tools_before_first_user_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a tool listing and conversation with two user messages."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), AssistantMessage(content="b"), UserMessage(content="c")],
        available_tools=[Tool(function=Function(name="f", description="", parameters={}))],
        settings=settings or ModelSettings.none(),
    )


# --- Image + tool variants ---


def build_image_and_tool_result_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with an image user turn, a plain assistant turn and a tool result."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content=[TextChunk(text="a"), build_red_image_chunk()]),
            AssistantMessage(content="b"),
            ToolMessage(tool_call_id="b", content="f"),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_prefixed_assistant_after_image_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose final assistant message is a prefix after an image user turn."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content=[TextChunk(text="a"), build_red_image_chunk()]),
            AssistantMessage(content="b", prefix=True),
        ],
        settings=settings or ModelSettings.none(),
    )


# --- Image content variants ---


def build_empty_text_then_two_images_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with an empty text chunk, a red 4x4 image and a blue 6x4 image."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[TextChunk(text=""), build_red_image_chunk(), build_blue_6x4_image_chunk()])],
        settings=settings or ModelSettings.none(),
    )


def build_text_then_two_images_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a text chunk, a red 4x4 image and a blue 6x4 image."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[TextChunk(text="x"), build_red_image_chunk(), build_blue_6x4_image_chunk()])],
        settings=settings or ModelSettings.none(),
    )


def build_two_images_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a red 4x4 image followed by a blue 6x4 image."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[build_red_image_chunk(), build_blue_6x4_image_chunk()])],
        settings=settings or ModelSettings.none(),
    )


def build_trailing_image_moves_first_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single image follows a text chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[TextChunk(text="x"), build_red_image_chunk()])],
        settings=settings or ModelSettings.none(),
    )


def build_leading_image_remains_first_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single image precedes a text chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[build_red_image_chunk(), TextChunk(text="x")])],
        settings=settings or ModelSettings.none(),
    )


# --- System list content ---


def build_system_list_content_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose system message holds a single text chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content=[TextChunk(text="s")]), UserMessage(content="a")],
        settings=settings or ModelSettings.none(),
    )


# --- Truncation variants (success) ---


def build_truncation_keep_sys_and_last_message_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request where, next to the system prompts, only the last user message fits."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="c"),
            UserMessage(content="c"),
            SystemMessage(content="a"),
            UserMessage(content="bbbbbbb"),
        ],
        truncate_at_max_tokens=15,
        settings=settings or ModelSettings.none(),
    )


def build_truncation_drop_first_assistant_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request where dropping the first assistant message makes everything else fit."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            AssistantMessage(content="c"),
            UserMessage(content="b"),
            UserMessage(content="a"),
            UserMessage(content="aaaaaaa"),
        ],
        truncate_at_max_tokens=15,
        settings=settings or ModelSettings.none(),
    )


def build_truncation_start_with_assistant_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request whose truncated result may start with an assistant message because the input did."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            AssistantMessage(content="c"),
            AssistantMessage(content="b"),
            UserMessage(content="a"),
            UserMessage(content="aaaaaaa"),
        ],
        truncate_at_max_tokens=15,
        settings=settings or ModelSettings.none(),
    )


def build_truncation_drop_by_chunk_1_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request where the first assistant message, then the user and tool pair, are dropped."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            AssistantMessage(content="c"),
            UserMessage(content="c"),
            ToolMessage(content="c", tool_call_id="1234"),
            UserMessage(content="a"),
            AssistantMessage(content="bbbbbbb"),
        ],
        truncate_at_max_tokens=15,
        settings=settings or ModelSettings.none(),
    )


def build_truncation_drop_by_chunk_2_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request where only the last message is kept because the first chunk of three is too big."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="c"),
            AssistantMessage(content="c"),
            AssistantMessage(content="c"),
            UserMessage(content="aaaaaaa"),
        ],
        truncate_at_max_tokens=15,
        settings=settings or ModelSettings.none(),
    )


def build_truncation_full_convo_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 15-token request over a conversation with several system messages, truncated at its start."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            SystemMessage(content="a"),
            UserMessage(content="c"),
            AssistantMessage(content="c"),
            UserMessage(content="a"),
            AssistantMessage(content="a"),
            SystemMessage(content="b"),
            UserMessage(content="a"),
        ],
        truncate_at_max_tokens=15,
        settings=settings or ModelSettings.none(),
    )


# --- Truncation variants (rejected) ---


def build_truncation_system_prompt_too_long_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 9-token request whose only message is a system prompt that does not fit."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content="a" * 10)],
        truncate_at_max_tokens=9,
        settings=settings or ModelSettings.none(),
    )


def build_truncation_last_user_too_long_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a 9-token request whose only message is a last user message that does not fit."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a" * 10)],
        truncate_at_max_tokens=9,
        settings=settings or ModelSettings.none(),
    )


def build_truncation_last_user_one_over_request(
    *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a one-message request whose tokens, BOS excluded, exceed the limit by one."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a" * 8)],
        truncate_at_max_tokens=9,
        settings=settings or ModelSettings.none(),
    )


# --- Assistant think ---


def build_assistant_think_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose assistant message contains a think chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(content=[ThinkChunk(thinking="t"), TextChunk(text="b")]),
        ],
        settings=settings or ModelSettings.none(),
    )


# --- Multimodal data builders ---


def build_silent_audio_url_chunk() -> AudioURLChunk:
    r"""Build a silent audio URL chunk (data URI)."""
    audio_chunk = build_silent_audio_chunk()
    return AudioURLChunk(audio_url=str(audio_chunk.input_audio))


def build_red_image_url_chunk() -> ImageURLChunk:
    r"""Build a 4x4 solid-red PNG image URL chunk (data URI)."""
    img = Image.new("RGB", (4, 4), "red")
    buf = BytesIO()
    img.save(buf, "PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    return ImageURLChunk(image_url=data_url)


# --- Arbitrary tool call IDs ---


def build_arbitrary_tool_call_id_request(
    tool_call_id: str, *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a tool call using an arbitrary (non-standard) tool call ID."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id=tool_call_id,
                        function=FunctionCall(name="f", arguments="{}"),
                    )
                ],
            ),
            ToolMessage(content="b", tool_call_id=tool_call_id),
        ],
        settings=settings or ModelSettings.none(),
    )


# --- Non-ASCII tool call arguments ---


def build_non_ascii_call_arguments_request(
    tool_call_id: str | None = None, *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a tool call whose JSON arguments contain non-ASCII text."""
    tool_call = ToolCall(function=FunctionCall(name="f", arguments='{"x": "é"}'))
    if tool_call_id is not None:
        tool_call = ToolCall(id=tool_call_id, function=FunctionCall(name="f", arguments='{"x": "é"}'))
    return build_user_then_assistant_request(assistant=AssistantMessage(tool_calls=[tool_call]), settings=settings)


# --- System message with audio ---


def build_system_audio_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose system message contains text and audio chunks."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            SystemMessage(content=[TextChunk(text="System with content"), build_silent_audio_chunk()]),
            UserMessage(content="Hello"),
        ],
        settings=settings or ModelSettings.none(),
    )


# --- User message with audio URL ---


def build_user_audio_url_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose user message contains text and an audio URL chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content=[TextChunk(text="Here is content"), build_silent_audio_url_chunk()]),
        ],
        settings=settings or ModelSettings.none(),
    )


# --- User message with image URL ---


def build_user_image_url_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose user message contains text and an image URL chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            UserMessage(content=[TextChunk(text="Here is content"), build_red_image_url_chunk()]),
        ],
        settings=settings or ModelSettings.none(),
    )


# --- System text then user audio ---


def build_system_then_audio_user_request(
    *, num_samples: int = 2400, sampling_rate: int = 24_000, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a text-only system message followed by a user message with audio."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            SystemMessage(content="hello"),
            UserMessage(content=[build_silent_audio_chunk(num_samples=num_samples, sampling_rate=sampling_rate)]),
        ],
        settings=settings or ModelSettings.none(),
    )


# --- System message content variants ---


def build_system_request(
    content: str | list[ContentChunk], *, settings: ModelSettings | None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message followed by the user turn."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content=content), UserMessage(content="a")],
        settings=settings or ModelSettings.none(),
    )


def build_system_think_chunk_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose system message contains a think chunk followed by a user turn (rejected in v15)."""
    return InstructRequest[ChatMessage, Tool](
        messages=[
            SystemMessage(content=[ThinkChunk(thinking="Hi")]),
            UserMessage(content="Hello"),
        ],
        settings=settings or ModelSettings.none(),
    )


def build_system_text_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with plain text content."""
    return build_system_request(content="S1", settings=settings)


def build_system_text_chunk_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with one text chunk."""
    return build_system_request(content=[TextChunk(text="S1")], settings=settings)


def build_system_think_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a lone system message containing a think chunk (rejected in v15)."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content=[ThinkChunk(thinking="Hi")])],
        settings=settings or ModelSettings.none(),
    )


def build_system_text_then_think_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with text followed by a think chunk."""
    return build_system_request(content=[TextChunk(text="S1"), ThinkChunk(thinking="Hi")], settings=settings)


def build_system_think_between_text_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with text on both sides of a think chunk."""
    return build_system_request(
        content=[TextChunk(text="S1"), ThinkChunk(thinking="TS"), TextChunk(text="S2")], settings=settings
    )


def build_system_two_closed_thinks_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message with two closed think chunks between text."""
    return build_system_request(
        content=[
            TextChunk(text="S1"),
            TextChunk(text="S3"),
            ThinkChunk(thinking="TS", closed=True),
            ThinkChunk(thinking="TS", closed=True),
            TextChunk(text="S2"),
        ],
        settings=settings,
    )


def build_system_open_think_last_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a system message ending with an open think chunk."""
    return build_system_request(
        content=[TextChunk(text="S1"), TextChunk(text="S3"), ThinkChunk(thinking="TS", closed=False)],
        settings=settings,
    )


# --- Assistant content variants ---


def build_assistant_text_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build an assistant turn with plain text content."""
    return build_user_then_assistant_request(assistant=AssistantMessage(content="A1"), settings=settings)


def build_assistant_text_chunk_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build an assistant turn containing one text chunk."""
    return build_user_then_assistant_request(
        assistant=AssistantMessage(content=[TextChunk(text="A1")]), settings=settings
    )


def build_closed_think_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a closed assistant think chunk request."""
    return build_user_then_assistant_request(
        assistant=AssistantMessage(content=[ThinkChunk(thinking="T1")]), settings=settings
    )


def build_open_think_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build an open assistant think chunk request."""
    return build_user_then_assistant_request(
        assistant=AssistantMessage(content=[ThinkChunk(thinking="T1", closed=False)]),
        settings=settings,
    )


def build_closed_think_then_text_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a closed think chunk followed by assistant text."""
    return build_user_then_assistant_request(
        assistant=AssistantMessage(content=[ThinkChunk(thinking="T1"), TextChunk(text="A1")]),
        settings=settings,
    )


def build_open_think_text_and_call_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build assistant content with an open think chunk, text and a tool call."""
    return build_user_then_assistant_request(
        assistant=AssistantMessage(
            content=[ThinkChunk(thinking="R1", closed=False), TextChunk(text="A1")],
            tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{'a': 1}"))],
        ),
        settings=settings,

# --- Model settings and JSON schema builders (v15) ---


def build_math_interpreter_tool() -> Tool:
    r"""Build the math_interpreter tool used by model settings tests."""
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


def build_pipeline_conversation() -> list[ChatMessage]:
    r"""Build the multi-turn conversation used by model settings tests."""
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
        ToolMessage(content="R1", tool_call_id="123456789"),
        ToolMessage(content="R2", tool_call_id="999999999"),
        AssistantMessage(content="A2"),
        UserMessage(content="U2"),
    ]


def build_tools_pipeline_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with the math_interpreter tool and the pipeline conversation."""
    return InstructRequest[ChatMessage, Tool](
        messages=build_pipeline_conversation(),
        available_tools=[build_math_interpreter_tool()],
        settings=settings or ModelSettings.none(),
    )


def build_no_tools_pipeline_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with the pipeline conversation and no tools."""
    return InstructRequest[ChatMessage, Tool](
        messages=build_pipeline_conversation(),
        available_tools=None,
        settings=settings or ModelSettings.none(),
    )


def build_hi_user_request(*, settings: ModelSettings | None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request with a single user message holding 'hi'."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="hi")],
        settings=settings or ModelSettings.none(),
    )

r"""Shared instruct-request builders for versioned tokenizer tests.

These builders construct requests whose shape is identical across tokenizer versions.
Expected token IDs and decoded text stay in each version's test module.
"""

import numpy as np
from PIL import Image

from mistral_common.protocol.instruct.chunk import AudioChunk, ContentChunk, ImageChunk, TextChunk
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


def build_normal_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
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


def build_system_singleturn_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the single-turn request with a system prompt."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a")], system_prompt="SYSTEM", settings=settings or ModelSettings.none()
    )


def build_system_multiturn_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
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


def build_prefixed_final_message_request(
    *, settings: ModelSettings | None = None
) -> InstructRequest[ChatMessage, Tool]:
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


def build_tools_singleturn_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the single-turn request with one available tool."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a")],
        available_tools=[Tool(function=Function(name="tool1", description="1", parameters={}))],
        settings=settings or ModelSettings.none(),
    )


def build_tools_multiturn_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
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


def build_system_tools_multiturn_request(
    *, settings: ModelSettings | None = None
) -> InstructRequest[ChatMessage, Tool]:
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


def build_empty_assistant_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request containing an empty assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), AssistantMessage(content="")], settings=settings or ModelSettings.none()
    )


def build_system_message_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request containing a system message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[SystemMessage(content="s"), UserMessage(content="a")], settings=settings or ModelSettings.none()
    )


def build_tool_message_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build the request containing a tool result without a preceding assistant turn."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), ToolMessage(content="d", tool_call_id="123456789")],
        settings=settings or ModelSettings.none(),
    )


def build_tool_call_request(
    tool_call: ToolCall, *, settings: ModelSettings | None = None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by an assistant turn with the supplied tool call."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), AssistantMessage(tool_calls=[tool_call])],
        settings=settings or ModelSettings.none(),
    )


def build_assistant_tool_calls_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by an assistant turn with one identified tool call."""
    return build_tool_call_request(
        tool_call=ToolCall(id="123456789", function=FunctionCall(name="b", arguments="{}")), settings=settings
    )


def build_user_then_assistant_request(
    assistant: AssistantMessage, *, settings: ModelSettings | None = None
) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a user turn followed by the supplied assistant message."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content="a"), assistant], settings=settings or ModelSettings.none()
    )


def build_red_image_chunk() -> ImageChunk:
    r"""Build an image chunk holding a red 4x4 image."""
    return ImageChunk(image=Image.new(mode="RGB", size=(4, 4), color="red"))


def build_silent_audio_chunk(*, num_samples: int = 2400, sampling_rate: int = 24_000) -> AudioChunk:
    r"""Build an audio chunk holding silence."""
    audio = Audio(audio_array=np.zeros(shape=num_samples, dtype=np.float32), sampling_rate=sampling_rate, format="wav")
    return AudioChunk.from_audio(audio=audio)


def build_image_user_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single user message holds a red 4x4 image followed by text."""
    content: list[ContentChunk] = [build_red_image_chunk(), TextChunk(text="a")]
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=content)], settings=settings or ModelSettings.none()
    )


def build_text_then_image_user_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single user message holds text followed by a red 4x4 image."""
    content: list[ContentChunk] = [TextChunk(text="a"), build_red_image_chunk()]
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=content)], settings=settings or ModelSettings.none()
    )


def build_audio_user_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single user message holds a silent audio chunk."""
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=[build_silent_audio_chunk()])], settings=settings or ModelSettings.none()
    )


def build_text_then_audio_user_request(*, settings: ModelSettings | None = None) -> InstructRequest[ChatMessage, Tool]:
    r"""Build a request whose single user message holds text followed by a silent audio chunk."""
    content: list[ContentChunk] = [TextChunk(text="a"), build_silent_audio_chunk()]
    return InstructRequest[ChatMessage, Tool](
        messages=[UserMessage(content=content)], settings=settings or ModelSettings.none()
    )

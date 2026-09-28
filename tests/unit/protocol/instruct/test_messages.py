import warnings
from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from mistral_common.exceptions import InvalidAssistantMessageException
from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    BaseMessage,
    ChatMessage,
    ReasoningFieldFormat,
    Roles,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from tests.fixtures.chunks import get_content_chunks


def _construct_message_for_role(role: str, content: list[Any]) -> BaseMessage:
    match role:
        case "assistant":
            return AssistantMessage(role=Roles.assistant, content=content)
        case "system":
            return SystemMessage(role=Roles.system, content=content)
        case "user":
            return UserMessage(role=Roles.user, content=content)
        case "tool":
            return ToolMessage(role=Roles.tool, content=content, tool_call_id="c1")
        case _:
            raise ValueError(f"Unsupported role: {role}")


def test_user_preserves_compound_multimodal_content() -> None:
    content = get_content_chunks(("text", "image", "image_url", "audio", "audio_url"))
    expected_content = tuple(content)

    message = UserMessage(content=content)

    assert message.model_dump(exclude={"content"}) == {"role": "user"}
    assert isinstance(message.content, list)
    assert tuple(message.content) == expected_content


def test_assistant_preserves_text_and_thinking_content() -> None:
    content = get_content_chunks(("text", "think"))
    expected_content = tuple(content)

    message = AssistantMessage(content=content)

    assert message.model_dump(exclude={"content"}) == {"role": "assistant", "tool_calls": None, "prefix": False}
    assert isinstance(message.content, list)
    assert tuple(message.content) == expected_content


def test_system_preserves_text_audio_and_thinking_content() -> None:
    content = get_content_chunks(("text", "audio", "think"))
    expected_content = tuple(content)

    message = SystemMessage(content=content)

    assert message.model_dump(exclude={"content"}) == {"role": "system"}
    assert isinstance(message.content, list)
    assert tuple(message.content) == expected_content


def test_tool_preserves_non_thinking_content() -> None:
    content = get_content_chunks(("text", "image", "image_url", "audio", "audio_url"))
    expected_content = tuple(content)

    message = ToolMessage(content=content, tool_call_id="c1")

    assert message.model_dump(exclude={"content"}) == {"role": "tool", "tool_call_id": "c1", "name": None}
    assert isinstance(message.content, list)
    assert tuple(message.content) == expected_content


@pytest.mark.parametrize(
    ("role", "chunk_name", "chunk_type"),
    [
        pytest.param("assistant", "image", "ImageChunk", id="assistant-image"),
        pytest.param("assistant", "image_url", "ImageURLChunk", id="assistant-image-url"),
        pytest.param("assistant", "audio", "AudioChunk", id="assistant-audio"),
        pytest.param("assistant", "audio_url", "AudioURLChunk", id="assistant-audio-url"),
        pytest.param("system", "image", "ImageChunk", id="system-image"),
        pytest.param("system", "image_url", "ImageURLChunk", id="system-image-url"),
        pytest.param("system", "audio_url", "AudioURLChunk", id="system-audio-url"),
        pytest.param("user", "think", "ThinkChunk", id="user-think"),
        pytest.param("tool", "think", "ThinkChunk", id="tool-think"),
    ],
)
def test_message_rejects_forbidden_chunk_for_role(
    role: str,
    chunk_name: str,
    chunk_type: str,
) -> None:
    with pytest.raises(ValidationError, match=rf"{chunk_type} cannot be used in {role} message\."):
        _construct_message_for_role(role=role, content=get_content_chunks((chunk_name,)))


@pytest.mark.parametrize(
    ("openai_message", "message", "warn_on_export"),
    [
        pytest.param(
            {"role": "user", "content": "Hello"},
            UserMessage(content="Hello"),
            False,
            id="user-text",
        ),
        pytest.param(
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            UserMessage(content=[TextChunk(text="Hello")]),
            False,
            id="user-text-chunk",
        ),
        pytest.param(
            {"role": "assistant", "content": "Hi"},
            AssistantMessage(content="Hi"),
            False,
            id="assistant-text",
        ),
        pytest.param(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Hello", "closed": True},
                    {"type": "thinking", "thinking": "Hello", "closed": False},
                    {"type": "text", "text": "Hi"},
                ],
            },
            AssistantMessage(
                content=[
                    ThinkChunk(thinking="Hello", closed=True),
                    ThinkChunk(thinking="Hello", closed=False),
                    TextChunk(text="Hi"),
                ]
            ),
            True,
            id="assistant-thinking-and-text",
        ),
        pytest.param(
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [
                    {
                        "id": "VvvODy9mT",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "Paris, France", "format": "celsius"}',
                        },
                    }
                ],
            },
            AssistantMessage(
                content="Hi",
                tool_calls=[
                    ToolCall(
                        id="VvvODy9mT",
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments='{"location": "Paris, France", "format": "celsius"}',
                        ),
                    )
                ],
            ),
            False,
            id="assistant-tool-call",
        ),
        pytest.param(
            {"role": "tool", "content": "22", "tool_call_id": "VvvODy9mT"},
            ToolMessage(tool_call_id="VvvODy9mT", content="22"),
            False,
            id="tool-text",
        ),
        pytest.param(
            {
                "role": "tool",
                "content": [{"type": "text", "text": "22"}, {"type": "text", "text": "23"}],
                "tool_call_id": "VvvODy9mT",
            },
            ToolMessage(
                tool_call_id="VvvODy9mT",
                content=[TextChunk(text="22"), TextChunk(text="23")],
            ),
            False,
            id="tool-text-chunks",
        ),
        pytest.param(
            {"role": "system", "content": "You are a helpful assistant."},
            SystemMessage(content="You are a helpful assistant."),
            False,
            id="system-text",
        ),
        pytest.param(
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                    {"type": "thinking", "thinking": "Hello", "closed": False},
                ],
            },
            SystemMessage(
                content=[TextChunk(text="You are a helpful assistant."), ThinkChunk(thinking="Hello", closed=False)]
            ),
            False,
            id="system-text-and-thinking",
        ),
    ],
)
def test_openai_message_round_trip(openai_message: dict[str, Any], message: ChatMessage, warn_on_export: bool) -> None:
    assert type(message).from_openai(deepcopy(openai_message)) == message

    if warn_on_export:
        with pytest.warns(FutureWarning, match=r"convert_thinking_format.*defaults to 'thinking_chunks'"):
            assert deepcopy(message).to_openai() == openai_message
    else:
        assert deepcopy(message).to_openai() == openai_message


@pytest.mark.parametrize(
    ("openai_message", "expected"),
    [
        pytest.param(
            {"role": "assistant", "content": "Hi", "reasoning": "Let me think..."},
            AssistantMessage(content=[ThinkChunk(thinking="Let me think...", closed=True), TextChunk(text="Hi")]),
            id="reasoning-before-string-content",
        ),
        pytest.param(
            {
                "role": "assistant",
                "content": None,
                "reasoning": "Thinking aloud",
                "reasoning_content": "Thinking aloud",
            },
            AssistantMessage(content=[ThinkChunk(thinking="Thinking aloud", closed=True)]),
            id="matching-reasoning-fields-without-content",
        ),
        pytest.param(
            {"role": "assistant", "reasoning": "Thinking aloud"},
            AssistantMessage(content=[ThinkChunk(thinking="Thinking aloud", closed=True)]),
            id="reasoning-without-content",
        ),
        pytest.param(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "reasoning": "Deep thought",
            },
            AssistantMessage(content=[ThinkChunk(thinking="Deep thought", closed=True), TextChunk(text="Hello")]),
            id="reasoning-before-text-chunk",
        ),
        pytest.param(
            {"role": "assistant", "content": "Hi", "reasoning_content": "Only reasoning"},
            AssistantMessage(content=[ThinkChunk(thinking="Only reasoning", closed=True), TextChunk(text="Hi")]),
            id="reasoning-content-before-string-content",
        ),
    ],
)
def test_from_openai_reasoning_in_assistant_message(openai_message: dict[str, Any], expected: AssistantMessage) -> None:
    assert AssistantMessage.from_openai(deepcopy(openai_message)) == expected


def test_from_openai_rejects_differing_reasoning_fields() -> None:
    openai_message = {"role": "assistant", "content": "Hi", "reasoning": "Primary", "reasoning_content": "Fallback"}

    with pytest.raises(ValueError, match=r"`reasoning_content` and `reasoning` should be equal"):
        AssistantMessage.from_openai(openai_message)


@pytest.mark.parametrize(
    "openai_message",
    [
        pytest.param(
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "hmm", "closed": True}, {"type": "text", "text": "Hi"}],
                "reasoning": "also thinking",
            },
            id="reasoning-with-thinking-and-text-chunks",
        ),
        pytest.param(
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "hmm", "closed": True}],
                "reasoning_content": "also thinking",
            },
            id="reasoning-content-with-thinking-chunk",
        ),
        pytest.param(
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "hmm", "closed": True}],
                "reasoning": "also thinking",
                "reasoning_content": "also thinking",
            },
            id="both-reasoning-fields-with-thinking-chunk",
        ),
    ],
)
def test_from_openai_rejects_thinking_chunks_with_reasoning_fields(openai_message: dict[str, Any]) -> None:
    with pytest.raises(InvalidAssistantMessageException):
        AssistantMessage.from_openai(deepcopy(openai_message))


def test_non_leading_think_chunks_are_allowed_at_construction() -> None:
    message = AssistantMessage(
        content=[
            ThinkChunk(thinking="First", closed=True),
            TextChunk(text="Reply"),
            ThinkChunk(thinking="Third", closed=False),
        ]
    )

    assert message == AssistantMessage(
        content=[
            ThinkChunk(thinking="First", closed=True),
            TextChunk(text="Reply"),
            ThinkChunk(thinking="Third", closed=False),
        ]
    )


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            [ThinkChunk(thinking="First", closed=True), TextChunk(text="Reply"), ThinkChunk(thinking="Third")],
            id="think-after-text",
        ),
        pytest.param(
            [TextChunk(text="Reply"), ThinkChunk(thinking="After", closed=True)],
            id="think-after-only-text",
        ),
        pytest.param(
            [TextChunk(text="A"), TextChunk(text="B"), ThinkChunk(thinking="End", closed=True)],
            id="think-after-multiple-text-chunks",
        ),
    ],
)
def test_non_leading_think_chunks_are_rejected_by_to_openai(content: list[Any]) -> None:
    message = AssistantMessage(content=deepcopy(content))

    with pytest.raises(InvalidAssistantMessageException, match="ThinkChunks must be leading"):
        message.to_openai()


@pytest.mark.parametrize(
    ("message", "reasoning_field_format", "expected"),
    [
        pytest.param(
            AssistantMessage(content=[ThinkChunk(thinking="Deep thought", closed=True), TextChunk(text="Answer")]),
            ReasoningFieldFormat.thinking_chunks,
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Deep thought", "closed": True},
                    {"type": "text", "text": "Answer"},
                ],
            },
            id="thinking-chunks-inline",
        ),
        pytest.param(
            AssistantMessage(
                content=[
                    ThinkChunk(thinking="First", closed=True),
                    ThinkChunk(thinking="Second", closed=False),
                    TextChunk(text="Reply"),
                ]
            ),
            ReasoningFieldFormat.thinking_chunks,
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "First", "closed": True},
                    {"type": "thinking", "thinking": "Second", "closed": False},
                    {"type": "text", "text": "Reply"},
                ],
            },
            id="thinking-chunks-preserve-order-and-closure",
        ),
        pytest.param(
            AssistantMessage(content=[ThinkChunk(thinking="Let me think", closed=True), TextChunk(text="Done")]),
            ReasoningFieldFormat.reasoning,
            {"role": "assistant", "reasoning": "Let me think", "content": "Done"},
            id="reasoning-single-leading-chunk",
        ),
        pytest.param(
            AssistantMessage(content=[ThinkChunk(thinking="Pondering", closed=True), TextChunk(text="Result")]),
            ReasoningFieldFormat.reasoning_content,
            {"role": "assistant", "reasoning_content": "Pondering", "content": "Result"},
            id="reasoning-content-single-leading-chunk",
        ),
        pytest.param(
            AssistantMessage(
                content=[
                    ThinkChunk(thinking="Part 1", closed=True),
                    ThinkChunk(thinking="Part 2", closed=True),
                    TextChunk(text="Final"),
                ]
            ),
            ReasoningFieldFormat.reasoning,
            {"role": "assistant", "reasoning": "Part 1\nPart 2", "content": "Final"},
            id="reasoning-joins-leading-chunks",
        ),
        pytest.param(
            AssistantMessage(content=[ThinkChunk(thinking="Just thinking", closed=True)]),
            ReasoningFieldFormat.thinking_chunks,
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Just thinking", "closed": True}]},
            id="thinking-chunks-without-text",
        ),
        pytest.param(
            AssistantMessage(content=[ThinkChunk(thinking="Only reasoning", closed=True)]),
            ReasoningFieldFormat.reasoning,
            {"role": "assistant", "reasoning": "Only reasoning"},
            id="reasoning-without-text",
        ),
        pytest.param(
            AssistantMessage(
                content=[
                    ThinkChunk(thinking="Think", closed=True),
                    TextChunk(text="A"),
                    TextChunk(text="B"),
                ]
            ),
            ReasoningFieldFormat.reasoning,
            {
                "role": "assistant",
                "reasoning": "Think",
                "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
            },
            id="reasoning-with-multiple-text-chunks",
        ),
        pytest.param(
            AssistantMessage(content="Simple text"),
            ReasoningFieldFormat.reasoning,
            {"role": "assistant", "content": "Simple text"},
            id="string-content-unchanged",
        ),
        pytest.param(
            AssistantMessage(content=None),
            ReasoningFieldFormat.thinking_chunks,
            {"role": "assistant"},
            id="none-content-unchanged",
        ),
    ],
)
def test_to_openai_converts_thinking_format(
    message: AssistantMessage,
    reasoning_field_format: ReasoningFieldFormat,
    expected: dict[str, Any],
) -> None:
    assert deepcopy(message).to_openai(reasoning_field_format=reasoning_field_format) == expected


def test_to_openai_warns_when_thinking_format_is_omitted_with_think_chunks() -> None:
    message = AssistantMessage(content=[ThinkChunk(thinking="Hmm", closed=True), TextChunk(text="Answer")])

    with pytest.warns(FutureWarning, match=r"convert_thinking_format.*defaults to 'thinking_chunks'"):
        result = message.to_openai()

    assert result == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Hmm", "closed": True},
            {"type": "text", "text": "Answer"},
        ],
    }


def test_to_openai_does_not_warn_without_think_chunks() -> None:
    message = AssistantMessage(content="Plain text")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = message.to_openai()

    assert result == {"role": "assistant", "content": "Plain text"}


def test_to_openai_does_not_warn_with_none_content() -> None:
    message = AssistantMessage(content=None)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = message.to_openai()

    assert result == {"role": "assistant"}

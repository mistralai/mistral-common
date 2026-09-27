from typing import Any

import pytest
from pydantic import ValidationError

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tests.fixtures.chunks import get_content_chunks


def _construct_message_for_role(role: str, content: list[Any]) -> BaseMessage:
    match role:
        case "assistant":
            return AssistantMessage(content=content)
        case "system":
            return SystemMessage(content=content)
        case "user":
            return UserMessage(content=content)
        case "tool":
            return ToolMessage(content=content, tool_call_id="c1")
        case _:
            raise ValueError(f"Unsupported role: {role}")


def test_user_preserves_compound_multimodal_content() -> None:
    content = get_content_chunks(("text", "image", "image_url", "audio", "audio_url"))

    message = UserMessage(content=content)

    assert message == UserMessage(role="user", content=content)


def test_assistant_preserves_text_and_thinking_content() -> None:
    content = get_content_chunks(("text", "think"))

    message = AssistantMessage(content=content)

    assert message == AssistantMessage(role="assistant", content=content)


def test_system_preserves_text_audio_and_thinking_content() -> None:
    content = get_content_chunks(("text", "audio", "think"))

    message = SystemMessage(content=content)

    assert message == SystemMessage(role="system", content=content)


def test_tool_preserves_non_thinking_content() -> None:
    content = get_content_chunks(("text", "image", "image_url", "audio", "audio_url"))

    message = ToolMessage(content=content, tool_call_id="c1")

    assert message == ToolMessage(role="tool", content=content, tool_call_id="c1")


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

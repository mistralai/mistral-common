import pytest
from pydantic import ValidationError

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tests.fixtures.chunks import get_content_chunks as _chunks


class TestMessageContentChunkUnions:
    r"""Pydantic-level (version-independent) content-chunk unions for each message role."""

    def test_user_allows_text_image_audio(self) -> None:
        UserMessage(content=_chunks(("text", "image", "image_url", "audio", "audio_url")))

    def test_user_rejects_think(self) -> None:
        for name in ("think",):
            with pytest.raises(ValidationError):
                UserMessage(content=_chunks((name,)))

    def test_assistant_allows_text_and_think(self) -> None:
        AssistantMessage(content=_chunks(("text", "think")))

    def test_assistant_rejects_image_and_audio(self) -> None:
        for name in ("image", "image_url", "audio", "audio_url"):
            with pytest.raises(ValidationError):
                AssistantMessage(content=_chunks((name,)))

    def test_system_allows_text_audio_think(self) -> None:
        SystemMessage(content=_chunks(("text", "audio", "think")))

    def test_system_rejects_image_and_audio_url(self) -> None:
        for name in ("image", "image_url", "audio_url"):
            with pytest.raises(ValidationError):
                SystemMessage(content=_chunks((name,)))

    def test_tool_allows_all_chunk_types(self) -> None:
        ToolMessage(
            content=_chunks(("text", "image", "image_url", "audio", "audio_url", "think")),
            tool_call_id="c1",
        )

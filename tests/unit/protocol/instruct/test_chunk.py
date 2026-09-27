from typing import Any

import pytest
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam as OpenAITextChunk,
)
from pydantic import ValidationError

from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURL, AudioURLChunk, TextChunk, ThinkChunk


def test_text_chunk_round_trip() -> None:
    chunk = TextChunk(text="Hello")
    openai_chunk = chunk.to_openai()

    assert openai_chunk == {"type": "text", "text": "Hello"}
    assert TextChunk.from_openai(openai_chunk) == chunk
    assert TextChunk.from_openai(OpenAITextChunk(**openai_chunk)) == chunk  # type: ignore[typeddict-item]


def test_think_chunk_round_trip() -> None:
    chunk = ThinkChunk(thinking="Hello", closed=False)

    assert chunk.to_openai() == {"type": "thinking", "thinking": "Hello", "closed": False}
    assert ThinkChunk.from_openai(chunk.to_openai()) == chunk


@pytest.mark.parametrize(
    ("openai_chunk", "expected"),
    [
        pytest.param(
            {"type": "text", "text": "Hello", "annotations": []},
            TextChunk(text="Hello"),
            id="text-annotations",
        ),
        pytest.param(
            {"type": "thinking", "thinking": "hmm", "closed": True, "extra": 1},
            ThinkChunk(thinking="hmm", closed=True),
            id="thinking-extra-field",
        ),
        pytest.param(
            {
                "type": "audio_url",
                "audio_url": {"url": "https://example.com/audio.wav"},
                "extra": True,
            },
            AudioURLChunk(audio_url=AudioURL(url="https://example.com/audio.wav")),
            id="audio-url-extra-field",
        ),
        pytest.param(
            {
                "type": "input_audio",
                "input_audio": {"data": "audio-data", "format": "wav"},
                "extra": True,
            },
            AudioChunk(input_audio="audio-data"),
            id="input-audio-data-and-extra-field",
        ),
    ],
)
def test_from_openai_drops_unknown_fields_and_converts_known_data(
    openai_chunk: dict[str, Any], expected: TextChunk | ThinkChunk | AudioURLChunk | AudioChunk
) -> None:
    chunk_type = expected.__class__

    assert chunk_type.from_openai(openai_chunk) == expected


@pytest.mark.parametrize(
    ("openai_chunk", "field"),
    [
        pytest.param({"type": "text", "text": 42}, "text", id="text-is-not-a-string"),
        pytest.param(
            {"type": "input_audio", "input_audio": {"data": []}},
            "input_audio",
            id="audio-data-is-not-a-string",
        ),
    ],
)
def test_from_openai_rejects_invalid_recognized_data(openai_chunk: dict[str, Any], field: str) -> None:
    chunk_type = TextChunk if field == "text" else AudioChunk

    with pytest.raises(ValidationError, match=field):
        chunk_type.from_openai(openai_chunk)

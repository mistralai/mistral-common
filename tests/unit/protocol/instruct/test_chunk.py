from typing import Any

import pytest
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam as OpenAITextChunk,
)
from pydantic import ValidationError

from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURL, AudioURLChunk, TextChunk, ThinkChunk


@pytest.mark.parametrize(
    ("chunk", "openai_chunk", "canonical_chunk"),
    [
        pytest.param(
            TextChunk(text="Hello"),
            {"type": "text", "text": "Hello"},
            TextChunk(text="Hello"),
            id="text",
        ),
        pytest.param(
            ThinkChunk(thinking="Hello", closed=False),
            {"type": "thinking", "thinking": "Hello", "closed": False},
            ThinkChunk(thinking="Hello", closed=False),
            id="thinking-open",
        ),
        pytest.param(
            ThinkChunk(thinking="Finished"),
            {"type": "thinking", "thinking": "Finished", "closed": True},
            ThinkChunk(thinking="Finished", closed=True),
            id="thinking-default-closed",
        ),
    ],
)
def test_text_and_thinking_chunks_convert_with_explicit_canonical_values(
    chunk: TextChunk | ThinkChunk,
    openai_chunk: dict[str, Any],
    canonical_chunk: TextChunk | ThinkChunk,
) -> None:
    assert chunk.to_openai() == openai_chunk
    assert type(chunk).from_openai(openai_chunk) == canonical_chunk


def test_text_chunk_from_openai_accepts_openai_typed_dict() -> None:
    openai_chunk = OpenAITextChunk(type="text", text="Hello")

    assert TextChunk.from_openai(openai_chunk) == TextChunk(text="Hello")  # type: ignore[arg-type]


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


@pytest.mark.parametrize(
    ("openai_chunk", "field"),
    [
        pytest.param({"type": "thinking", "thinking": 42}, "thinking", id="thinking-is-not-a-string"),
        pytest.param({"type": "thinking", "thinking": "hmm", "closed": "unknown"}, "closed", id="closed-is-not-a-bool"),
    ],
)
def test_think_chunk_from_openai_rejects_invalid_recognized_fields(openai_chunk: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        ThinkChunk.from_openai(openai_chunk)

import base64
import io
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam as OpenAIImageChunk,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam as OpenAITextChunk,
)
from PIL import Image
from pydantic import ValidationError

from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURL,
    AudioURLChunk,
    ImageChunk,
    ImageURL,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
)


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


def test_thinking_chunk_from_openai_defaults_missing_closed_to_true() -> None:
    openai_chunk = {"type": "thinking", "thinking": "Finished"}

    imported = ThinkChunk.from_openai(openai_chunk)

    assert imported == ThinkChunk(thinking="Finished", closed=True)
    assert imported.to_openai() == {"type": "thinking", "thinking": "Finished", "closed": True}


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
    ("format", "representation"),
    [
        pytest.param("wav", "base64", id="wav-base64"),
        pytest.param("wav", "bytes", id="wav-raw-bytes"),
        pytest.param("wav", "data-url", id="wav-data-url"),
        pytest.param("flac", "base64", id="flac-base64"),
        pytest.param("flac", "bytes", id="flac-raw-bytes"),
        pytest.param("flac", "data-url", id="flac-data-url"),
    ],
)
def test_audio_chunk_conversion_preserves_data_and_detected_format(format: str, representation: str) -> None:
    audio_buffer = io.BytesIO()
    audio_samples = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)
    sf.write(file=audio_buffer, data=audio_samples, samplerate=16000, format=format)
    audio_bytes = audio_buffer.getvalue()
    expected_data = base64.b64encode(audio_bytes).decode("ascii")

    if representation == "bytes":
        input_audio: str | bytes = audio_bytes
    elif representation == "data-url":
        input_audio = f"data:audio/{format};base64,{expected_data}"
    else:
        input_audio = expected_data

    chunk = AudioChunk(input_audio=input_audio)
    expected_openai_chunk = {
        "type": "input_audio",
        "input_audio": {"data": expected_data, "format": format},
    }
    openai_chunk = chunk.to_openai()

    assert openai_chunk == expected_openai_chunk
    assert AudioChunk.from_openai(openai_chunk) == AudioChunk(input_audio=expected_data)


@pytest.mark.parametrize(
    ("url", "representation"),
    [
        pytest.param("https://example.com/audio.wav", "nested", id="https-url-object"),
        pytest.param("YXVkaW8=", "nested", id="base64-url-object"),
        pytest.param("YXVkaW8=", "string", id="base64-plain-string"),
        pytest.param("data:audio/wav;base64,YXVkaW8=", "nested", id="prefixed-data-url-object"),
    ],
)
def test_audio_url_chunk_conversion_preserves_url(url: str, representation: str) -> None:
    audio_url = url if representation == "string" else AudioURL(url=url)
    chunk = AudioURLChunk(audio_url=audio_url)
    expected_openai_chunk = {"type": "audio_url", "audio_url": {"url": url}}
    canonical_chunk = AudioURLChunk(audio_url=AudioURL(url=url))
    openai_chunk = chunk.to_openai()

    assert openai_chunk == expected_openai_chunk
    assert AudioURLChunk.from_openai(openai_chunk) == canonical_chunk


def test_audio_url_chunk_from_openai_rejects_invalid_known_url() -> None:
    openai_chunk = {"type": "audio_url", "audio_url": {"url": 42}}

    with pytest.raises(ValidationError, match="url"):
        AudioURLChunk.from_openai(openai_chunk)


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


def test_image_chunk_export_and_import_preserve_image_values() -> None:
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 0, 255))
    chunk = ImageChunk(image=image)
    expected_openai_chunk = {
        "type": "image_url",
        "image_url": {
            "url": (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAD0lEQVR4nGP4z8DAwPAfAAcAAf9+CLHQAAAAAElFTkSuQmCC"
            )
        },
    }

    assert chunk.to_openai() == expected_openai_chunk

    canonical_chunk = ImageChunk.from_openai(expected_openai_chunk)
    assert canonical_chunk.image.mode == "RGB"
    assert canonical_chunk.image.size == (2, 1)
    assert list(canonical_chunk.image.getdata()) == [(255, 0, 0), (0, 0, 255)]


def test_image_chunk_from_openai_does_not_mutate_input() -> None:
    openai_chunk = {
        "type": "image_url",
        "image_url": {
            "url": (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAD0lEQVR4nGP4z8DAwPAfAAcAAf9+CLHQAAAAAElFTkSuQmCC"
            )
        },
    }
    original_openai_chunk = deepcopy(openai_chunk)

    ImageChunk.from_openai(openai_chunk)

    assert openai_chunk == original_openai_chunk


def test_image_chunk_from_openai_rejects_missing_nested_url() -> None:
    openai_chunk = {"type": "image_url", "image_url": {"detail": "high"}}

    with pytest.raises(AssertionError, match=r"\{'detail': 'high'\}"):
        ImageChunk.from_openai(openai_chunk)


@pytest.mark.parametrize(
    ("openai_chunk", "image_url_chunk", "canonical_chunk"),
    [
        pytest.param(
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                    "detail": "auto",
                },
            },
            ImageURLChunk(
                image_url=ImageURL(
                    url="https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                    detail="auto",
                )
            ),
            ImageURLChunk(
                image_url=ImageURL(
                    url="https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
                    detail="auto",
                )
            ),
            id="https-url-with-detail",
        ),
        pytest.param(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0"}},
            ImageURLChunk(image_url="data:image/png;base64,iVBORw0"),
            ImageURLChunk(image_url=ImageURL(url="data:image/png;base64,iVBORw0", detail=None)),
            id="data-url-without-detail",
        ),
    ],
)
def test_image_url_chunk_conversion(
    openai_chunk: dict[str, Any],
    image_url_chunk: ImageURLChunk,
    canonical_chunk: ImageURLChunk,
) -> None:
    assert image_url_chunk.to_openai() == openai_chunk
    assert ImageURLChunk.from_openai(openai_chunk) == canonical_chunk
    assert ImageURLChunk.from_openai(OpenAIImageChunk(**openai_chunk)) == canonical_chunk  # type: ignore[typeddict-item]

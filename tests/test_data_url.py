import base64
import io

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from mistral_common.protocol.instruct.chunk import AudioChunk, ImageChunk, ImageURLChunk
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.image import image_from_chunk
from mistral_common.utils.data_url import maybe_strip_base64_data_url_prefix


def _png_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _wav_b64() -> str:
    audio = Audio(audio_array=np.zeros(1600, dtype=np.float32), sampling_rate=16000, format="wav")
    return audio.to_base64("wav")


@pytest.mark.parametrize(
    "value,expected",
    [
        # Base64 data URLs are unwrapped.
        ("data:image/png;base64,aGVsbG8=", "aGVsbG8="),
        ("data:audio/wav;base64,aGVsbG8=", "aGVsbG8="),
        # RFC 6838 subtypes contain "-", "+" and "." (mistralai/mistral-common#340).
        ("data:image/svg+xml;base64,aGVsbG8=", "aGVsbG8="),
        ("data:image/x-icon;base64,aGVsbG8=", "aGVsbG8="),
        ("data:image/vnd.microsoft.icon;base64,aGVsbG8=", "aGVsbG8="),
        ("data:audio/x-wav;base64,aGVsbG8=", "aGVsbG8="),
        # Media-type parameters are part of the header, not the payload.
        ("data:image/png;charset=utf-8;base64,aGVsbG8=", "aGVsbG8="),
        # Only the first comma splits header from payload.
        ("data:image/png;base64,YWJj,ZGVm", "YWJj,ZGVm"),
        # Non-base64 data URLs, and strings that are not data URLs, pass through.
        ("data:text/plain,hello", "data:text/plain,hello"),
        ("data:image/png;base64", "data:image/png;base64"),
        ("data:,hello", "data:,hello"),
        ("aGVsbG8=", "aGVsbG8="),
        ("https://example.com/image.png", "https://example.com/image.png"),
        ("", ""),
    ],
)
def test_maybe_strip_base64_data_url_prefix_lenient(value: str, expected: str) -> None:
    assert maybe_strip_base64_data_url_prefix(value, strict=False) == expected


def test_image_chunk_round_trips_data_url_serialization() -> None:
    chunk = ImageChunk(image=Image.new("RGB", (4, 4), "red"))
    dumped = chunk.model_dump(context={"add_format_prefix": True})
    assert dumped["image"].startswith("data:image/png;base64,")
    assert ImageChunk.model_validate(dumped).image.size == (4, 4)


def test_image_chunk_accepts_data_url_string() -> None:
    # `image` is typed as a PIL Image; the BeforeValidator accepts base64 strings and data URLs.
    chunk = ImageChunk(image=f"data:image/png;base64,{_png_b64()}")  # type: ignore[arg-type]
    assert chunk.image.size == (1, 1)


@pytest.mark.parametrize("subtype", ["image/x-icon", "image/svg+xml", "image/vnd.microsoft.icon"])
def test_image_chunk_from_openai_strips_data_url_with_non_word_subtype(subtype: str) -> None:
    chunk = ImageChunk.from_openai({"type": "image_url", "image_url": {"url": f"data:{subtype};base64,{_png_b64()}"}})
    assert chunk.image.size == (1, 1)


def test_audio_from_base64_strips_data_url_with_non_word_subtype() -> None:
    b64 = _wav_b64()
    audio = Audio.from_base64(f"data:audio/x-wav;base64,{b64}")
    assert audio.format == "wav"


def test_audio_chunk_to_openai_strips_data_url_with_non_word_subtype() -> None:
    b64 = _wav_b64()
    chunk = AudioChunk(input_audio=f"data:audio/x-wav;base64,{b64}")

    result = chunk.to_openai()

    assert result["input_audio"]["format"] == "wav"
    assert result["input_audio"]["data"] == b64


def test_audio_chunk_to_openai_strips_data_url_with_media_type_parameter() -> None:
    b64 = _wav_b64()
    chunk = AudioChunk(input_audio=f"data:audio/wav;charset=utf-8;base64,{b64}")

    result = chunk.to_openai()

    assert result["input_audio"]["format"] == "wav"
    assert result["input_audio"]["data"] == b64


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_audio_chunk_to_openai_strips_base64_data_url_prefix(fmt: str) -> None:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16000, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    chunk = AudioChunk(input_audio=f"data:audio/{fmt};base64,{b64}")

    result = chunk.to_openai()

    assert result["input_audio"]["format"] == fmt
    assert result["input_audio"]["data"] == b64
    assert AudioChunk.from_openai(result).input_audio == b64


@pytest.mark.parametrize(
    "value",
    [
        "data:text/plain,hello",  # non-base64 data URL
        "data:,hello",  # no `;base64` header
        "data:image/png;base64",  # no comma
        "data:image/png;base64,",  # empty payload
    ],
)
def test_maybe_strip_base64_data_url_prefix_strict_rejects_malformed_data_urls(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid base64 data URL"):
        maybe_strip_base64_data_url_prefix(value, strict=True)


@pytest.mark.parametrize("value", ["aGVsbG8=", "https://example.com/image.png", ""])
def test_maybe_strip_base64_data_url_prefix_strict_passes_through_non_data_urls(value: str) -> None:
    assert maybe_strip_base64_data_url_prefix(value, strict=True) == value


def test_image_from_chunk_accepts_mislabeled_mime_type() -> None:
    # The MIME type is not checked: the payload is sniffed by PIL, so a real
    # image under a generic MIME type still loads.
    chunk = ImageURLChunk(image_url=f"data:application/octet-stream;base64,{_png_b64()}")
    assert image_from_chunk(chunk).size == (1, 1)


def test_image_from_chunk_rejects_non_image_payload() -> None:
    # Valid base64 of valid audio, passed as an image.
    chunk = ImageURLChunk(image_url=f"data:audio/wav;base64,{_wav_b64()}")
    with pytest.raises(ValueError, match="payload is not a decodable image"):
        image_from_chunk(chunk)


def test_image_from_chunk_rejects_invalid_base64_payload() -> None:
    chunk = ImageURLChunk(image_url="data:image/png;base64,aGVsbG8")  # bad padding
    with pytest.raises(ValueError, match="payload is not valid base64"):
        image_from_chunk(chunk)


@pytest.mark.parametrize("url", ["data:text/plain,hello", "data:image/png;base64", "data:image/png;base64,"])
def test_image_from_chunk_rejects_malformed_data_url(url: str) -> None:
    with pytest.raises(ValueError, match="Invalid base64 data URL"):
        image_from_chunk(ImageURLChunk(image_url=url))


def test_image_chunk_from_openai_malformed_data_url_fails_in_validator() -> None:
    # from_openai strips leniently; the malformed data URL then fails in the
    # image validator, like a direct ImageChunk(image=...) construction.
    openai_chunk = {"type": "image_url", "image_url": {"url": "data:text/plain,hello"}}
    with pytest.raises(RuntimeError, match="Encountered an error when loading image"):
        ImageChunk.from_openai(openai_chunk)


def test_audio_from_base64_rejects_non_audio_payload() -> None:
    # Valid base64 of a valid image, passed as audio.
    with pytest.raises(ValueError, match="Failed to decode the base64 payload as audio"):
        Audio.from_base64(f"data:audio/wav;base64,{_png_b64()}")


def test_audio_from_base64_rejects_malformed_data_url() -> None:
    with pytest.raises(ValueError, match="Invalid base64 data URL"):
        Audio.from_base64("data:text/plain,hello")


def test_audio_chunk_to_openai_rejects_malformed_data_url() -> None:
    chunk = AudioChunk(input_audio="data:text/plain,hello")
    with pytest.raises(ValueError, match="Invalid base64 data URL"):
        chunk.to_openai()


def test_audio_chunk_to_openai_rejects_invalid_base64() -> None:
    chunk = AudioChunk(input_audio="aGVsbG8")  # bad padding
    with pytest.raises(ValueError, match="base64 decoding failed"):
        chunk.to_openai()

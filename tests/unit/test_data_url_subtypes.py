import io
import pytest
from PIL import Image

from mistral_common.protocol.instruct.chunk import ImageChunk, _strip_audio_data_url_prefix
from mistral_common.tokens.tokenizers.audio import Audio


def _create_minimal_wav_bytes() -> bytes:
    # Minimal 44-byte WAV header with 0 samples
    return (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        b"\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00"
        b"data\x00\x00\x00\x00"
    )


def test_strip_audio_data_url_prefix_standard_and_subtypes() -> None:
    # Standard format
    assert _strip_audio_data_url_prefix("data:audio/wav;base64,AAAA") == "AAAA"
    # RFC 6838 MIME subtypes with hyphens, pluses, periods
    assert _strip_audio_data_url_prefix("data:audio/x-wav;base64,BBBB") == "BBBB"
    assert _strip_audio_data_url_prefix("data:audio/vnd.wave;base64,CCCC") == "CCCC"
    assert _strip_audio_data_url_prefix("data:audio/ogg; codecs=opus;base64,DDDD") == "DDDD"
    # Non-data-url or non-audio prefix should remain unchanged
    assert _strip_audio_data_url_prefix("AAAA") == "AAAA"
    assert _strip_audio_data_url_prefix("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"


@pytest.mark.parametrize(
    "mime_type",
    [
        "image/jpeg",
        "image/png",
        "image/svg+xml",
        "image/x-icon",
        "image/vnd.microsoft.icon",
        "image/webp",
    ],
)
def test_image_chunk_from_openai_mime_subtypes(mime_type: str) -> None:
    # Create 1x1 image base64
    img = Image.new("RGB", (1, 1), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64
    b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    data_url = f"data:{mime_type};base64,{b64_data}"
    openai_chunk = {"type": "image_url", "image_url": {"url": data_url}}

    chunk = ImageChunk.from_openai(openai_chunk)
    # The base64 prefix should have been stripped, so chunk.image holds the loaded PIL image
    assert chunk is not None
    assert isinstance(chunk.image, Image.Image)


@pytest.mark.parametrize(
    "mime_type",
    [
        "audio/wav",
        "audio/x-wav",
        "audio/vnd.wave",
    ],
)
def test_audio_from_base64_mime_subtypes(mime_type: str) -> None:
    import base64
    wav_bytes = _create_minimal_wav_bytes()
    b64_payload = base64.b64encode(wav_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64_payload}"

    audio = Audio.from_base64(data_url)
    assert audio.sampling_rate == 44100
    assert audio.format == "wav"

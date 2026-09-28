import base64

import numpy as np
import pytest
from openai.types.audio.transcription_create_params import TranscriptionCreateParamsBase
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
from tests.unit.protocol.audio_conversion import assert_audio_buffer, audio_bytes


@pytest.mark.parametrize(
    ("language", "stream"),
    [
        pytest.param(None, False, id="no-language-unstreamed"),
        pytest.param("en", False, id="english-unstreamed"),
        pytest.param("en", True, id="english-streamed-export"),
    ],
)
def test_transcription_openai_round_trip(
    audio_samples: np.ndarray, language: LanguageAlpha2 | None, stream: bool
) -> None:
    raw_audio = audio_bytes(samples=audio_samples, fmt="wav")
    canonical_audio = base64.b64encode(raw_audio).decode("ascii")
    request = TranscriptionRequest(
        audio=canonical_audio,
        language=language,
        model="model",
        random_seed=43,
        id="internal-id",
        max_tokens=80,
        strict_audio_validation=False,
        streaming=StreamingMode.ONLINE,
        target_streaming_delay_ms=150,
    )

    exported = request.to_openai(stream=stream)
    expected_fields = {
        "temperature": 0.7,
        "top_p": 1.0,
        "model": "model",
        "language": language,
        "target_streaming_delay_ms": 150,
        "seed": 43,
        "stream": stream,
    }
    assert {key: value for key, value in exported.items() if key != "file"} == expected_fields
    assert_audio_buffer(buffer=exported["file"], raw_audio=raw_audio, fmt="wav", samples=audio_samples)

    expected_import = TranscriptionRequest(
        audio=canonical_audio,
        language=language,
        model="model",
        random_seed=43,
        target_streaming_delay_ms=150,
        streaming=StreamingMode.DISABLED,
    )
    assert TranscriptionRequest.from_openai(exported) == expected_import
    assert TranscriptionRequest.from_openai(TranscriptionCreateParamsBase(**exported)) == expected_import  # type: ignore[typeddict-item]


@pytest.mark.parametrize(
    ("fmt", "representation"),
    [
        pytest.param("wav", "base64", id="wav-base64"),
        pytest.param("flac", "base64", id="flac-base64"),
        pytest.param("wav", "bytes", id="wav-bytes"),
        pytest.param("flac", "bytes", id="flac-bytes"),
    ],
)
def test_transcription_export_preserves_audio_buffer_and_import_canonicalizes(
    audio_samples: np.ndarray, fmt: str, representation: str
) -> None:
    raw_audio = audio_bytes(samples=audio_samples, fmt=fmt)
    canonical_audio = base64.b64encode(raw_audio).decode("ascii")
    input_audio = canonical_audio if representation == "base64" else raw_audio
    request = TranscriptionRequest(audio=input_audio, model="model", language=None, target_streaming_delay_ms=None)

    exported = request.to_openai()

    assert {key: value for key, value in exported.items() if key != "file"} == {
        "temperature": 0.7,
        "top_p": 1.0,
        "model": "model",
        "language": None,
        "target_streaming_delay_ms": None,
        "seed": None,
    }
    assert_audio_buffer(buffer=exported["file"], raw_audio=raw_audio, fmt=fmt, samples=audio_samples)
    assert TranscriptionRequest.from_openai(exported) == TranscriptionRequest(
        audio=canonical_audio, model="model", language=None, target_streaming_delay_ms=None
    )


def test_transcription_export_rejects_invalid_audio_bytes() -> None:
    request = TranscriptionRequest(
        audio=b"not valid audio data", model="model", language=None, target_streaming_delay_ms=None
    )

    with pytest.raises(ValueError, match="Failed to detect audio format"):
        request.to_openai()

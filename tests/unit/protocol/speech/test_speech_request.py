import base64
import io
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.audio import Audio


@pytest.fixture
def audio_samples() -> np.ndarray:
    return np.tile(np.array([0.0, 0.25, -0.5, 0.75]), 100)


def _audio_bytes(samples: np.ndarray, fmt: str) -> bytes:
    buffer = io.BytesIO()
    sf.write(file=buffer, data=samples, samplerate=16000, format=fmt)
    return buffer.getvalue()


def _assert_audio_buffer(buffer: object, raw_audio: bytes, fmt: str, samples: np.ndarray) -> None:
    assert isinstance(buffer, io.BytesIO)
    assert buffer.name == f"audio.{fmt}"
    assert buffer.getvalue() == raw_audio
    decoded = Audio.from_bytes(buffer.getvalue())
    assert decoded.format == fmt
    assert decoded.sampling_rate == 16000
    np.testing.assert_allclose(decoded.audio_array, samples, atol=1e-3)


def test_speech_from_openai_filters_instructions_and_decodes_reference_audio(audio_samples: np.ndarray) -> None:
    raw_audio = _audio_bytes(samples=audio_samples, fmt="wav")
    incoming: dict[str, Any] = {
        "input": "Hello world",
        "model": "tts-1",
        "voice": "female",
        "ref_audio": io.BytesIO(raw_audio),
        "instructions": "Speak slowly",
    }
    imported = SpeechRequest.from_openai(incoming)

    assert imported == SpeechRequest(
        input="Hello world",
        model="tts-1",
        voice="female",
        ref_audio=base64.b64encode(raw_audio).decode("ascii"),
    )
    assert {key: value for key, value in incoming.items() if key != "ref_audio"} == {
        "input": "Hello world",
        "model": "tts-1",
        "voice": "female",
        "instructions": "Speak slowly",
    }
    assert isinstance(incoming["ref_audio"], io.BytesIO)
    assert incoming["ref_audio"].getvalue() == raw_audio
    assert isinstance(imported.ref_audio, str)
    decoded = Audio.from_base64(imported.ref_audio)
    assert decoded.format == "wav"
    assert decoded.sampling_rate == 16000
    np.testing.assert_allclose(decoded.audio_array, audio_samples, atol=1e-3)

    voice_object = {"input": "Hello", "voice": {"id": "custom-voice-123"}}
    assert SpeechRequest.from_openai(voice_object) == SpeechRequest(input="Hello", voice="custom-voice-123")
    assert voice_object == {"input": "Hello", "voice": {"id": "custom-voice-123"}}


@pytest.mark.parametrize(
    ("fmt", "representation"),
    [
        pytest.param("wav", "base64", id="wav-base64"),
        pytest.param("flac", "base64", id="flac-base64"),
        pytest.param("wav", "bytes", id="wav-bytes"),
        pytest.param("flac", "bytes", id="flac-bytes"),
    ],
)
def test_speech_reference_audio_export_and_canonical_import(
    audio_samples: np.ndarray, fmt: str, representation: str
) -> None:
    raw_audio = _audio_bytes(samples=audio_samples, fmt=fmt)
    canonical_audio = base64.b64encode(raw_audio).decode("ascii")
    source_audio = canonical_audio if representation == "base64" else raw_audio
    request = SpeechRequest(input="Hello world", ref_audio=source_audio)

    exported = request.to_openai()

    assert {key: value for key, value in exported.items() if key != "ref_audio"} == {
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": None,
        "id": None,
        "model": None,
        "input": "Hello world",
        "voice": None,
        "seed": None,
    }
    _assert_audio_buffer(buffer=exported["ref_audio"], raw_audio=raw_audio, fmt=fmt, samples=audio_samples)
    assert SpeechRequest.from_openai(exported) == SpeechRequest(input="Hello world", ref_audio=canonical_audio)


def test_speech_export_rejects_invalid_reference_audio_bytes() -> None:
    request = SpeechRequest(input="Hello world", ref_audio=b"not valid audio data")

    with pytest.raises(ValueError, match="Failed to detect audio format"):
        request.to_openai()


@pytest.mark.parametrize(
    ("voice", "with_ref_audio", "seed"),
    [
        pytest.param("female", True, None, id="voice-and-reference"),
        pytest.param("female", False, 1234, id="voice-and-positive-seed"),
        pytest.param(None, True, 0, id="reference-and-zero-seed"),
    ],
)
def test_speech_request_openai_round_trip(
    audio_samples: np.ndarray, voice: str | None, with_ref_audio: bool, seed: int | None
) -> None:
    raw_audio = _audio_bytes(samples=audio_samples, fmt="wav")
    canonical_audio = base64.b64encode(raw_audio).decode("ascii") if with_ref_audio else None
    original = SpeechRequest(
        input="Round trip test", ref_audio=canonical_audio, voice=voice, model="tts-1", random_seed=seed
    )

    exported = original.to_openai()

    assert {key: value for key, value in exported.items() if key != "ref_audio"} == {
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": None,
        "id": None,
        "model": "tts-1",
        "input": "Round trip test",
        "voice": voice,
        "seed": seed,
    }
    if with_ref_audio:
        _assert_audio_buffer(buffer=exported["ref_audio"], raw_audio=raw_audio, fmt="wav", samples=audio_samples)
    else:
        assert "ref_audio" not in exported
    assert SpeechRequest.from_openai(exported) == SpeechRequest(
        input="Round trip test", ref_audio=canonical_audio, voice=voice, model="tts-1", random_seed=seed
    )

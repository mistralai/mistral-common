import io
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.audio import Audio

from .test_tokenizer_v7_audio_tts import _make_fake_audio


def _audio_to_wav_bytes(audio: Audio) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio.audio_array, audio.sampling_rate, format="wav")
    return buffer.getvalue()


def test_convert_speech_request_from_openai() -> None:
    audio = _make_fake_audio(0.5)
    raw_bytes = _audio_to_wav_bytes(audio)
    openai_dict: dict[str, Any] = {
        "input": "Hello world",
        "model": "tts-1",
        "voice": "female",
        "ref_audio": io.BytesIO(raw_bytes),
        "instructions": "Speak slowly",  # OAI-only field, should be ignored
    }
    request = SpeechRequest.from_openai(openai_dict)

    assert request.input == "Hello world"
    assert request.model == "tts-1"
    assert request.voice == "female"
    assert isinstance(request.ref_audio, str)
    decoded_audio = Audio.from_base64(request.ref_audio)
    assert np.allclose(decoded_audio.audio_array, audio.audio_array, atol=1e-3)

    # Voice as dict with "id" (OAI format) should be normalized to string
    openai_dict_voice_obj: dict[str, Any] = {
        "input": "Hello",
        "voice": {"id": "custom-voice-123"},
    }
    request_voice = SpeechRequest.from_openai(openai_dict_voice_obj)
    assert request_voice.voice == "custom-voice-123"


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_speech_to_openai_base64_ref_audio_filename(fmt: str) -> None:
    audio = _make_fake_audio(0.5)
    request = SpeechRequest(input="Hello world", ref_audio=audio.to_base64(fmt))

    buffer = request.to_openai()["ref_audio"]

    assert isinstance(buffer, io.BytesIO)
    assert buffer.name == f"audio.{fmt}"

    recovered = Audio.from_bytes(buffer.getvalue())
    assert np.allclose(recovered.audio_array, audio.audio_array, atol=1e-3)


@pytest.mark.parametrize("fmt", ["wav", "flac"])
def test_speech_to_openai_bytes_ref_audio_filename(fmt: str) -> None:
    audio = _make_fake_audio(0.5)
    source = io.BytesIO()
    sf.write(source, audio.audio_array, audio.sampling_rate, format=fmt)
    request = SpeechRequest(input="Hello world", ref_audio=source.getvalue())

    buffer = request.to_openai()["ref_audio"]

    assert isinstance(buffer, io.BytesIO)
    assert buffer.name == f"audio.{fmt}"


def test_speech_to_openai_bytes_invalid_format() -> None:
    """Verify that invalid reference audio bytes raise a ValueError."""
    request = SpeechRequest(input="Hello world", ref_audio=b"not valid audio data")

    with pytest.raises(ValueError, match="Failed to detect audio format"):
        request.to_openai()


@pytest.mark.parametrize(
    ["voice", "with_ref_audio", "random_seed"],
    [
        ("female", True, None),
        ("female", False, 1234),
        (None, True, 0),
    ],
)
def test_convert_speech_request_round_trip(voice: str | None, with_ref_audio: bool, random_seed: int | None) -> None:
    audio = _make_fake_audio(0.5)
    original = SpeechRequest(
        input="Round trip test",
        ref_audio=audio.to_base64("wav") if with_ref_audio else None,
        voice=voice,
        model="tts-1",
        random_seed=random_seed,
    )

    openai_dict = original.to_openai()

    # OpenAI uses "seed"; mistral-common uses "random_seed" (parity with TranscriptionRequest).
    assert "random_seed" not in openai_dict
    assert openai_dict["seed"] == random_seed

    if with_ref_audio:
        assert isinstance(openai_dict["ref_audio"], io.BytesIO)
    else:
        assert "ref_audio" not in openai_dict

    restored = SpeechRequest.from_openai(openai_dict)
    # ref_audio is re-encoded through the OpenAI buffer, so it is not byte-identical; compare it separately.
    assert original.model_dump(exclude={"ref_audio"}) == restored.model_dump(exclude={"ref_audio"})

    if with_ref_audio:
        assert isinstance(original.ref_audio, str)
        assert isinstance(restored.ref_audio, str)
        original_audio = Audio.from_base64(original.ref_audio)
        restored_audio = Audio.from_base64(restored.ref_audio)
        assert np.allclose(restored_audio.audio_array, original_audio.audio_array, atol=1e-3)
    else:
        assert restored.ref_audio is None

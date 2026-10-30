from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mistral_common.protocol.instruct.chunk import AudioURLChunk
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)

_AUDIO_FILENAMES = [
    "clip one.wav",
    "clip%20one.wav",
    "café.wav",
    "clip+one.wav",
    "clip.wav",
]


def _write_audio_file(path: Path) -> None:
    """Write a finite mono WAV fixture to a path."""
    samples = np.linspace(start=-0.5, stop=0.5, num=800, dtype=np.float32)
    sf.write(file=path, data=samples, samplerate=16_000)


def _audio_encoder() -> AudioEncoder:
    """Build a local audio encoder without model or network dependencies."""
    return AudioEncoder(
        audio_config=AudioConfig(
            sampling_rate=16_000,
            frame_rate=100.0,
            encoding_config=AudioSpectrogramConfig(
                num_mel_bins=80,
                hop_length=160,
                window_size=400,
            ),
        ),
        special_ids=SpecialAudioIDs(
            audio=1,
            begin_audio=2,
            streaming_pad=None,
            text_to_audio=None,
            audio_to_text=None,
        ),
    )


@pytest.mark.parametrize(argnames="filename", argvalues=_AUDIO_FILENAMES)
def test_audio_from_file_uri_matches_plain_path(tmp_path: Path, filename: str) -> None:
    """Decode a file URI once while preserving plain local paths."""
    path = tmp_path / filename
    _write_audio_file(path=path)

    plain_audio = Audio.from_file(file=str(path))
    uri_audio = Audio.from_file(file=path.as_uri())

    assert uri_audio.sampling_rate == plain_audio.sampling_rate
    np.testing.assert_allclose(actual=uri_audio.audio_array, desired=plain_audio.audio_array)


def test_plain_path_with_literal_percent_is_not_decoded(tmp_path: Path) -> None:
    """A plain path containing %20 keeps the literal filename."""
    path = tmp_path / "clip%20one.wav"
    _write_audio_file(path=path)

    audio = Audio.from_file(file=str(path))

    assert audio.sampling_rate == 16_000


def test_audio_url_chunk_file_uri_loads_through_audio_encoder(tmp_path: Path) -> None:
    """Load a percent-encoded file URI through the AudioURLChunk workflow."""
    path = tmp_path / "clip one% café.wav"
    _write_audio_file(path=path)

    plain_audio = Audio.from_file(file=str(path))
    encoded_audio = _audio_encoder()(content=AudioURLChunk(audio_url=path.as_uri())).audio

    assert encoded_audio is not None
    assert encoded_audio.sampling_rate == plain_audio.sampling_rate
    np.testing.assert_allclose(actual=encoded_audio.audio_array, desired=plain_audio.audio_array)


def test_missing_audio_file_uri_raises_file_not_found(tmp_path: Path) -> None:
    """Raise FileNotFoundError when a decoded file URI has no target."""
    missing_path = tmp_path / "missing clip.wav"

    with pytest.raises(FileNotFoundError):
        Audio.from_file(file=missing_path.as_uri())

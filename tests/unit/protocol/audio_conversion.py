import io

import numpy as np
import soundfile as sf

from mistral_common.tokens.tokenizers.audio import Audio


def audio_bytes(samples: np.ndarray, fmt: str) -> bytes:
    r"""Encode conversion test samples independently of the request converter."""
    buffer = io.BytesIO()
    sf.write(file=buffer, data=samples, samplerate=16000, format=fmt)
    return buffer.getvalue()


def assert_audio_buffer(buffer: object, raw_audio: bytes, fmt: str, samples: np.ndarray) -> None:
    r"""Check the full named payload and decoded values in an OpenAI export."""
    assert isinstance(buffer, io.BytesIO)
    assert buffer.name == f"audio.{fmt}"
    assert buffer.getvalue() == raw_audio
    decoded = Audio.from_bytes(buffer.getvalue())
    assert decoded.format == fmt
    assert decoded.sampling_rate == 16000
    np.testing.assert_allclose(decoded.audio_array, samples, atol=1e-3)

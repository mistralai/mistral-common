r"""Python-authored transcription recipes for public workflow cases.

Each recipe constructs a fresh request on every call, so cases never share
mutable request state. Recipes are reusable across cases and never carry
expectations; the case registry binds a recipe to one tokenizer configuration
and one expected outcome.
"""

import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest


@dataclass(frozen=True)
class TranscriptionRecipe:
    """One Python-authored transcription input recipe identified by a stable recipe id."""

    recipe_id: str
    build: Callable[[], TranscriptionRequest]


def multichannel_wav_bytes() -> bytes:
    r"""Build a fresh two-channel 24 kHz float32 WAV of 2,400 zero frames."""
    buffer = io.BytesIO()
    sf.write(file=buffer, data=np.zeros(shape=(2_400, 2), dtype=np.float32), samplerate=24_000, format="WAV")
    return buffer.getvalue()


def _transcription_recipe(recipe_id: str, *, audio: str | bytes, streaming: StreamingMode) -> TranscriptionRecipe:
    def build() -> TranscriptionRequest:
        return TranscriptionRequest(audio=audio, streaming=streaming, language=None, target_streaming_delay_ms=None)

    return TranscriptionRecipe(recipe_id=recipe_id, build=build)


TRANSCRIPTION_INVALID_BASE64 = _transcription_recipe(
    "transcription-invalid-base64", audio="not base64!", streaming=StreamingMode.DISABLED
)
TRANSCRIPTION_NON_AUDIO_BYTES = _transcription_recipe(
    "transcription-non-audio-bytes", audio=b"not audio", streaming=StreamingMode.DISABLED
)
TRANSCRIPTION_MULTICHANNEL = _transcription_recipe(
    "transcription-multichannel", audio=multichannel_wav_bytes(), streaming=StreamingMode.DISABLED
)
TRANSCRIPTION_EMPTY_BYTES = _transcription_recipe(
    "transcription-empty-bytes", audio=b"", streaming=StreamingMode.DISABLED
)
TRANSCRIPTION_OFFLINE_INVALID_BASE64 = _transcription_recipe(
    "transcription-offline-invalid-base64", audio="not base64!", streaming=StreamingMode.OFFLINE
)
TRANSCRIPTION_OFFLINE_NON_AUDIO_BYTES = _transcription_recipe(
    "transcription-offline-non-audio-bytes", audio=b"not audio", streaming=StreamingMode.OFFLINE
)
TRANSCRIPTION_OFFLINE_MULTICHANNEL = _transcription_recipe(
    "transcription-offline-multichannel", audio=multichannel_wav_bytes(), streaming=StreamingMode.OFFLINE
)

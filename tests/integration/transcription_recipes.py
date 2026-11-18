r"""Python-authored transcription recipes for public workflow cases.

Each recipe constructs a fresh request on every call, so cases never share
mutable request state. Recipes are reusable across cases and never carry
expectations; the case registry binds a recipe to one tokenizer configuration
and one expected outcome.
"""

import base64
import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from pydantic_extra_types.language_code import LanguageAlpha2

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


def mono_wav_base64(*, sampling_rate: int, num_samples: int, seed: int = 0) -> str:
    r"""Build a fresh deterministic mono float32 WAV and return it base64-encoded.

    Args:
        sampling_rate: The WAV sampling rate in Hz.
        num_samples: The number of samples.
        seed: Seed of the deterministic uniform waveform, so each recipe
            builds the same non-silent input on every call.

    Returns:
        The base64-encoded WAV text.
    """
    rng = np.random.default_rng(seed)
    buffer = io.BytesIO()
    sf.write(
        file=buffer,
        data=rng.uniform(low=-0.5, high=0.5, size=num_samples).astype(np.float32),
        samplerate=sampling_rate,
        format="WAV",
    )
    return base64.b64encode(buffer.getvalue()).decode()


def _transcription_recipe(
    recipe_id: str,
    *,
    audio: str | bytes,
    streaming: StreamingMode,
    language: LanguageAlpha2 | None = None,
    target_streaming_delay_ms: int | None = None,
) -> TranscriptionRecipe:
    def build() -> TranscriptionRequest:
        return TranscriptionRequest(
            audio=audio,
            streaming=streaming,
            language=language,
            target_streaming_delay_ms=target_streaming_delay_ms,
        )

    return TranscriptionRecipe(recipe_id=recipe_id, build=build)


def _transcription_audio_recipe(
    recipe_id: str,
    *,
    streaming: StreamingMode,
    language: LanguageAlpha2 | None = None,
    target_streaming_delay_ms: int | None = None,
) -> TranscriptionRecipe:
    def build() -> TranscriptionRequest:
        return TranscriptionRequest(
            audio=mono_wav_base64(sampling_rate=24_000, num_samples=2_400),
            streaming=streaming,
            language=language,
            target_streaming_delay_ms=target_streaming_delay_ms,
        )

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

# Success recipes: each build returns a fresh request with freshly serialized
# deterministic audio, so cases never share mutable request or audio state.
TRANSCRIPTION_INSTRUCT_NO_LANGUAGE = _transcription_audio_recipe(
    "transcription-instruct-no-language", streaming=StreamingMode.DISABLED
)
TRANSCRIPTION_INSTRUCT_EN = _transcription_audio_recipe(
    "transcription-instruct-en", streaming=StreamingMode.DISABLED, language=LanguageAlpha2("en")
)
TRANSCRIPTION_REALTIME_OFFLINE_DEFAULT_DELAY = _transcription_audio_recipe(
    "transcription-realtime-offline-default-delay", streaming=StreamingMode.OFFLINE
)
TRANSCRIPTION_REALTIME_OFFLINE_DELAY_0MS = _transcription_audio_recipe(
    "transcription-realtime-offline-delay-0ms", streaming=StreamingMode.OFFLINE, target_streaming_delay_ms=0
)
TRANSCRIPTION_REALTIME_ONLINE_EMPTY_PREFILL_DEFAULT_DELAY = _transcription_recipe(
    "transcription-realtime-online-empty-prefill-default-delay", audio="", streaming=StreamingMode.ONLINE
)

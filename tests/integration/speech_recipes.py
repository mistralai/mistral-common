r"""Python-authored speech recipes for public workflow cases.

Each recipe constructs a fresh request on every call, so cases never share
mutable request state. Recipes are reusable across cases and never carry
expectations; the case registry binds a recipe to one tokenizer configuration
and one expected outcome.
"""

from collections.abc import Callable
from dataclasses import dataclass

from mistral_common.protocol.speech.request import SpeechRequest
from tests.integration.transcription_recipes import mono_wav_base64, multichannel_wav_bytes


@dataclass(frozen=True)
class SpeechRecipe:
    """One Python-authored speech input recipe identified by a stable recipe id."""

    recipe_id: str
    build: Callable[[], SpeechRequest]


def _speech_recipe(recipe_id: str, *, ref_audio: str | bytes) -> SpeechRecipe:
    def build() -> SpeechRequest:
        return SpeechRequest(input="hello", ref_audio=ref_audio)

    return SpeechRecipe(recipe_id=recipe_id, build=build)


def _speech_reference_recipe(recipe_id: str, *, voice: str | None = None) -> SpeechRecipe:
    def build() -> SpeechRequest:
        return SpeechRequest(
            input="hello",
            ref_audio=mono_wav_base64(sampling_rate=22_050, num_samples=2_205, seed=1),
            voice=voice,
        )

    return SpeechRecipe(recipe_id=recipe_id, build=build)


SPEECH_INVALID_BASE64 = _speech_recipe("speech-invalid-base64", ref_audio="not base64!")
SPEECH_NON_AUDIO_BYTES = _speech_recipe("speech-non-audio-bytes", ref_audio=b"not audio")
SPEECH_MULTICHANNEL = _speech_recipe("speech-multichannel", ref_audio=multichannel_wav_bytes())

# Success recipes: each build returns a fresh request with freshly serialized
# deterministic 22050-Hz reference audio, so the synthetic 24-kHz profile's
# resampling is exercised on every call.
SPEECH_REFERENCE_AUDIO = _speech_reference_recipe("speech-reference-audio")
SPEECH_SYNTHETIC_PRESET = SpeechRecipe(
    recipe_id="speech-synthetic-preset",
    build=lambda: SpeechRequest(input="hello", voice="preset"),
)
SPEECH_REFERENCE_OVER_PRESET = _speech_reference_recipe("speech-reference-over-preset", voice="preset")

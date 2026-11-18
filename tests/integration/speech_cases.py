r"""Public speech cases for the synthetic v7 speech profile.

One executable case pairs one reusable Python recipe with one identified
tokenizer configuration and exactly one expected outcome: a reviewed success
manifest for success cases, or one expected raw decode error for error cases.
The requests are constructed before the public call, so a construction failure
cannot satisfy an encode-rejection case. Error cases have no success manifest.

All success cases bind the explicitly synthetic ``synthetic-v7-speech-test``
configuration: the verified pinned v7 transcription profiles lack the speech
markers, and the only released speech artifact is excluded by its CC-BY-NC
license. Synthetic preset mappings establish no released voice fidelity.
"""

from dataclasses import dataclass

from mistral_common.exceptions import AudioDecodeException
from tests.integration.speech_recipes import (
    SPEECH_INVALID_BASE64,
    SPEECH_MULTICHANNEL,
    SPEECH_NON_AUDIO_BYTES,
    SPEECH_REFERENCE_AUDIO,
    SPEECH_REFERENCE_OVER_PRESET,
    SPEECH_SYNTHETIC_PRESET,
    SpeechRecipe,
)
from tests.integration.tokenizer_configurations import (
    SYNTHETIC_V7_SPEECH_AUDIO_TEST,
    SYNTHETIC_V7_SPEECH_NO_MARKER_AUDIO_TEST,
    TokenizerConfiguration,
)


@dataclass(frozen=True)
class PublicSpeechSuccessCase:
    """One executable public speech success case identified by its semantic id."""

    case_id: str
    recipe: SpeechRecipe
    configuration: TokenizerConfiguration


@dataclass(frozen=True)
class PublicSpeechErrorCase:
    """One executable public speech rejection case identified by its semantic id."""

    case_id: str
    recipe: SpeechRecipe
    configuration: TokenizerConfiguration
    expected_exception: type[Exception]
    message_pattern: str


SPEECH_SUCCESS_CASES: tuple[PublicSpeechSuccessCase, ...] = (
    PublicSpeechSuccessCase(
        case_id="speech-synthetic-v7-reference-audio",
        recipe=SPEECH_REFERENCE_AUDIO,
        configuration=SYNTHETIC_V7_SPEECH_AUDIO_TEST,
    ),
    PublicSpeechSuccessCase(
        case_id="speech-synthetic-v7-preset",
        recipe=SPEECH_SYNTHETIC_PRESET,
        configuration=SYNTHETIC_V7_SPEECH_AUDIO_TEST,
    ),
    PublicSpeechSuccessCase(
        case_id="speech-synthetic-v7-reference-over-preset",
        recipe=SPEECH_REFERENCE_OVER_PRESET,
        configuration=SYNTHETIC_V7_SPEECH_AUDIO_TEST,
    ),
)


SPEECH_ERROR_CASES: tuple[PublicSpeechErrorCase, ...] = (
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-invalid-base64-rejected-test",
        recipe=SPEECH_INVALID_BASE64,
        configuration=SYNTHETIC_V7_SPEECH_AUDIO_TEST,
        expected_exception=AudioDecodeException,
        message_pattern=r"^Could not decode the audio",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-non-audio-bytes-rejected-test",
        recipe=SPEECH_NON_AUDIO_BYTES,
        configuration=SYNTHETIC_V7_SPEECH_AUDIO_TEST,
        expected_exception=AudioDecodeException,
        message_pattern=r"^Could not decode the audio",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-multichannel-rejected-test",
        recipe=SPEECH_MULTICHANNEL,
        configuration=SYNTHETIC_V7_SPEECH_AUDIO_TEST,
        expected_exception=AudioDecodeException,
        message_pattern=r"^Could not decode the audio",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-decoding-precedes-marker-test",
        recipe=SPEECH_INVALID_BASE64,
        configuration=SYNTHETIC_V7_SPEECH_NO_MARKER_AUDIO_TEST,
        expected_exception=AudioDecodeException,
        message_pattern=r"^Could not decode the audio",
    ),
)

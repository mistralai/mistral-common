r"""Public speech decode-error cases for synthetic v7 audio profiles.

One executable case pairs one reusable Python recipe with one identified
tokenizer configuration and exactly one expected raw decode error. The
requests are constructed before the public call, so a construction failure
cannot satisfy an encode-rejection case. All cases are errors and have no
success manifest.
"""

from dataclasses import dataclass

import soundfile as sf

from tests.integration.speech_recipes import (
    SPEECH_INVALID_BASE64,
    SPEECH_MULTICHANNEL,
    SPEECH_NON_AUDIO_BYTES,
    SpeechRecipe,
)
from tests.integration.tokenizer_configurations import (
    SYNTHETIC_V7_SPEECH_AUDIO_TEST,
    SYNTHETIC_V7_SPEECH_NO_MARKER_AUDIO_TEST,
    TokenizerConfiguration,
)


@dataclass(frozen=True)
class PublicSpeechErrorCase:
    """One executable public speech rejection case identified by its semantic id."""

    case_id: str
    recipe: SpeechRecipe
    configuration: TokenizerConfiguration
    expected_exception: type[Exception]
    message_pattern: str


SPEECH_ERROR_CASES: tuple[PublicSpeechErrorCase, ...] = (
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-invalid-base64-rejected-test",
        recipe=SPEECH_INVALID_BASE64,
        configuration=SYNTHETIC_V7_SPEECH_AUDIO_TEST,
        expected_exception=ValueError,
        message_pattern=r"^base64 decoding failed",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-non-audio-bytes-rejected-test",
        recipe=SPEECH_NON_AUDIO_BYTES,
        configuration=SYNTHETIC_V7_SPEECH_AUDIO_TEST,
        expected_exception=sf.LibsndfileError,
        message_pattern=r"Format not recognised",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-multichannel-rejected-test",
        recipe=SPEECH_MULTICHANNEL,
        configuration=SYNTHETIC_V7_SPEECH_AUDIO_TEST,
        expected_exception=ValueError,
        message_pattern=r"^audio_array\.ndim=2$",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-decoding-precedes-marker-test",
        recipe=SPEECH_INVALID_BASE64,
        configuration=SYNTHETIC_V7_SPEECH_NO_MARKER_AUDIO_TEST,
        expected_exception=ValueError,
        message_pattern=r"^base64 decoding failed",
    ),
)

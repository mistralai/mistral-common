r"""Public transcription decode-error cases for synthetic v7 audio profiles.

One executable case pairs one reusable Python recipe with one identified
tokenizer configuration and exactly one expected raw decode error. The
requests are constructed before the public call, so a construction failure
cannot satisfy an encode-rejection case. All cases are errors and have no
success manifest.
"""

from dataclasses import dataclass

import soundfile as sf
from pydantic import ValidationError

from mistral_common.exceptions import UnsupportedTokenizerFeatureException
from tests.integration.tokenizer_configurations import (
    SYNTHETIC_V7_INSTRUCT_AUDIO_TEST,
    SYNTHETIC_V7_STREAMING_AUDIO_TEST,
    SYNTHETIC_V7_STREAMING_NO_PAD_AUDIO_TEST,
    TokenizerConfiguration,
)
from tests.integration.transcription_recipes import (
    TRANSCRIPTION_EMPTY_BYTES,
    TRANSCRIPTION_INVALID_BASE64,
    TRANSCRIPTION_MULTICHANNEL,
    TRANSCRIPTION_NON_AUDIO_BYTES,
    TRANSCRIPTION_OFFLINE_INVALID_BASE64,
    TRANSCRIPTION_OFFLINE_MULTICHANNEL,
    TRANSCRIPTION_OFFLINE_NON_AUDIO_BYTES,
    TranscriptionRecipe,
)


@dataclass(frozen=True)
class PublicTranscriptionErrorCase:
    """One executable public transcription rejection case identified by its semantic id."""

    case_id: str
    recipe: TranscriptionRecipe
    configuration: TokenizerConfiguration
    expected_exception: type[Exception]
    message_pattern: str


TRANSCRIPTION_ERROR_CASES: tuple[PublicTranscriptionErrorCase, ...] = (
    PublicTranscriptionErrorCase(
        case_id="audio-v7-instruct-invalid-base64-rejected-test",
        recipe=TRANSCRIPTION_INVALID_BASE64,
        configuration=SYNTHETIC_V7_INSTRUCT_AUDIO_TEST,
        expected_exception=ValueError,
        message_pattern=r"^base64 decoding failed",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-instruct-non-audio-bytes-rejected-test",
        recipe=TRANSCRIPTION_NON_AUDIO_BYTES,
        configuration=SYNTHETIC_V7_INSTRUCT_AUDIO_TEST,
        expected_exception=sf.LibsndfileError,
        message_pattern=r"Format not recognised",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-instruct-multichannel-rejected-test",
        recipe=TRANSCRIPTION_MULTICHANNEL,
        configuration=SYNTHETIC_V7_INSTRUCT_AUDIO_TEST,
        expected_exception=ValueError,
        message_pattern=r"^audio_array\.ndim=2$",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-instruct-empty-audio-rejected-test",
        recipe=TRANSCRIPTION_EMPTY_BYTES,
        configuration=SYNTHETIC_V7_INSTRUCT_AUDIO_TEST,
        expected_exception=ValidationError,
        message_pattern=r"`input_audio` should not be empty",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-offline-invalid-base64-rejected-test",
        recipe=TRANSCRIPTION_OFFLINE_INVALID_BASE64,
        configuration=SYNTHETIC_V7_STREAMING_AUDIO_TEST,
        expected_exception=ValueError,
        message_pattern=r"^base64 decoding failed",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-offline-non-audio-bytes-rejected-test",
        recipe=TRANSCRIPTION_OFFLINE_NON_AUDIO_BYTES,
        configuration=SYNTHETIC_V7_STREAMING_AUDIO_TEST,
        expected_exception=sf.LibsndfileError,
        message_pattern=r"Format not recognised",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-offline-multichannel-rejected-test",
        recipe=TRANSCRIPTION_OFFLINE_MULTICHANNEL,
        configuration=SYNTHETIC_V7_STREAMING_AUDIO_TEST,
        expected_exception=ValueError,
        message_pattern=r"^audio_array\.ndim=2$",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-offline-streaming-pad-precedes-decoding-test",
        recipe=TRANSCRIPTION_OFFLINE_NON_AUDIO_BYTES,
        configuration=SYNTHETIC_V7_STREAMING_NO_PAD_AUDIO_TEST,
        expected_exception=UnsupportedTokenizerFeatureException,
        message_pattern=r"^Audio encoder does not provide the streaming_pad marker required for transcription\.$",
    ),
)

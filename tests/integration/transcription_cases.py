r"""Public transcription cases for pinned and synthetic v7 audio profiles.

One executable case pairs one reusable Python recipe with one identified
tokenizer configuration and exactly one expected outcome: a reviewed success
manifest for success cases, or one expected raw decode error for error cases.
The requests are constructed before the public call, so a construction failure
cannot satisfy an encode-rejection case. Error cases have no success manifest.
"""

from dataclasses import dataclass

from pydantic import ValidationError

from mistral_common.exceptions import AudioDecodeException, UnsupportedTokenizerFeatureException
from tests.integration.tokenizer_configurations import (
    PINNED_V7_AUDIO_TEST,
    PINNED_V7_REALTIME_TEST,
    SYNTHETIC_V7_INSTRUCT_AUDIO_TEST,
    SYNTHETIC_V7_STREAMING_AUDIO_TEST,
    SYNTHETIC_V7_STREAMING_NO_PAD_AUDIO_TEST,
    TokenizerConfiguration,
)
from tests.integration.transcription_recipes import (
    TRANSCRIPTION_EMPTY_BYTES,
    TRANSCRIPTION_INSTRUCT_EN,
    TRANSCRIPTION_INSTRUCT_NO_LANGUAGE,
    TRANSCRIPTION_INVALID_BASE64,
    TRANSCRIPTION_MULTICHANNEL,
    TRANSCRIPTION_NON_AUDIO_BYTES,
    TRANSCRIPTION_OFFLINE_INVALID_BASE64,
    TRANSCRIPTION_OFFLINE_MULTICHANNEL,
    TRANSCRIPTION_OFFLINE_NON_AUDIO_BYTES,
    TRANSCRIPTION_REALTIME_OFFLINE_DEFAULT_DELAY,
    TRANSCRIPTION_REALTIME_OFFLINE_DELAY_0MS,
    TRANSCRIPTION_REALTIME_ONLINE_EMPTY_PREFILL_DEFAULT_DELAY,
    TranscriptionRecipe,
)


@dataclass(frozen=True)
class PublicTranscriptionSuccessCase:
    """One executable public transcription success case identified by its semantic id."""

    case_id: str
    recipe: TranscriptionRecipe
    configuration: TokenizerConfiguration


@dataclass(frozen=True)
class PublicTranscriptionErrorCase:
    """One executable public transcription rejection case identified by its semantic id."""

    case_id: str
    recipe: TranscriptionRecipe
    configuration: TokenizerConfiguration
    expected_exception: type[Exception]
    message_pattern: str


TRANSCRIPTION_SUCCESS_CASES: tuple[PublicTranscriptionSuccessCase, ...] = (
    PublicTranscriptionSuccessCase(
        case_id="transcription-pinned-v7-audio-instruct-no-language",
        recipe=TRANSCRIPTION_INSTRUCT_NO_LANGUAGE,
        configuration=PINNED_V7_AUDIO_TEST,
    ),
    PublicTranscriptionSuccessCase(
        case_id="transcription-pinned-v7-audio-instruct-en",
        recipe=TRANSCRIPTION_INSTRUCT_EN,
        configuration=PINNED_V7_AUDIO_TEST,
    ),
    PublicTranscriptionSuccessCase(
        case_id="transcription-pinned-v7-realtime-offline-default-delay",
        recipe=TRANSCRIPTION_REALTIME_OFFLINE_DEFAULT_DELAY,
        configuration=PINNED_V7_REALTIME_TEST,
    ),
    PublicTranscriptionSuccessCase(
        case_id="transcription-pinned-v7-realtime-offline-delay-0ms",
        recipe=TRANSCRIPTION_REALTIME_OFFLINE_DELAY_0MS,
        configuration=PINNED_V7_REALTIME_TEST,
    ),
    PublicTranscriptionSuccessCase(
        case_id="transcription-pinned-v7-realtime-online-empty-prefill-default-delay",
        recipe=TRANSCRIPTION_REALTIME_ONLINE_EMPTY_PREFILL_DEFAULT_DELAY,
        configuration=PINNED_V7_REALTIME_TEST,
    ),
)


TRANSCRIPTION_ERROR_CASES: tuple[PublicTranscriptionErrorCase, ...] = (
    PublicTranscriptionErrorCase(
        case_id="audio-v7-instruct-invalid-base64-rejected-test",
        recipe=TRANSCRIPTION_INVALID_BASE64,
        configuration=SYNTHETIC_V7_INSTRUCT_AUDIO_TEST,
        expected_exception=AudioDecodeException,
        message_pattern=r"^Could not decode the audio",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-instruct-non-audio-bytes-rejected-test",
        recipe=TRANSCRIPTION_NON_AUDIO_BYTES,
        configuration=SYNTHETIC_V7_INSTRUCT_AUDIO_TEST,
        expected_exception=AudioDecodeException,
        message_pattern=r"^Could not decode the audio",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-instruct-multichannel-rejected-test",
        recipe=TRANSCRIPTION_MULTICHANNEL,
        configuration=SYNTHETIC_V7_INSTRUCT_AUDIO_TEST,
        expected_exception=AudioDecodeException,
        message_pattern=r"^Could not decode the audio",
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
        expected_exception=AudioDecodeException,
        message_pattern=r"^Could not decode the audio",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-offline-non-audio-bytes-rejected-test",
        recipe=TRANSCRIPTION_OFFLINE_NON_AUDIO_BYTES,
        configuration=SYNTHETIC_V7_STREAMING_AUDIO_TEST,
        expected_exception=AudioDecodeException,
        message_pattern=r"^Could not decode the audio",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-offline-multichannel-rejected-test",
        recipe=TRANSCRIPTION_OFFLINE_MULTICHANNEL,
        configuration=SYNTHETIC_V7_STREAMING_AUDIO_TEST,
        expected_exception=AudioDecodeException,
        message_pattern=r"^Could not decode the audio",
    ),
    PublicTranscriptionErrorCase(
        case_id="audio-v7-offline-streaming-pad-precedes-decoding-test",
        recipe=TRANSCRIPTION_OFFLINE_NON_AUDIO_BYTES,
        configuration=SYNTHETIC_V7_STREAMING_NO_PAD_AUDIO_TEST,
        expected_exception=UnsupportedTokenizerFeatureException,
        message_pattern=r"^Audio encoder does not provide the streaming_pad marker required for transcription\.$",
    ),
)

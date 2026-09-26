from dataclasses import dataclass
from typing import Literal

import pytest

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
from tests.audio_error_test_support import (
    SYNTHETIC_V7_INSTRUCT,
    SYNTHETIC_V7_SPEECH,
    SYNTHETIC_V7_STREAMING,
    SyntheticV7AudioProfile,
    build_synthetic_v7_audio_tokenizer,
    valid_reference_audio,
    valid_reference_audio_bytes,
)


@dataclass(frozen=True)
class PublicInvalidAudioRequestCase:
    case_id: str
    configuration_id: str
    profile: SyntheticV7AudioProfile
    request_recipe: Literal[
        "instruct-offline",
        "instruct-online",
        "streaming-disabled",
        "speech-no-source",
        "speech-unknown-voice",
        "online-bytes",
    ]
    message: str


PUBLIC_INVALID_AUDIO_REQUEST_CASES = (
    PublicInvalidAudioRequestCase(
        case_id="audio-v7-instruct-offline-rejected-test",
        configuration_id="synthetic-v7-instruct-test",
        profile=SYNTHETIC_V7_INSTRUCT,
        request_recipe="instruct-offline",
        message=r"Instruct transcription.*DISABLED.*OFFLINE",
    ),
    PublicInvalidAudioRequestCase(
        case_id="audio-v7-instruct-online-rejected-test",
        configuration_id="synthetic-v7-instruct-test",
        profile=SYNTHETIC_V7_INSTRUCT,
        request_recipe="instruct-online",
        message=r"Instruct transcription.*DISABLED.*ONLINE",
    ),
    PublicInvalidAudioRequestCase(
        case_id="audio-v7-streaming-disabled-rejected-test",
        configuration_id="synthetic-v7-streaming-test",
        profile=SYNTHETIC_V7_STREAMING,
        request_recipe="streaming-disabled",
        message=r"Streaming transcription.*OFFLINE.*ONLINE.*DISABLED",
    ),
    PublicInvalidAudioRequestCase(
        case_id="audio-v7-speech-source-required-test",
        configuration_id="synthetic-v7-speech-test",
        profile=SYNTHETIC_V7_SPEECH,
        request_recipe="speech-no-source",
        message=r"Either ref_audio or voice must be defined",
    ),
    PublicInvalidAudioRequestCase(
        case_id="audio-v7-speech-unknown-voice-test",
        configuration_id="synthetic-v7-speech-test",
        profile=SYNTHETIC_V7_SPEECH,
        request_recipe="speech-unknown-voice",
        message=r"Unknown voice.*expected one of \['preset'\]",
    ),
    PublicInvalidAudioRequestCase(
        case_id="audio-v7-online-bytes-rejected-test",
        configuration_id="synthetic-v7-streaming-test",
        profile=SYNTHETIC_V7_STREAMING,
        request_recipe="online-bytes",
        message=r"ONLINE streaming audio.*base64 text.*bytes",
    ),
)


def _build_request(
    recipe: Literal[
        "instruct-offline",
        "instruct-online",
        "streaming-disabled",
        "speech-no-source",
        "speech-unknown-voice",
        "online-bytes",
    ],
) -> TranscriptionRequest | SpeechRequest:
    if recipe == "instruct-offline":
        return TranscriptionRequest(
            audio=valid_reference_audio(),
            streaming=StreamingMode.OFFLINE,
            language=None,
            target_streaming_delay_ms=None,
        )
    if recipe == "instruct-online":
        return TranscriptionRequest(
            audio=valid_reference_audio(),
            streaming=StreamingMode.ONLINE,
            language=None,
            target_streaming_delay_ms=None,
        )
    if recipe == "streaming-disabled":
        return TranscriptionRequest(
            audio=valid_reference_audio(),
            streaming=StreamingMode.DISABLED,
            language=None,
            target_streaming_delay_ms=None,
        )
    if recipe == "speech-no-source":
        return SpeechRequest(input="hello")
    if recipe == "speech-unknown-voice":
        return SpeechRequest(input="hello", voice="not-configured")
    return TranscriptionRequest(
        audio=valid_reference_audio_bytes(),
        streaming=StreamingMode.ONLINE,
        language=None,
        target_streaming_delay_ms=None,
    )


@pytest.mark.parametrize("case", PUBLIC_INVALID_AUDIO_REQUEST_CASES, ids=lambda case: case.case_id)
def test_public_invalid_audio_request(case: PublicInvalidAudioRequestCase) -> None:
    assert case.configuration_id == case.profile.configuration_id
    mode = ValidationMode.test
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=case.profile, mode=mode)
    assert tokenizer.mode == mode

    request = _build_request(case.request_recipe)
    with pytest.raises(InvalidRequestException, match=case.message):
        if isinstance(request, TranscriptionRequest):
            tokenizer.encode_transcription(request)
        else:
            tokenizer.encode_speech_request(request)

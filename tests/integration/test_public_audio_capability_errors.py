from dataclasses import dataclass
from typing import Literal

import pytest

from mistral_common.exceptions import UnsupportedTokenizerFeatureException
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
from tests.audio_error_test_support import (
    BUNDLED_V7_NO_AUDIO_CONFIGURATION_ID,
    SYNTHETIC_V7_INSTRUCT_NO_AUDIO,
    SYNTHETIC_V7_INSTRUCT_NO_BEGIN_AUDIO,
    SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE,
    SYNTHETIC_V7_SPEECH_NO_AUDIO,
    SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO,
    SYNTHETIC_V7_SPEECH_NO_MARKER,
    SYNTHETIC_V7_SPEECH_NO_VOICE_MAP,
    SYNTHETIC_V7_STREAMING_NO_PAD,
    SyntheticV7AudioProfile,
    build_synthetic_v7_audio_tokenizer,
    load_bundled_v7_no_audio_tokenizer,
    valid_reference_audio,
)


@dataclass(frozen=True)
class PublicAudioErrorCase:
    case_id: str
    configuration_id: str
    mode: ValidationMode
    profile: SyntheticV7AudioProfile | None
    operation: Literal["transcription", "speech"]
    request_recipe: Literal[
        "transcription",
        "transcription-audio",
        "streaming-online",
        "speech-no-source",
        "speech-reference",
        "speech-preset",
    ]
    message: str


PUBLIC_AUDIO_ERROR_CASES = (
    PublicAudioErrorCase(
        case_id="audio-v7-transcription-no-encoder-test",
        configuration_id=BUNDLED_V7_NO_AUDIO_CONFIGURATION_ID,
        mode=ValidationMode.test,
        profile=None,
        operation="transcription",
        request_recipe="transcription",
        message=r"audio encoder.*transcription",
    ),
    PublicAudioErrorCase(
        case_id="audio-v7-speech-no-encoder-test",
        configuration_id=BUNDLED_V7_NO_AUDIO_CONFIGURATION_ID,
        mode=ValidationMode.test,
        profile=None,
        operation="speech",
        request_recipe="speech-no-source",
        message=r"audio encoder.*speech",
    ),
    PublicAudioErrorCase(
        case_id="audio-v7-transcription-no-marker-test",
        configuration_id=SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE,
        operation="transcription",
        request_recipe="transcription",
        message=r"TRANSCRIBE marker.*transcription",
    ),
    PublicAudioErrorCase(
        case_id="audio-v7-speech-no-marker-test",
        configuration_id=SYNTHETIC_V7_SPEECH_NO_MARKER.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_SPEECH_NO_MARKER,
        operation="speech",
        request_recipe="speech-reference",
        message=r"text_to_audio marker.*speech",
    ),
    PublicAudioErrorCase(
        case_id="audio-v7-speech-no-voice-map-test",
        configuration_id=SYNTHETIC_V7_SPEECH_NO_VOICE_MAP.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_SPEECH_NO_VOICE_MAP,
        operation="speech",
        request_recipe="speech-preset",
        message=r"(?i)preset voices.*not configured",
    ),
    PublicAudioErrorCase(
        case_id="audio-v7-instruct-transcription-no-audio-test",
        configuration_id=SYNTHETIC_V7_INSTRUCT_NO_AUDIO.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_INSTRUCT_NO_AUDIO,
        operation="transcription",
        request_recipe="transcription-audio",
        message=r"audio marker.*transcription",
    ),
    PublicAudioErrorCase(
        case_id="audio-v7-instruct-transcription-no-begin-audio-test",
        configuration_id=SYNTHETIC_V7_INSTRUCT_NO_BEGIN_AUDIO.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_INSTRUCT_NO_BEGIN_AUDIO,
        operation="transcription",
        request_recipe="transcription-audio",
        message=r"begin_audio marker.*transcription",
    ),
    PublicAudioErrorCase(
        case_id="audio-v7-speech-no-audio-token-test",
        configuration_id=SYNTHETIC_V7_SPEECH_NO_AUDIO.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_SPEECH_NO_AUDIO,
        operation="speech",
        request_recipe="speech-reference",
        message=r"audio marker.*speech",
    ),
    PublicAudioErrorCase(
        case_id="audio-v7-speech-no-begin-audio-test",
        configuration_id=SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO,
        operation="speech",
        request_recipe="speech-reference",
        message=r"begin_audio marker.*speech",
    ),
    PublicAudioErrorCase(
        case_id="audio-v7-streaming-transcription-no-streaming-pad-test",
        configuration_id=SYNTHETIC_V7_STREAMING_NO_PAD.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_STREAMING_NO_PAD,
        operation="transcription",
        request_recipe="streaming-online",
        message=r"streaming_pad marker.*transcription",
    ),
)


def _build_request(recipe: str) -> TranscriptionRequest | SpeechRequest:
    if recipe in {"transcription", "transcription-audio"}:
        return TranscriptionRequest(
            audio=b"not decoded before capability check" if recipe == "transcription" else valid_reference_audio(),
            language=None,
            target_streaming_delay_ms=None,
        )
    if recipe == "streaming-online":
        return TranscriptionRequest(
            audio="",
            streaming=StreamingMode.ONLINE,
            language=None,
            target_streaming_delay_ms=None,
        )
    if recipe == "speech-no-source":
        return SpeechRequest(input="hello")
    if recipe == "speech-reference":
        return SpeechRequest(input="hello", ref_audio=valid_reference_audio())
    assert recipe == "speech-preset"
    return SpeechRequest(input="hello", voice="preset")


@pytest.mark.parametrize("case", PUBLIC_AUDIO_ERROR_CASES, ids=lambda case: case.case_id)
def test_public_audio_capability_errors(case: PublicAudioErrorCase) -> None:
    if case.profile is None:
        assert case.configuration_id == BUNDLED_V7_NO_AUDIO_CONFIGURATION_ID
        tokenizer = load_bundled_v7_no_audio_tokenizer(mode=case.mode)
    else:
        assert case.configuration_id == case.profile.configuration_id
        tokenizer = build_synthetic_v7_audio_tokenizer(profile=case.profile, mode=case.mode)
    assert tokenizer.mode == case.mode

    request = _build_request(case.request_recipe)
    with pytest.raises(UnsupportedTokenizerFeatureException, match=case.message):
        if case.operation == "transcription":
            assert isinstance(request, TranscriptionRequest)
            tokenizer.encode_transcription(request)
        else:
            assert isinstance(request, SpeechRequest)
            tokenizer.encode_speech_request(request)

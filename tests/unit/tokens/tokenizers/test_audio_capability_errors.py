import pytest

from mistral_common.exceptions import TokenizerException, UnsupportedTokenizerFeatureException
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.audio_error_test_support import (
    SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE,
    SYNTHETIC_V7_SPEECH_NO_AUDIO_TO_TEXT,
    SYNTHETIC_V7_SPEECH_NO_MARKER,
    SYNTHETIC_V7_SPEECH_NO_VOICE_MAP,
    build_synthetic_v7_audio_tokenizer,
    load_bundled_v7_no_audio_tokenizer,
    valid_reference_audio,
)


@pytest.mark.parametrize(
    ("configuration", "operation", "request_recipe", "message"),
    [
        pytest.param(
            "bundled-v7-no-audio",
            "transcription",
            "transcription",
            r"audio encoder.*transcription",
            id="v7-transcription-without-audio-encoder",
        ),
        pytest.param(
            "bundled-v7-no-audio",
            "speech",
            "speech-no-source",
            r"audio encoder.*speech",
            id="v7-speech-without-encoder-takes-priority-over-missing-source",
        ),
        pytest.param(
            "no-transcribe-marker",
            "transcription",
            "transcription",
            r"TRANSCRIBE marker.*transcription",
            id="v7-instruct-profile-without-transcribe-marker",
        ),
        pytest.param(
            "no-speech-marker",
            "speech",
            "speech-reference",
            r"text_to_audio marker.*speech",
            id="v7-speech-profile-without-required-marker",
        ),
        pytest.param(
            "no-audio-to-text-marker",
            "speech",
            "speech-reference",
            r"audio_to_text marker.*speech",
            id="v7-speech-profile-without-audio-to-text-marker",
        ),
        pytest.param(
            "no-voice-map",
            "speech",
            "speech-preset",
            r"(?i)preset voices.*not configured",
            id="v7-speech-profile-without-voice-map",
        ),
    ],
)
def test_direct_unsupported_audio_capability(
    configuration: str,
    operation: str,
    request_recipe: str,
    message: str,
) -> None:
    if configuration == "bundled-v7-no-audio":
        tokenizer = load_bundled_v7_no_audio_tokenizer(mode=ValidationMode.test)
    elif configuration == "no-transcribe-marker":
        tokenizer = build_synthetic_v7_audio_tokenizer(
            profile=SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE, mode=ValidationMode.test
        )
    elif configuration == "no-speech-marker":
        tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_SPEECH_NO_MARKER, mode=ValidationMode.test)
    elif configuration == "no-audio-to-text-marker":
        tokenizer = build_synthetic_v7_audio_tokenizer(
            profile=SYNTHETIC_V7_SPEECH_NO_AUDIO_TO_TEXT, mode=ValidationMode.test
        )
    else:
        assert configuration == "no-voice-map"
        tokenizer = build_synthetic_v7_audio_tokenizer(
            profile=SYNTHETIC_V7_SPEECH_NO_VOICE_MAP, mode=ValidationMode.test
        )

    request: TranscriptionRequest | SpeechRequest
    if request_recipe == "transcription":
        request = TranscriptionRequest(
            audio=b"not decoded before capability check",
            language=None,
            target_streaming_delay_ms=None,
        )
    elif request_recipe == "speech-no-source":
        request = SpeechRequest(input="hello")
    elif request_recipe == "speech-reference":
        request = SpeechRequest(input="hello", ref_audio=valid_reference_audio())
    else:
        assert request_recipe == "speech-preset"
        request = SpeechRequest(input="hello", voice="preset")

    with pytest.raises(UnsupportedTokenizerFeatureException, match=message):
        if operation == "transcription":
            assert isinstance(request, TranscriptionRequest)
            tokenizer.instruct_tokenizer.encode_transcription(request)
        else:
            assert operation == "speech"
            assert isinstance(request, SpeechRequest)
            tokenizer.instruct_tokenizer.encode_speech_request(request)


@pytest.mark.parametrize(
    ("version", "operation", "message"),
    [
        pytest.param(
            "v1", "transcription", r"Transcription not available for TokenizerVersion\.v1", id="v1-transcription"
        ),
        pytest.param("v1", "speech", r"Speech request not available for tokenizer v1", id="v1-speech"),
        pytest.param(
            "v2", "transcription", r"Transcription not available for TokenizerVersion\.v2", id="v2-transcription"
        ),
        pytest.param("v2", "speech", r"Speech request not available for tokenizer v2", id="v2-speech"),
        pytest.param(
            "v3", "transcription", r"Transcription not available for TokenizerVersion\.v3", id="v3-transcription"
        ),
        pytest.param("v3", "speech", r"Speech request not available for tokenizer v3", id="v3-speech"),
    ],
)
def test_pre_v7_audio_operation_remains_tokenizer_error(version: str, operation: str, message: str) -> None:
    if version == "v1":
        tokenizer = MistralTokenizer.v1()
    elif version == "v2":
        tokenizer = MistralTokenizer.v2()
    else:
        assert version == "v3"
        tokenizer = MistralTokenizer.v3()

    request: TranscriptionRequest | SpeechRequest
    if operation == "transcription":
        request = TranscriptionRequest(audio=b"audio", language=None, target_streaming_delay_ms=None)
    else:
        assert operation == "speech"
        request = SpeechRequest(input="hello")

    with pytest.raises(TokenizerException, match=message):
        if operation == "transcription":
            assert isinstance(request, TranscriptionRequest)
            tokenizer.instruct_tokenizer.encode_transcription(request)
        else:
            assert isinstance(request, SpeechRequest)
            tokenizer.instruct_tokenizer.encode_speech_request(request)

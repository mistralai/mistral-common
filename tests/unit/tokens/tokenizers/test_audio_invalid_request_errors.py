import numpy as np
import pytest

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import Audio
from tests.audio_error_test_support import (
    SYNTHETIC_V7_INSTRUCT,
    SYNTHETIC_V7_SPEECH,
    SYNTHETIC_V7_STREAMING,
    SyntheticV7AudioProfile,
    build_synthetic_v7_audio_tokenizer,
    valid_reference_audio,
    valid_reference_audio_bytes,
)


@pytest.mark.parametrize(
    ("profile", "request_recipe", "message"),
    [
        pytest.param(
            SYNTHETIC_V7_INSTRUCT,
            "instruct-offline",
            r"Instruct transcription.*DISABLED.*OFFLINE",
            id="instruct-offline-mode",
        ),
        pytest.param(
            SYNTHETIC_V7_INSTRUCT,
            "instruct-online",
            r"Instruct transcription.*DISABLED.*ONLINE",
            id="instruct-online-mode",
        ),
        pytest.param(
            SYNTHETIC_V7_STREAMING,
            "streaming-disabled",
            r"Streaming transcription.*OFFLINE.*ONLINE.*DISABLED",
            id="streaming-disabled-mode",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH,
            "speech-no-source",
            r"Either ref_audio or voice must be defined",
            id="speech-source-required",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH,
            "speech-unknown-voice",
            r"Unknown voice.*expected one of \['preset'\]",
            id="speech-unknown-configured-voice",
        ),
        pytest.param(
            SYNTHETIC_V7_STREAMING,
            "online-bytes",
            r"ONLINE streaming audio.*base64 text.*bytes",
            id="online-nonempty-wav-bytes",
        ),
    ],
)
def test_direct_invalid_audio_request(
    profile: SyntheticV7AudioProfile,
    request_recipe: str,
    message: str,
) -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=profile, mode=ValidationMode.test)
    request: TranscriptionRequest | SpeechRequest

    if request_recipe == "instruct-offline":
        request = TranscriptionRequest(
            audio=valid_reference_audio(),
            streaming=StreamingMode.OFFLINE,
            language=None,
            target_streaming_delay_ms=None,
        )
    elif request_recipe == "instruct-online":
        request = TranscriptionRequest(
            audio=valid_reference_audio(),
            streaming=StreamingMode.ONLINE,
            language=None,
            target_streaming_delay_ms=None,
        )
    elif request_recipe == "streaming-disabled":
        request = TranscriptionRequest(
            audio=valid_reference_audio(),
            streaming=StreamingMode.DISABLED,
            language=None,
            target_streaming_delay_ms=None,
        )
    elif request_recipe == "speech-no-source":
        request = SpeechRequest(input="hello")
    elif request_recipe == "speech-unknown-voice":
        request = SpeechRequest(input="hello", voice="not-configured")
    else:
        assert request_recipe == "online-bytes"
        request = TranscriptionRequest(
            audio=valid_reference_audio_bytes(),
            streaming=StreamingMode.ONLINE,
            language=None,
            target_streaming_delay_ms=None,
        )

    with pytest.raises(InvalidRequestException, match=message):
        if isinstance(request, TranscriptionRequest):
            tokenizer.instruct_tokenizer.encode_transcription(request)
        else:
            tokenizer.instruct_tokenizer.encode_speech_request(request)


def test_reference_audio_ignores_unknown_voice_with_direct_encoder() -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_SPEECH, mode=ValidationMode.test)
    request = SpeechRequest(input="hello", ref_audio=valid_reference_audio(), voice="not-configured")

    tokenized = tokenizer.instruct_tokenizer.encode_speech_request(request)

    expected_audio = Audio.from_base64(valid_reference_audio())
    assert len(tokenized.audios) == 1
    assert np.allclose(tokenized.audios[0].audio_array, expected_audio.audio_array, atol=1e-3)


def test_online_base64_audio_keeps_direct_warning_and_combine_path() -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_STREAMING, mode=ValidationMode.test)
    input_audio = Audio(
        audio_array=np.linspace(-0.75, 0.75, 2_400, dtype=np.float32),
        sampling_rate=24_000,
        format="wav",
    )
    encoded_audio = input_audio.to_base64("wav")
    request = TranscriptionRequest(
        audio=encoded_audio,
        streaming=StreamingMode.ONLINE,
        language=None,
        target_streaming_delay_ms=None,
    )

    with pytest.warns(FutureWarning, match="Passing audio.*deprecated"):
        tokenized = tokenizer.instruct_tokenizer.encode_transcription(request)

    decoded_audio = Audio.from_base64(encoded_audio)
    assert len(tokenized.audios) == 1
    assert tokenized.audios[0].sampling_rate == decoded_audio.sampling_rate
    audio_encoder = tokenizer.instruct_tokenizer.audio_encoder
    assert audio_encoder is not None
    audio_config = audio_encoder.audio_config
    left_padding_samples = audio_config.n_left_pad_tokens * audio_config.raw_audio_length_per_tok
    expected_audio = np.concatenate(
        (np.zeros(left_padding_samples, dtype=decoded_audio.audio_array.dtype), decoded_audio.audio_array)
    )
    np.testing.assert_array_equal(tokenized.audios[0].audio_array, expected_audio)

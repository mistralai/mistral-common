from collections.abc import Callable

import numpy as np
import pytest
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.exceptions import InvalidRequestException, TokenizerException, UnsupportedTokenizerFeatureException
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import (
    Audio,
)
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV7,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.test_tokenizer_v7_audio import (
    SYNTHETIC_V7_INSTRUCT,
    SYNTHETIC_V7_INSTRUCT_NO_AUDIO,
    SYNTHETIC_V7_INSTRUCT_NO_BEGIN_AUDIO,
    SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE,
    SyntheticV7AudioProfile,
    _get_audio_chunk,
    _get_specials,
    get_tekkenizer_with_audio,
    load_v7_audio_error_tokenizer,
    undecodable_transcription_request,
    valid_reference_audio,
    valid_transcription_request,
)
from tests.utils import decode_keep


@pytest.fixture(scope="session")
def tekkenizer() -> InstructTokenizerV7:
    return get_tekkenizer_with_audio()


def get_transcription_request(duration: float, language: LanguageAlpha2 | None = None) -> TranscriptionRequest:
    audio_chunk = _get_audio_chunk(duration)

    return TranscriptionRequest(
        model="dummy", audio=audio_chunk.input_audio, language=language, target_streaming_delay_ms=None
    )


def test_tokenize_transcribe(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    num_expected_frames = int(np.ceil(duration * frame_rate))

    request = get_transcription_request(duration)

    tokenized = tekkenizer.encode_transcription(request)
    BOS, _, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, TRANSCRIBE = _get_specials(tekkenizer)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        *audio_toks,
        END_INST,
        TRANSCRIBE,
    ]
    text = decode_keep(tekkenizer, tokenized)
    assert text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST][TRANSCRIBE]")
    assert len(tokenized.audios) == 1
    base64_audio = request.audio
    assert isinstance(base64_audio, str)
    audio_array = Audio.from_base64(base64_audio).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


def test_tokenize_transcribe_with_lang(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    num_expected_frames = int(np.ceil(duration * frame_rate))

    request = get_transcription_request(duration, language=LanguageAlpha2("en"))

    tokenized = tekkenizer.encode_transcription(request)
    BOS, _, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, TRANSCRIBE = _get_specials(tekkenizer)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        *audio_toks,
        END_INST,
        208,
        197,
        210,
        203,
        158,
        201,
        210,
        TRANSCRIBE,
    ]
    text = decode_keep(tekkenizer, tokenized)
    assert text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST]lang:en[TRANSCRIBE]")
    assert len(tokenized.audios) == 1
    base64_audio = request.audio
    assert isinstance(base64_audio, str)
    audio_array = Audio.from_base64(base64_audio).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


TRANSCRIPTION_ERROR_CASES = (
    pytest.param(
        None,
        undecodable_transcription_request,
        UnsupportedTokenizerFeatureException,
        r"Tokenizer v7 does not provide an audio encoder for transcription",
        id="audio-v7-transcription-no-encoder-test",
    ),
    pytest.param(
        SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE,
        undecodable_transcription_request,
        UnsupportedTokenizerFeatureException,
        r"the TRANSCRIBE marker required for transcription",
        id="audio-v7-transcription-no-marker-test",
    ),
    pytest.param(
        SYNTHETIC_V7_INSTRUCT_NO_AUDIO,
        valid_transcription_request,
        UnsupportedTokenizerFeatureException,
        r"the audio marker required for transcription",
        id="audio-v7-instruct-transcription-no-audio-test",
    ),
    pytest.param(
        SYNTHETIC_V7_INSTRUCT_NO_BEGIN_AUDIO,
        valid_transcription_request,
        UnsupportedTokenizerFeatureException,
        r"the begin_audio marker required for transcription",
        id="audio-v7-instruct-transcription-no-begin-audio-test",
    ),
    pytest.param(
        SYNTHETIC_V7_INSTRUCT,
        lambda: TranscriptionRequest(
            audio=b"not decodable audio",
            streaming=StreamingMode.OFFLINE,
            language=None,
            target_streaming_delay_ms=None,
        ),
        InvalidRequestException,
        r"^Instruct transcription requires streaming mode DISABLED, got OFFLINE\.$",
        id="audio-v7-instruct-offline-rejected-test",
    ),
    pytest.param(
        SYNTHETIC_V7_INSTRUCT,
        lambda: TranscriptionRequest(
            audio=valid_reference_audio(),
            streaming=StreamingMode.ONLINE,
            language=None,
            target_streaming_delay_ms=None,
        ),
        InvalidRequestException,
        r"Instruct transcription.*DISABLED.*ONLINE",
        id="audio-v7-instruct-online-rejected-test",
    ),
)


@pytest.mark.parametrize(("profile", "request_factory", "error_type", "message"), TRANSCRIPTION_ERROR_CASES)
def test_public_transcription_errors(
    profile: SyntheticV7AudioProfile | None,
    request_factory: Callable[[], TranscriptionRequest],
    error_type: type[Exception],
    message: str,
) -> None:
    tokenizer = load_v7_audio_error_tokenizer(profile=profile)
    assert tokenizer.mode == ValidationMode.test
    request = request_factory()

    with pytest.raises(error_type, match=message):
        tokenizer.encode_transcription(request)


@pytest.mark.parametrize(("profile", "request_factory", "error_type", "message"), TRANSCRIPTION_ERROR_CASES)
def test_direct_transcription_errors(
    profile: SyntheticV7AudioProfile | None,
    request_factory: Callable[[], TranscriptionRequest],
    error_type: type[Exception],
    message: str,
) -> None:
    tokenizer = load_v7_audio_error_tokenizer(profile=profile)
    assert tokenizer.mode == ValidationMode.test
    request = request_factory()

    with pytest.raises(error_type, match=message):
        tokenizer.instruct_tokenizer.encode_transcription(request)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        pytest.param(
            MistralTokenizer.v1,
            r"^Transcription not available for (?:TokenizerVersion\.)?v1$",
            id="v1-transcription",
        ),
        pytest.param(
            MistralTokenizer.v2,
            r"^Transcription not available for (?:TokenizerVersion\.)?v2$",
            id="v2-transcription",
        ),
        pytest.param(
            MistralTokenizer.v3,
            r"^Transcription not available for (?:TokenizerVersion\.)?v3$",
            id="v3-transcription",
        ),
    ],
)
def test_pre_v7_transcription_remains_tokenizer_error(factory: Callable[[], MistralTokenizer], message: str) -> None:
    tokenizer = factory()
    request = TranscriptionRequest(audio=b"audio", language=None, target_streaming_delay_ms=None)
    with pytest.raises(TokenizerException, match=message):
        tokenizer.instruct_tokenizer.encode_transcription(request)

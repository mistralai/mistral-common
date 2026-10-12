from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

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
    BUNDLED_V7_NO_AUDIO_CONFIGURATION_ID,
    SYNTHETIC_V7_INSTRUCT,
    SYNTHETIC_V7_INSTRUCT_NO_AUDIO,
    SYNTHETIC_V7_INSTRUCT_NO_BEGIN_AUDIO,
    SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE,
    SyntheticV7AudioProfile,
    _get_audio_chunk,
    _get_specials,
    get_tekkenizer_with_audio,
    load_v7_audio_error_tokenizer,
    v7_audio_error_configuration_id,
    valid_reference_audio,
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


@dataclass(frozen=True)
class TranscriptionErrorCase:
    case_id: str
    configuration_id: str
    mode: ValidationMode
    profile: SyntheticV7AudioProfile | None
    request_recipe: Literal["unparsed-audio", "valid-audio", "instruct-offline", "instruct-online"]
    error_type: type[Exception]
    message: str


TRANSCRIPTION_ERROR_CASES = (
    TranscriptionErrorCase(
        case_id="audio-v7-transcription-no-encoder-test",
        configuration_id=BUNDLED_V7_NO_AUDIO_CONFIGURATION_ID,
        mode=ValidationMode.test,
        profile=None,
        request_recipe="unparsed-audio",
        error_type=UnsupportedTokenizerFeatureException,
        message=r"Tokenizer v7 does not provide an audio encoder for transcription",
    ),
    TranscriptionErrorCase(
        case_id="audio-v7-transcription-no-marker-test",
        configuration_id=SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE,
        request_recipe="unparsed-audio",
        error_type=UnsupportedTokenizerFeatureException,
        message=r"the TRANSCRIBE marker required for transcription",
    ),
    TranscriptionErrorCase(
        case_id="audio-v7-instruct-transcription-no-audio-test",
        configuration_id=SYNTHETIC_V7_INSTRUCT_NO_AUDIO.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_INSTRUCT_NO_AUDIO,
        request_recipe="valid-audio",
        error_type=UnsupportedTokenizerFeatureException,
        message=r"the audio marker required for transcription",
    ),
    TranscriptionErrorCase(
        case_id="audio-v7-instruct-transcription-no-begin-audio-test",
        configuration_id=SYNTHETIC_V7_INSTRUCT_NO_BEGIN_AUDIO.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_INSTRUCT_NO_BEGIN_AUDIO,
        request_recipe="valid-audio",
        error_type=UnsupportedTokenizerFeatureException,
        message=r"the begin_audio marker required for transcription",
    ),
    TranscriptionErrorCase(
        case_id="audio-v7-instruct-offline-rejected-test",
        configuration_id=SYNTHETIC_V7_INSTRUCT.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_INSTRUCT,
        request_recipe="instruct-offline",
        error_type=InvalidRequestException,
        message=r"^Instruct transcription requires streaming mode DISABLED, got OFFLINE\.$",
    ),
    TranscriptionErrorCase(
        case_id="audio-v7-instruct-online-rejected-test",
        configuration_id=SYNTHETIC_V7_INSTRUCT.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_INSTRUCT,
        request_recipe="instruct-online",
        error_type=InvalidRequestException,
        message=r"Instruct transcription.*DISABLED.*ONLINE",
    ),
)


def _build_transcription_error_request(
    recipe: Literal["unparsed-audio", "valid-audio", "instruct-offline", "instruct-online"],
) -> TranscriptionRequest:
    if recipe == "unparsed-audio":
        return TranscriptionRequest(
            audio=b"not decoded before capability check", language=None, target_streaming_delay_ms=None
        )
    if recipe == "valid-audio":
        return TranscriptionRequest(audio=valid_reference_audio(), language=None, target_streaming_delay_ms=None)
    if recipe == "instruct-offline":
        return TranscriptionRequest(
            audio=b"not decodable audio",
            streaming=StreamingMode.OFFLINE,
            language=None,
            target_streaming_delay_ms=None,
        )

    assert recipe == "instruct-online"
    return TranscriptionRequest(
        audio=valid_reference_audio(),
        streaming=StreamingMode.ONLINE,
        language=None,
        target_streaming_delay_ms=None,
    )


@pytest.mark.parametrize("case", TRANSCRIPTION_ERROR_CASES, ids=lambda case: case.case_id)
def test_public_transcription_errors(case: TranscriptionErrorCase) -> None:
    assert case.configuration_id == v7_audio_error_configuration_id(profile=case.profile)
    tokenizer = load_v7_audio_error_tokenizer(profile=case.profile, mode=case.mode)
    assert tokenizer.mode == case.mode
    request = _build_transcription_error_request(recipe=case.request_recipe)

    with pytest.raises(case.error_type, match=case.message):
        tokenizer.encode_transcription(request)


@pytest.mark.parametrize("case", TRANSCRIPTION_ERROR_CASES, ids=lambda case: case.case_id)
def test_direct_transcription_errors(case: TranscriptionErrorCase) -> None:
    assert case.configuration_id == v7_audio_error_configuration_id(profile=case.profile)
    tokenizer = load_v7_audio_error_tokenizer(profile=case.profile, mode=case.mode)
    assert tokenizer.mode == case.mode
    request = _build_transcription_error_request(recipe=case.request_recipe)

    with pytest.raises(case.error_type, match=case.message):
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

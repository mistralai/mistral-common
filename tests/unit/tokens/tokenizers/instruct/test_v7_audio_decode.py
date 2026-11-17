r"""Direct v7 transcription and speech audio decode-error behavior.

Undecodable audio on the four request decode paths (instruct transcription,
streaming OFFLINE, and speech reference audio) leaks raw built-in or backend
errors: a base64 ``ValueError``, a ``soundfile.LibsndfileError`` for bytes
soundfile cannot open, and the strict multichannel ``ValueError``. These cases
pin the raw errors as the current direct-encoder behavior; the chat chunk
path keeps the same raw errors, so the controls below prove the boundary.
"""

import base64
import binascii
import io
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from mistral_common.exceptions import InvalidRequestException, UnsupportedTokenizerFeatureException
from mistral_common.protocol.instruct.chunk import AudioChunk
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ModelSettings, ReasoningEffort
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import (
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV15
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder, ReasoningEffortEnumBuilder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.test_tokenizer_v7_audio import (
    SYNTHETIC_V7_INSTRUCT,
    SYNTHETIC_V7_SPEECH,
    SYNTHETIC_V7_SPEECH_NO_MARKER,
    SYNTHETIC_V7_STREAMING,
    SYNTHETIC_V7_STREAMING_NO_PAD,
    SyntheticV7AudioProfile,
    _get_audio_chunk,
    build_synthetic_v7_audio_tokenizer,
    get_tekkenizer_with_audio,
)


def multichannel_wav_bytes() -> bytes:
    r"""Build a fresh two-channel 24 kHz float32 WAV of 2,400 zero frames."""
    buffer = io.BytesIO()
    sf.write(file=buffer, data=np.zeros(shape=(2_400, 2), dtype=np.float32), samplerate=24_000, format="WAV")
    return buffer.getvalue()


def assert_audio_decode_error(
    *,
    encode: Callable[[], object],
    error_type: type[Exception],
    message: str,
    cause_type: type[Exception] | None,
    cause_message: str | None,
) -> None:
    r"""Run `encode` and assert the raised error type, message and `__cause__`.

    Args:
        encode: The zero-argument call expected to raise.
        error_type: The exact exception type expected.
        message: Regex matched against the raised error's message.
        cause_type: The exact type expected for `__cause__`, or `None` to
            require no cause.
        cause_message: Regex matched against the cause's message when
            `cause_type` is set.
    """
    with pytest.raises(error_type, match=message) as exc_info:
        encode()
    cause = exc_info.value.__cause__
    if cause_type is None:
        assert cause is None
    else:
        assert type(cause) is cause_type
        assert cause_message is not None
        assert re.search(cause_message, str(cause))


def _base64_cause_type(error_type: type[Exception], message: str) -> type[Exception] | None:
    r"""The `binascii.Error` cause of a base64 `ValueError`, or `None`."""
    if error_type is ValueError and "base64" in message:
        return binascii.Error
    return None


def _base64_cause_message(error_type: type[Exception], message: str) -> str | None:
    r"""The cause-message pattern of a base64 `ValueError`, or `None`."""
    if error_type is ValueError and "base64" in message:
        return r"Invalid base64-encoded string"
    return None


@dataclass(frozen=True)
class _DecodeErrorCase:
    case_id: str
    profile: SyntheticV7AudioProfile
    audio: str | bytes
    cause_type: type[Exception]
    cause_message: str


_TRANSCRIPTION_DECODE_ERROR_CASES: tuple[_DecodeErrorCase, ...] = (
    _DecodeErrorCase(
        case_id="audio-v7-instruct-invalid-base64-rejected-test",
        profile=SYNTHETIC_V7_INSTRUCT,
        audio="not base64!",
        cause_type=ValueError,
        cause_message=r"^base64 decoding failed",
    ),
    _DecodeErrorCase(
        case_id="audio-v7-instruct-non-audio-bytes-rejected-test",
        profile=SYNTHETIC_V7_INSTRUCT,
        audio=b"not audio",
        cause_type=sf.LibsndfileError,
        cause_message=r"Format not recognised",
    ),
    _DecodeErrorCase(
        case_id="audio-v7-instruct-multichannel-rejected-test",
        profile=SYNTHETIC_V7_INSTRUCT,
        audio=multichannel_wav_bytes(),
        cause_type=ValueError,
        cause_message=r"^audio_array\.ndim=2$",
    ),
    _DecodeErrorCase(
        case_id="audio-v7-instruct-empty-audio-rejected-test",
        profile=SYNTHETIC_V7_INSTRUCT,
        audio=b"",
        cause_type=ValidationError,
        cause_message=r"`input_audio` should not be empty",
    ),
)


@pytest.mark.parametrize("case", _TRANSCRIPTION_DECODE_ERROR_CASES, ids=lambda case: case.case_id)
def test_direct_transcription_decode_errors(case: _DecodeErrorCase) -> None:
    r"""Direct instruct transcription raises the raw decode error for each input kind."""
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=case.profile)
    assert tokenizer.mode == ValidationMode.test
    request = TranscriptionRequest(audio=case.audio, language=None, target_streaming_delay_ms=None)

    assert_audio_decode_error(
        encode=lambda: tokenizer.instruct_tokenizer.encode_transcription(request),
        error_type=case.cause_type,
        message=case.cause_message,
        cause_type=_base64_cause_type(case.cause_type, case.cause_message),
        cause_message=_base64_cause_message(case.cause_type, case.cause_message),
    )


@pytest.mark.parametrize("audio_kind", ["base64-wav", "wav-bytes"])
def test_instruct_transcription_matches_user_message_encoding(audio_kind: str) -> None:
    r"""Instruct transcription tokens and audio equal the shared user-message encoding."""
    tekkenizer = get_tekkenizer_with_audio()
    audio = _get_audio_chunk(duration=1.7).input_audio
    assert isinstance(audio, str)
    request_audio: str | bytes = audio if audio_kind == "base64-wav" else base64.b64decode(audio)

    request = TranscriptionRequest(audio=request_audio, language=None, target_streaming_delay_ms=None)
    tokenized = tekkenizer.encode_transcription(request)

    oracle_tokens, oracle_images, oracle_audios = tekkenizer.encode_user_message(
        UserMessage(content=[AudioChunk(input_audio=request_audio)]),
        available_tools=[],
        is_last=True,
        is_first=True,
        system_prompt=None,
        settings=ModelSettings.none(),
    )
    assert not oracle_images
    assert tokenized.tokens == [*tekkenizer.start(), *oracle_tokens, tekkenizer.TRANSCRIBE]
    assert len(tokenized.audios) == 1
    assert len(oracle_audios) == 1
    np.testing.assert_array_equal(tokenized.audios[0].audio_array, oracle_audios[0].audio_array)
    assert tokenized.audios[0].sampling_rate == oracle_audios[0].sampling_rate
    assert tokenized.audios[0].format == oracle_audios[0].format


def _build_v15_audio_tokenizer() -> InstructTokenizerV15:
    r"""Synthetic v15 tokenizer with an audio encoder and a reasoning-effort builder."""
    model_settings_builder = ModelSettingsBuilder(
        reasoning_effort=ReasoningEffortEnumBuilder(values=list(ReasoningEffort), default=list(ReasoningEffort)[0])
    )
    tekkenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"a", b"b", b"c", b"f", b"de"]),
        special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v15, add_think=False, add_audio=True),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v15,
        model_settings_builder=model_settings_builder,
    )
    return InstructTokenizerV15(
        tokenizer=tekkenizer,
        audio_encoder=AudioEncoder(
            audio_config=AudioConfig(
                sampling_rate=24_000,
                frame_rate=12.5,
                encoding_config=AudioSpectrogramConfig(num_mel_bins=128, window_size=400, hop_length=160),
            ),
            special_ids=SpecialAudioIDs(
                audio=tekkenizer.get_special_token(SpecialTokens.audio.value),
                begin_audio=tekkenizer.get_special_token(SpecialTokens.begin_audio.value),
                streaming_pad=None,
                text_to_audio=None,
                audio_to_text=None,
            ),
        ),
    )


def test_instruct_transcription_settings_check_precedes_decoding() -> None:
    r"""A v15 reasoning-effort tokenizer rejects empty settings before decoding audio."""
    tokenizer = _build_v15_audio_tokenizer()
    request = TranscriptionRequest(audio=b"not audio", language=None, target_streaming_delay_ms=None)

    with pytest.raises(InvalidRequestException, match=r"^reasoning_effort should be set for this model\.$") as exc_info:
        tokenizer.encode_transcription(request)
    assert exc_info.value.__cause__ is None


@dataclass(frozen=True)
class _StreamingDecodeErrorCase:
    case_id: str
    profile: SyntheticV7AudioProfile
    streaming: StreamingMode
    audio: str | bytes
    error_type: type[Exception]
    message: str
    cause_type: type[Exception] | None
    cause_message: str | None


_STREAMING_DECODE_ERROR_CASES: tuple[_StreamingDecodeErrorCase, ...] = (
    _StreamingDecodeErrorCase(
        case_id="audio-v7-offline-invalid-base64-rejected-test",
        profile=SYNTHETIC_V7_STREAMING,
        streaming=StreamingMode.OFFLINE,
        audio="not base64!",
        error_type=ValueError,
        message=r"^base64 decoding failed",
        cause_type=None,
        cause_message=None,
    ),
    _StreamingDecodeErrorCase(
        case_id="audio-v7-offline-non-audio-bytes-rejected-test",
        profile=SYNTHETIC_V7_STREAMING,
        streaming=StreamingMode.OFFLINE,
        audio=b"not audio",
        error_type=sf.LibsndfileError,
        message=r"Format not recognised",
        cause_type=None,
        cause_message=None,
    ),
    _StreamingDecodeErrorCase(
        case_id="audio-v7-offline-multichannel-rejected-test",
        profile=SYNTHETIC_V7_STREAMING,
        streaming=StreamingMode.OFFLINE,
        audio=multichannel_wav_bytes(),
        error_type=ValueError,
        message=r"^audio_array\.ndim=2$",
        cause_type=None,
        cause_message=None,
    ),
    _StreamingDecodeErrorCase(
        case_id="audio-v7-offline-streaming-pad-precedes-decoding-test",
        profile=SYNTHETIC_V7_STREAMING_NO_PAD,
        streaming=StreamingMode.OFFLINE,
        audio=b"not audio",
        error_type=UnsupportedTokenizerFeatureException,
        message=r"^Audio encoder does not provide the streaming_pad marker required for transcription\.$",
        cause_type=None,
        cause_message=None,
    ),
)


@pytest.mark.parametrize("case", _STREAMING_DECODE_ERROR_CASES, ids=lambda case: case.case_id)
def test_direct_streaming_transcription_decode_errors(case: _StreamingDecodeErrorCase) -> None:
    r"""Direct streaming OFFLINE transcription raises the raw decode error per input kind."""
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=case.profile)
    assert tokenizer.mode == ValidationMode.test
    request = TranscriptionRequest(
        audio=case.audio, streaming=case.streaming, language=None, target_streaming_delay_ms=None
    )

    assert_audio_decode_error(
        encode=lambda: tokenizer.instruct_tokenizer.encode_transcription(request),
        error_type=case.error_type,
        message=case.message,
        cause_type=_base64_cause_type(case.error_type, case.message) or case.cause_type,
        cause_message=_base64_cause_message(case.error_type, case.message) or case.cause_message,
    )


_SPEECH_DECODE_ERROR_CASES: tuple[_DecodeErrorCase, ...] = (
    _DecodeErrorCase(
        case_id="audio-v7-speech-invalid-base64-rejected-test",
        profile=SYNTHETIC_V7_SPEECH,
        audio="not base64!",
        cause_type=ValueError,
        cause_message=r"^base64 decoding failed",
    ),
    _DecodeErrorCase(
        case_id="audio-v7-speech-non-audio-bytes-rejected-test",
        profile=SYNTHETIC_V7_SPEECH,
        audio=b"not audio",
        cause_type=sf.LibsndfileError,
        cause_message=r"Format not recognised",
    ),
    _DecodeErrorCase(
        case_id="audio-v7-speech-multichannel-rejected-test",
        profile=SYNTHETIC_V7_SPEECH,
        audio=multichannel_wav_bytes(),
        cause_type=ValueError,
        cause_message=r"^audio_array\.ndim=2$",
    ),
    _DecodeErrorCase(
        case_id="audio-v7-speech-decoding-precedes-marker-test",
        profile=SYNTHETIC_V7_SPEECH_NO_MARKER,
        audio="not base64!",
        cause_type=ValueError,
        cause_message=r"^base64 decoding failed",
    ),
)


@pytest.mark.parametrize("case", _SPEECH_DECODE_ERROR_CASES, ids=lambda case: case.case_id)
def test_direct_speech_decode_errors(case: _DecodeErrorCase) -> None:
    r"""Direct speech reference decoding raises the raw decode error before marker checks."""
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=case.profile)
    assert tokenizer.mode == ValidationMode.test
    request = SpeechRequest(input="hello", ref_audio=case.audio)

    assert_audio_decode_error(
        encode=lambda: tokenizer.instruct_tokenizer.encode_speech_request(request),
        error_type=case.cause_type,
        message=case.cause_message,
        cause_type=_base64_cause_type(case.cause_type, case.cause_message),
        cause_message=_base64_cause_message(case.cause_type, case.cause_message),
    )


_CHAT_RAW_ERROR_CASES = (
    pytest.param("not base64!", ValueError, r"^base64 decoding failed", id="invalid-base64"),
    pytest.param(b"not audio", sf.LibsndfileError, r"Format not recognised", id="non-audio-bytes"),
    pytest.param(multichannel_wav_bytes(), ValueError, r"^audio_array\.ndim=2$", id="multichannel-wav"),
)


@pytest.mark.parametrize(("audio", "error_type", "message"), _CHAT_RAW_ERROR_CASES)
def test_chat_audio_decode_errors_stay_raw(audio: str | bytes, error_type: type[Exception], message: str) -> None:
    r"""Chat inline AudioChunk decoding keeps raising the raw error, not a project error."""
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_INSTRUCT)
    assert tokenizer.mode == ValidationMode.test
    request = ChatCompletionRequest(messages=[UserMessage(content=[AudioChunk(input_audio=audio)])])

    with pytest.raises(error_type, match=message):
        tokenizer.encode_chat_completion(request)

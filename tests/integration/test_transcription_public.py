r"""Public transcription workflow cases.

Each case builds a fresh request from its Python recipe, loads its bound
tokenizer configuration through the session-cached public loader and calls the
public ``encode_transcription`` entry point. Success cases compare the complete
token ids, decoded text and returned audio against the reviewed manifest under
``tests/data/expected/<case-id>/``; error cases assert the raw decode error the
current public boundary raises. Requests are constructed before the
expected-error assertion, so a construction failure cannot satisfy a case.
"""

import base64
import io
from collections.abc import Callable

import numpy as np
import pytest
import soundfile as sf

from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.expected_results import assert_public_success, load_expected_success
from tests.integration.tokenizer_configurations import (
    PINNED_V7_AUDIO_TEST,
    PINNED_V7_REALTIME_TEST,
    TokenizerConfiguration,
)
from tests.integration.transcription_cases import (
    TRANSCRIPTION_ERROR_CASES,
    TRANSCRIPTION_REALTIME_OFFLINE_DEFAULT_DELAY,
    TRANSCRIPTION_REALTIME_OFFLINE_DELAY_0MS,
    TRANSCRIPTION_REALTIME_ONLINE_EMPTY_PREFILL_DEFAULT_DELAY,
    TRANSCRIPTION_SUCCESS_CASES,
    PublicTranscriptionErrorCase,
    PublicTranscriptionSuccessCase,
)
from tests.utils import TOKENIZED_ATOL, TOKENIZED_RTOL, assert_tokenized_equal, decode_keep


@pytest.mark.parametrize(argnames="case", argvalues=TRANSCRIPTION_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_public_transcription_success(
    case: PublicTranscriptionSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    tokenized = tokenizer.encode_transcription(request)
    decoded_text = decode_keep(tokenizer=tokenizer, tokenized=tokenized)
    expected = load_expected_success(
        case_id=case.case_id,
        tokenizer_configuration_id=case.configuration.configuration_id,
    )
    assert_public_success(expected=expected, tokenized=tokenized, decoded_text=decoded_text)


@pytest.mark.parametrize(argnames="case", argvalues=TRANSCRIPTION_ERROR_CASES, ids=lambda case: case.case_id)
def test_public_transcription_decode_error(
    case: PublicTranscriptionErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    with pytest.raises(expected_exception=case.expected_exception, match=case.message_pattern):
        tokenizer.encode_transcription(request)


def _offline_result(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer], *, target_streaming_delay_ms: int | None
) -> Tokenized:
    recipe = (
        TRANSCRIPTION_REALTIME_OFFLINE_DEFAULT_DELAY
        if target_streaming_delay_ms is None
        else TRANSCRIPTION_REALTIME_OFFLINE_DELAY_0MS
    )
    tokenizer = public_tokenizer(PINNED_V7_REALTIME_TEST)
    tokenized: Tokenized = tokenizer.encode_transcription(recipe.build())
    return tokenized


def test_public_streaming_offline_zero_delay_differs_from_default(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    default = _offline_result(public_tokenizer, target_streaming_delay_ms=None)
    zero_delay = _offline_result(public_tokenizer, target_streaming_delay_ms=0)

    assert zero_delay.tokens != default.tokens
    assert zero_delay.audios[0].audio_array.shape != default.audios[0].audio_array.shape


def test_public_streaming_online_prefill_returns_two_ordered_pads(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    tokenizer = public_tokenizer(PINNED_V7_REALTIME_TEST)
    tokenized = tokenizer.encode_transcription(TRANSCRIPTION_REALTIME_ONLINE_EMPTY_PREFILL_DEFAULT_DELAY.build())

    assert len(tokenized.audios) == 2
    left_pad, right_pad = tokenized.audios
    assert left_pad.sampling_rate == right_pad.sampling_rate == 16_000
    assert left_pad.audio_array.shape[0] > right_pad.audio_array.shape[0]
    assert np.all(left_pad.audio_array == 0) and np.all(right_pad.audio_array == 0)


def test_changed_transcription_media_input_fails_shared_tolerance(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    tokenizer = public_tokenizer(PINNED_V7_AUDIO_TEST)
    expected = load_expected_success(
        case_id="transcription-pinned-v7-audio-instruct-no-language",
        tokenizer_configuration_id=PINNED_V7_AUDIO_TEST.configuration_id,
    )

    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    scaled = (rng.uniform(low=-0.5, high=0.5, size=2_400) * 0.8).astype(np.float32)
    sf.write(file=buffer, data=scaled, samplerate=24_000, format="WAV")

    request = TranscriptionRequest(
        audio=base64.b64encode(buffer.getvalue()).decode(),
        streaming=StreamingMode.DISABLED,
        language=None,
        target_streaming_delay_ms=None,
    )
    tokenized = tokenizer.encode_transcription(request)

    assert len(tokenized.audios) == len(expected.tokenized.audios) == 1
    assert tokenized.audios[0].sampling_rate == expected.tokenized.audios[0].sampling_rate
    with pytest.raises(AssertionError, match="audios differ at index 0"):
        assert_tokenized_equal(
            actual=tokenized,
            expected=expected.tokenized.model_copy(update={"prefix_ids": tokenized.prefix_ids}),
            atol=TOKENIZED_ATOL,
            rtol=TOKENIZED_RTOL,
        )

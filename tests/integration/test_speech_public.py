r"""Public speech workflow cases.

Each case builds a fresh request from its Python recipe, loads its bound
tokenizer configuration through the session-cached public loader and calls the
public ``encode_speech_request`` entry point. Success cases compare the complete
token ids, decoded text and returned audio against the reviewed manifest under
``tests/data/expected/<case-id>/``; error cases assert the raw decode error the
current public boundary raises. Requests are constructed before the
expected-error assertion, so a construction failure cannot satisfy a case.

All success cases use the explicitly synthetic ``synthetic-v7-speech-test``
profile: no released voice-token fidelity is claimed from synthetic mappings.
"""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.expected_results import assert_public_success, load_expected_success
from tests.integration.speech_cases import (
    SPEECH_ERROR_CASES,
    SPEECH_REFERENCE_AUDIO,
    SPEECH_REFERENCE_OVER_PRESET,
    SPEECH_SUCCESS_CASES,
    SPEECH_SYNTHETIC_PRESET,
    PublicSpeechErrorCase,
    PublicSpeechSuccessCase,
)
from tests.integration.tokenizer_configurations import SYNTHETIC_V7_SPEECH_AUDIO_TEST, TokenizerConfiguration
from tests.utils.tokenized import decode_keep


@pytest.mark.parametrize(argnames="case", argvalues=SPEECH_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_public_speech_success(
    case: PublicSpeechSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    tokenized = tokenizer.encode_speech_request(request)
    decoded_text = decode_keep(tokenizer=tokenizer, tokenized=tokenized)
    expected = load_expected_success(
        case_id=case.case_id,
        tokenizer_configuration_id=case.configuration.configuration_id,
    )
    assert_public_success(expected=expected, tokenized=tokenized, decoded_text=decoded_text)


@pytest.mark.parametrize(argnames="case", argvalues=SPEECH_ERROR_CASES, ids=lambda case: case.case_id)
def test_public_speech_decode_error(
    case: PublicSpeechErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    with pytest.raises(expected_exception=case.expected_exception, match=case.message_pattern):
        tokenizer.encode_speech_request(request)


def test_public_speech_synthetic_preset_returns_no_audio(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    tokenizer = public_tokenizer(SYNTHETIC_V7_SPEECH_AUDIO_TEST)
    tokenized = tokenizer.encode_speech_request(SPEECH_SYNTHETIC_PRESET.build())

    assert tokenized.audios == []


def test_public_speech_reference_takes_precedence_over_preset(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    tokenizer = public_tokenizer(SYNTHETIC_V7_SPEECH_AUDIO_TEST)
    reference_only = tokenizer.encode_speech_request(SPEECH_REFERENCE_AUDIO.build())
    reference_over_preset = tokenizer.encode_speech_request(SPEECH_REFERENCE_OVER_PRESET.build())

    assert reference_over_preset.tokens == reference_only.tokens
    assert len(reference_over_preset.audios) == len(reference_only.audios) == 1
    assert reference_over_preset.audios[0].sampling_rate == 24_000
    assert reference_over_preset.audios[0].audio_array.shape == reference_only.audios[0].audio_array.shape

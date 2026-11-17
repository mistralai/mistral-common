r"""Public speech decode-error workflow cases.

Each case builds a fresh request from its Python recipe, loads its bound
tokenizer configuration through the session-cached public loader, calls the
public ``encode_speech_request`` entry point and asserts the raw decode error
the current public boundary raises. Requests are constructed before the
expected-error assertion, so a construction failure cannot satisfy a case.
"""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.speech_cases import SPEECH_ERROR_CASES, PublicSpeechErrorCase
from tests.integration.tokenizer_configurations import TokenizerConfiguration


@pytest.mark.parametrize(argnames="case", argvalues=SPEECH_ERROR_CASES, ids=lambda case: case.case_id)
def test_public_speech_decode_error(
    case: PublicSpeechErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    with pytest.raises(expected_exception=case.expected_exception, match=case.message_pattern):
        tokenizer.encode_speech_request(request)

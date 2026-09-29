r"""Public chat workflow cases for pinned and synthetic v13 configurations."""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatSuccessCase
from tests.integration.chat_v13_cases import V13_SUCCESS_CASES
from tests.integration.expected_results import assert_public_success, load_expected_success
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.utils import decode_keep


def _encode_and_verify(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    tokenized: Tokenized = tokenizer.encode_chat_completion(request)
    decoded_text = decode_keep(tokenizer=tokenizer, tokenized=tokenized)
    expected = load_expected_success(
        case_id=case.case_id,
        tokenizer_configuration_id=case.configuration.configuration_id,
    )
    assert_public_success(expected=expected, tokenized=tokenized, decoded_text=decoded_text)


@pytest.mark.parametrize(argnames="case", argvalues=V13_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_public_chat_v13_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    _encode_and_verify(case=case, public_tokenizer=public_tokenizer)

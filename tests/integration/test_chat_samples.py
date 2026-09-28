r"""Public chat workflow cases for the bundled SPM sample recipes.

Replaces the legacy serialized sample tests: each case builds a fresh
``ChatCompletionRequest`` from a Python recipe, encodes it through the
public ``encode_chat_completion`` entry point and compares the complete
token ids and decoded text against the reviewed manifest. Encode-rejection
cases construct the request first and then assert the project error raised
by the public call.
"""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import (
    SAMPLE_ERROR_CASES,
    SAMPLE_SUCCESS_CASES,
    PublicChatErrorCase,
    PublicChatSuccessCase,
)
from tests.integration.expected_results import assert_public_success, load_expected_success
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.utils import decode_keep


@pytest.mark.parametrize(argnames="case", argvalues=SAMPLE_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_public_chat_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    tokenized = tokenizer.encode_chat_completion(request)
    decoded_text = decode_keep(tokenizer=tokenizer, tokenized=tokenized)
    expected = load_expected_success(
        case_id=case.case_id,
        tokenizer_configuration_id=case.configuration.configuration_id,
    )
    assert_public_success(expected=expected, tokenized=tokenized, decoded_text=decoded_text)


@pytest.mark.parametrize(argnames="case", argvalues=SAMPLE_ERROR_CASES, ids=lambda case: case.case_id)
def test_public_chat_rejection(
    case: PublicChatErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    with pytest.raises(expected_exception=case.expected_exception, match=case.message_pattern):
        tokenizer.encode_chat_completion(request)

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
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.integration.utils import encode_and_verify


@pytest.mark.parametrize(argnames="case", argvalues=SAMPLE_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_public_chat_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    encode_and_verify(case=case, public_tokenizer=public_tokenizer)


@pytest.mark.parametrize(argnames="case", argvalues=SAMPLE_ERROR_CASES, ids=lambda case: case.case_id)
def test_public_chat_rejection(
    case: PublicChatErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    with pytest.raises(expected_exception=case.expected_exception, match=case.message_pattern):
        tokenizer.encode_chat_completion(request)

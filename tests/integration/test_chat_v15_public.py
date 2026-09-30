r"""Public chat workflow cases for pinned and synthetic v15 configurations."""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatErrorCase, PublicChatSuccessCase
from tests.integration.chat_v15_cases import V15_ERROR_CASES, V15_SUCCESS_CASES
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.integration.utils import encode_and_verify


@pytest.mark.parametrize(argnames="case", argvalues=V15_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_public_chat_v15_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    encode_and_verify(case=case, public_tokenizer=public_tokenizer)


@pytest.mark.parametrize(argnames="case", argvalues=V15_ERROR_CASES, ids=lambda case: case.case_id)
def test_public_chat_v15_error(
    case: PublicChatErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)

    with pytest.raises(expected_exception=case.expected_exception, match=case.message_pattern):
        tokenizer.encode_chat_completion(request)

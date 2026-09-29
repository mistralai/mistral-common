r"""Public chat workflow cases for pinned and synthetic v13 configurations."""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatSuccessCase
from tests.integration.chat_v13_cases import V13_SUCCESS_CASES
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.integration.utils import encode_and_verify


@pytest.mark.parametrize(argnames="case", argvalues=V13_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_public_chat_v13_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    encode_and_verify(case=case, public_tokenizer=public_tokenizer)

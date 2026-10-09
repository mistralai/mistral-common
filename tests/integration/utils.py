from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatErrorCase, PublicChatSuccessCase
from tests.integration.expected_results import assert_public_success, load_expected_success
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.utils import decode_keep


def encode_and_expect_error(
    *,
    case: PublicChatErrorCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    r"""Encode a fresh public chat request and assert its expected error.

    Args:
        case: Error case pairing a recipe, tokenizer configuration and expectation.
        public_tokenizer: Session-cached loader for verified configurations.
    """
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    with pytest.raises(expected_exception=case.expected_exception, match=case.message_pattern):
        tokenizer.encode_chat_completion(request)


def encode_and_verify(
    *,
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> tuple[MistralTokenizer, Tokenized, str]:
    r"""Encode a fresh public chat request and check its reviewed full result.

    Args:
        case: Success case pairing a recipe, tokenizer configuration and manifest.
        public_tokenizer: Session-cached loader for verified configurations.

    Returns:
        The loaded tokenizer, tokenized result and decoded text for additional
        case-specific assertions.
    """
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    tokenized = tokenizer.encode_chat_completion(request)
    decoded_text = decode_keep(tokenizer=tokenizer, tokenized=tokenized)
    expected = load_expected_success(
        case_id=case.case_id,
        tokenizer_configuration_id=case.configuration.configuration_id,
    )
    assert_public_success(expected=expected, tokenized=tokenized, decoded_text=decoded_text)
    return tokenizer, tokenized, decoded_text

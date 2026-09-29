r"""Public chat workflow cases for bundled and released v7 configurations."""

from collections.abc import Callable

import pytest

from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatSuccessCase
from tests.integration.chat_v7_cases import V7_DIRECT_EQUALITY_CASES, V7_SUCCESS_CASES
from tests.integration.expected_results import assert_public_success, load_expected_success
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.utils import decode_keep

_DIRECT_RECIPES: dict[str, Callable[[], InstructRequest]] = {
    case.case_id: case.build_direct_request for case in V7_DIRECT_EQUALITY_CASES
}


def _encode_and_verify(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> tuple[MistralTokenizer, Tokenized, str]:
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


@pytest.mark.parametrize(argnames="case", argvalues=V7_SUCCESS_CASES, ids=lambda case: case.case_id)
def test_public_chat_v7_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    tokenizer, tokenized, _ = _encode_and_verify(case=case, public_tokenizer=public_tokenizer)

    if case.case_id == "chat-v7-prefixed-final":
        eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
        assert tokenized.prefix_ids is not None
        assert eos_id not in tokenized.prefix_ids
    elif case.case_id in _DIRECT_RECIPES:
        instruct_request = _DIRECT_RECIPES[case.case_id]()
        direct = tokenizer.instruct_tokenizer.encode_instruct(instruct_request)
        assert tokenized.tokens == direct.tokens

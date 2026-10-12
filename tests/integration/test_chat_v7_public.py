r"""Public chat workflow cases for bundled and released v7 configurations."""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatSuccessCase
from tests.integration.chat_v7_cases import (
    V7_DIRECT_EQUALITY_CASES,
    V7_PREFIXED_FINAL_CASE,
    V7_SUCCESS_CASES,
    V7DirectEqualityCase,
)
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.integration.utils import encode_and_verify

_V7_SUCCESS_CASES_WITHOUT_EXTRA_RELATIONS = tuple(
    case
    for case in V7_SUCCESS_CASES
    if case is not V7_PREFIXED_FINAL_CASE and all(case is not direct.case for direct in V7_DIRECT_EQUALITY_CASES)
)


@pytest.mark.parametrize(
    argnames="case",
    argvalues=_V7_SUCCESS_CASES_WITHOUT_EXTRA_RELATIONS,
    ids=lambda case: case.case_id,
)
def test_public_chat_v7_success(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    encode_and_verify(case=case, public_tokenizer=public_tokenizer)


@pytest.mark.parametrize(
    argnames="direct_case",
    argvalues=V7_DIRECT_EQUALITY_CASES,
    ids=lambda direct_case: direct_case.case.case_id,
)
def test_public_chat_v7_direct_equality(
    direct_case: V7DirectEqualityCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    tokenizer, tokenized = encode_and_verify(case=direct_case.case, public_tokenizer=public_tokenizer)
    direct = tokenizer.instruct_tokenizer.encode_instruct(direct_case.build_direct_request())

    assert tokenized.tokens == direct.tokens


def test_public_chat_v7_prefixed_final(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    tokenizer, tokenized = encode_and_verify(case=V7_PREFIXED_FINAL_CASE, public_tokenizer=public_tokenizer)
    eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id

    assert tokenized.prefix_ids is not None
    assert eos_id not in tokenized.prefix_ids

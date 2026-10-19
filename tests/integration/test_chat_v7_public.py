r"""Public chat workflow cases for bundled and released v7 configurations."""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_v7_cases import (
    V7_DIRECT_EQUALITY_CASES,
    V7_PREFIXED_FINAL_CASE,
    V7_SYSTEM_TOOLS_IMAGE_CASE,
    V7DirectEqualityCase,
)
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.integration.utils import encode_and_verify


def test_public_chat_v7_system_tools_image(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    encode_and_verify(case=V7_SYSTEM_TOOLS_IMAGE_CASE, public_tokenizer=public_tokenizer)


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

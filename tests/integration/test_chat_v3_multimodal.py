r"""Public chat workflow cases for bundled v3 multimodal configurations.

Each paired row encodes both requests through the public entry point,
compares each complete result against its reviewed manifest, and only then
asserts the retained cross-output relation. Image and ordering rows compare
their complete outputs against reviewed manifests and keep the legacy
image-span and placement relations as additional assertions.
"""

from collections.abc import Callable

import pytest

from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.image import SpecialImageIDs
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.integration.chat_cases import PublicChatSuccessCase
from tests.integration.chat_v3_cases import (
    AGREEMENT_PAIRS,
    EQUAL_TOKENS,
    IMAGE_CASES,
    LEADING_IMAGE_CASE,
    MULTI_IMAGE_ORDER_CASES,
    SWAP_PAIRS,
    TRAILING_IMAGE_CASE,
    PairedChatCase,
)
from tests.integration.expected_results import assert_public_success, load_expected_success
from tests.integration.tokenizer_configurations import TokenizerConfiguration
from tests.utils import decode_keep


def _encode_and_verify(
    case: PublicChatSuccessCase,
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> Tokenized:
    request = case.recipe.build()
    tokenizer = public_tokenizer(case.configuration)
    tokenized = tokenizer.encode_chat_completion(request)
    decoded_text = decode_keep(tokenizer=tokenizer, tokenized=tokenized)
    expected = load_expected_success(
        case_id=case.case_id,
        tokenizer_configuration_id=case.configuration.configuration_id,
    )
    assert_public_success(expected=expected, tokenized=tokenized, decoded_text=decoded_text)
    return tokenized


@pytest.mark.parametrize("pair", AGREEMENT_PAIRS, ids=lambda pair: pair.pair_id)
def test_public_chat_agreement_pair(
    pair: PairedChatCase, public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer]
) -> None:
    first = _encode_and_verify(pair.first, public_tokenizer)
    second = _encode_and_verify(pair.second, public_tokenizer)
    assert first.tokens == second.tokens, "Text-only and multimodal outputs differ for the same request"


@pytest.mark.parametrize("pair", SWAP_PAIRS, ids=lambda pair: pair.pair_id)
def test_public_chat_swap_pair(
    pair: PairedChatCase, public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer]
) -> None:
    first = _encode_and_verify(pair.first, public_tokenizer)
    second = _encode_and_verify(pair.second, public_tokenizer)
    if pair.token_relation == EQUAL_TOKENS:
        assert first.tokens == second.tokens, "Image-first and text-first outputs were expected to agree"
    else:
        assert first.tokens != second.tokens, "Appending text was expected to break the swap agreement"


@pytest.mark.parametrize("case", IMAGE_CASES, ids=lambda case: case.case_id)
def test_public_chat_image_case(
    case: PublicChatSuccessCase, public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer]
) -> None:
    _encode_and_verify(case, public_tokenizer)


def _image_tokens(width: int, height: int, special_ids: SpecialImageIDs) -> list[int]:
    image_tokens = ([special_ids.img] * width + [special_ids.img_break]) * height
    image_tokens[-1] = special_ids.img_end
    return image_tokens


def _image_tokenizer_spans(tokens: list[int], special_ids: SpecialImageIDs) -> list[list[int]]:
    spans: list[list[int]] = []
    start_idx: int | None = None
    for idx, token in enumerate(tokens):
        if start_idx is None:
            if token == special_ids.img:
                start_idx = idx
        elif token == special_ids.img_end:
            spans.append(tokens[start_idx : idx + 1])
            start_idx = None
    return spans


def _patch2_special_ids(
    case: PublicChatSuccessCase, public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer]
) -> tuple[SpecialImageIDs, MistralTokenizer]:
    tokenizer = public_tokenizer(case.configuration)
    image_encoder = tokenizer.instruct_tokenizer.image_encoder
    assert image_encoder is not None
    return image_encoder.special_ids, tokenizer


@pytest.mark.parametrize("case", MULTI_IMAGE_ORDER_CASES, ids=lambda case: case.case_id)
def test_public_chat_multi_image_order(
    case: PublicChatSuccessCase, public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer]
) -> None:
    special_ids, _ = _patch2_special_ids(case, public_tokenizer)
    tokenized = _encode_and_verify(case, public_tokenizer)
    assert _image_tokenizer_spans(tokenized.tokens, special_ids) == [
        _image_tokens(width=2, height=2, special_ids=special_ids),
        _image_tokens(width=3, height=2, special_ids=special_ids),
    ]


def test_public_chat_trailing_image_moves_first(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    special_ids, tokenizer = _patch2_special_ids(TRAILING_IMAGE_CASE, public_tokenizer)
    tokenized = _encode_and_verify(TRAILING_IMAGE_CASE, public_tokenizer)
    assert _image_tokenizer_spans(tokenized.tokens, special_ids) == [
        _image_tokens(width=2, height=2, special_ids=special_ids)
    ]
    x_token = tokenizer.instruct_tokenizer.tokenizer.encode("x", bos=False, eos=False)[0]
    assert tokenized.tokens.index(special_ids.img) < tokenized.tokens.index(x_token)


def test_public_chat_leading_image_remains_first(
    public_tokenizer: Callable[[TokenizerConfiguration], MistralTokenizer],
) -> None:
    special_ids, tokenizer = _patch2_special_ids(LEADING_IMAGE_CASE, public_tokenizer)
    tokenized = _encode_and_verify(LEADING_IMAGE_CASE, public_tokenizer)
    assert _image_tokenizer_spans(tokenized.tokens, special_ids) == [
        _image_tokens(width=2, height=2, special_ids=special_ids)
    ]
    x_token = tokenizer.instruct_tokenizer.tokenizer.encode("x", bos=False, eos=False)[0]
    assert tokenized.tokens.index(special_ids.img) < tokenized.tokens.index(x_token)

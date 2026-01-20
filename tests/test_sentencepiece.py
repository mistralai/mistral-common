import os
from pathlib import Path

import numpy as np
import pytest

from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer


@pytest.fixture(scope="module")
def tokenizer_v7() -> SentencePieceTokenizer:
    _model_path = (
        Path(os.path.abspath(__file__)).parent.parent
        / "src"
        / "mistral_common"
        / "data"
        / "mistral_instruct_tokenizer_241114.model.v7"
    )
    return SentencePieceTokenizer(model_path=_model_path)


@pytest.mark.parametrize(
    ("token", "is_control"),
    [
        ("</s>", True),
        ("a", False),
        (1, True),
        (230, True),
        (1001, False),
        (np.int64(1), True),
        (np.int64(1001), False),
    ],
)
def test_is_control(tokenizer_v7: SentencePieceTokenizer, token: str | int, is_control: bool) -> None:
    assert tokenizer_v7.is_special(token) is is_control


def test_sentencepiece_tokenizer_special_ids_property(tokenizer_v7: SentencePieceTokenizer) -> None:
    special_ids = tokenizer_v7.special_ids
    assert isinstance(special_ids, set)
    assert len(special_ids) > 0


def test_sentencepiece_tokenizer_num_specials_property(tokenizer_v7: SentencePieceTokenizer) -> None:
    num_special = tokenizer_v7.num_special_tokens
    assert isinstance(num_special, int)
    assert num_special == 748

    # Test that num_special_tokens matches the length of special_ids
    assert num_special == len(tokenizer_v7.special_ids)

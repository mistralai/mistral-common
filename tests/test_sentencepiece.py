import os
from pathlib import Path

import numpy as np
import pytest

from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer


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
def test_is_control(token: str | int, is_control: bool) -> None:
    # get current file
    _model_path = (
        Path(os.path.abspath(__file__)).parent.parent
        / "src"
        / "mistral_common"
        / "data"
        / "mistral_instruct_tokenizer_241114.model.v7"
    )
    tokenizer = SentencePieceTokenizer(model_path=_model_path)
    assert tokenizer.is_special(token) is is_control

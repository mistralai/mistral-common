from typing import Optional
from unittest.mock import patch

import huggingface_hub as huggingface_hub
import pytest

from mistral_common.tokens.tokenizers.utils import download_tokenizer_from_hf_hub


@pytest.mark.parametrize(
    ["files", "expected"],
    [
        ([], None),
        (["unvalid1.txt", "path/to/tekken.json"], "path/to/tekken.json"),
        (["unvalid1.txt", "tekken.json"], "tekken.json"),
        (["unvalid1.txt", "sentencepiece.model"], "sentencepiece.model"),
        (["unvalid1.txt", "sentencepiece.model.v1"], "sentencepiece.model.v1"),
        (["unvalid1.txt", "sentencepiece.model.v1", "sentencepiece.model.v1m1"], "sentencepiece.model.v1m1"),
        (["unvalid1.txt", "sentencepiece.model", "sentencepiece.model.v1m1"], "sentencepiece.model.v1m1"),
        (["unvalid1.txt", "unvalid2.txt"], None),
    ],
)
def test_download_tokenizer_from_hf_hub(files: list[str], expected: Optional[str]) -> None:
    with patch("huggingface_hub.HfApi.list_repo_files", return_value=files):
        if expected is None:
            with pytest.raises(ValueError):
                download_tokenizer_from_hf_hub(repo_id="mistralai/Mistral-7B-v0.1", token=True, revision=None)
        else:
            with patch("huggingface_hub.hf_hub_download", return_value=expected):
                tokenizer = download_tokenizer_from_hf_hub(
                    repo_id="mistralai/Mistral-7B-v0.1", token=True, revision=None
                )
                assert tokenizer == expected

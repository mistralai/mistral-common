import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import huggingface_hub as huggingface_hub
import pytest

from mistral_common.tokens.tokenizers.utils import download_tokenizer_from_hf_hub, list_local_hf_repo_files


def _create_temporary_hf_model_cache(repo_id: str) -> Path:
    tmp_dir = tempfile.mkdtemp()
    hub_folder = Path(tmp_dir) / "hf_hub"
    model_folder = huggingface_hub.constants.REPO_ID_SEPARATOR.join(["models", *repo_id.split("/")])

    revision_file = hub_folder / model_folder / "refs" / huggingface_hub.constants.DEFAULT_REVISION
    revision_file.parent.mkdir(parents=True, exist_ok=True)
    revision_file.write_text("RANDOM_REVISION")

    revision_folder = hub_folder / model_folder / "snapshots" / "RANDOM_REVISION"
    revision_folder.mkdir(parents=True, exist_ok=True)
    (revision_folder / "tekken.json").write_text("{'test': 'test'}")
    (revision_folder / "file2.txt").write_text("test")
    return hub_folder


@patch("huggingface_hub.constants.HF_HUB_CACHE", "/tmp/hf_cache")
def test_list_local_hf_repo_files() -> None:
    hf_cache = _create_temporary_hf_model_cache("mistralai/Mistral-7B-v0.1")
    with patch("huggingface_hub.constants.HF_HUB_CACHE", hf_cache):
        # Test without revision
        files = list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", None)
        assert files == ["file2.txt", "tekken.json"]

        # Test with a specific revision
        files = list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", "RANDOM_REVISION")
        assert files == ["file2.txt", "tekken.json"]

        # Test with non-existent revision
        files = list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", "non_existent_revision")
        assert files == []

    # Test huggingface_hub not installed
    with patch("mistral_common.tokens.tokenizers.utils._hub_installed", False):
        with pytest.raises(ImportError):
            list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", "test_revision")


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

    # Test huggingface_hub not installed
    with patch("mistral_common.tokens.tokenizers.utils._hub_installed", False):
        with pytest.raises(ImportError):
            list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", "test_revision")


@patch("huggingface_hub.HfApi.list_repo_files")
def test_download_tokenizer_from_hf_hub_without_connection(mock_list_repo_files: MagicMock) -> None:
    mock_list_repo_files.side_effect = ConnectionError()

    hf_cache = _create_temporary_hf_model_cache("mistralai/Mistral-7B-v0.1")

    # Test with local cache
    with patch("huggingface_hub.constants.HF_HUB_CACHE", hf_cache):
        tokenizer = download_tokenizer_from_hf_hub(repo_id="mistralai/Mistral-7B-v0.1", token=True, revision=None)
        assert tokenizer == str(hf_cache / "models--mistralai--Mistral-7B-v0.1/snapshots/RANDOM_REVISION/tekken.json")

    # Test without local cache
    with patch("mistral_common.tokens.tokenizers.utils.list_local_hf_repo_files", return_value=[]):
        with pytest.raises(ConnectionError):
            download_tokenizer_from_hf_hub(repo_id="mistralai/Mistral-7B-v0.1", token=True, revision=None)

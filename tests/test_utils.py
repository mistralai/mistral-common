import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import huggingface_hub as huggingface_hub
import pytest

from mistral_common.tokens.tokenizers.utils import (
    download_tokenizer_from_hf_hub,
    get_one_valid_tokenizer_file,
    list_local_hf_repo_files,
)


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
        files = sorted(list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", None))
        assert files == ["file2.txt", "tekken.json"]

        # Test with a specific revision
        files = sorted(list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", "RANDOM_REVISION"))
        assert files == ["file2.txt", "tekken.json"]

        # Test with a branch/tag revision that must be resolved via refs (e.g. "main")
        files = sorted(list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", "main"))
        assert files == ["file2.txt", "tekken.json"]

        # Test with non-existent revision
        assert list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", "non_existent_revision") == []

    # Test huggingface_hub not installed
    with patch("mistral_common.tokens.tokenizers.utils._hub_installed", False):
        with pytest.raises(ImportError):
            list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", "test_revision")


@pytest.mark.parametrize(
    ["files", "expected"],
    [
        ([], None),
        (["unvalid1.txt", "/path/to/tekken.json"], "/path/to/tekken.json"),
        (["unvalid1.txt", "tekken.json"], "tekken.json"),
        (["unvalid1.txt", "/tekken.json"], "/tekken.json"),
        (["unvalid1.txt", "sentencepiece.model"], "sentencepiece.model"),
        (["unvalid1.txt", "sentencepiece.model.v1"], "sentencepiece.model.v1"),
        (["unvalid1.txt", "/path/to/sentencepiece.model.v1", "sentencepiece.model.v1m1"], "sentencepiece.model.v1m1"),
        (
            ["unvalid1.txt", "/sentencepiece.model.v1", "/path/to/sentencepiece.model.v1m1"],
            "/path/to/sentencepiece.model.v1m1",
        ),
        (
            [
                "unvalid1.txt",
                "/sentencepiece.model.v1",
                "/path/to/sentencepiece.model.v1m1",
                "/sentencepiece.model.v1/tekken.json",
            ],
            "/sentencepiece.model.v1/tekken.json",
        ),
        (["unvalid1.txt", "sentencepiece.model", "sentencepiece.model.v1m1"], "sentencepiece.model.v1m1"),
        (["unvalid1.txt", "unvalid2.txt"], None),
    ],
)
def test_get_one_valid_tokenizer_files_abs(files: list[str], expected: str | None) -> None:
    if expected is None:
        with pytest.raises(ValueError):
            get_one_valid_tokenizer_file(files=files)
    else:
        tokenizer = get_one_valid_tokenizer_file(files=files)
        assert tokenizer == expected


@pytest.mark.parametrize(
    ["files", "expected"],
    [
        ([], None),
        (["unvalid1.txt", "path/to/tekken.json"], "path/to/tekken.json"),
        (["unvalid1.txt", "tekken.json"], "tekken.json"),
        (["unvalid1.txt", "sentencepiece.model"], "sentencepiece.model"),
        (["unvalid1.txt", "sentencepiece.model.v1"], "sentencepiece.model.v1"),
        (["unvalid1.txt", "sentencepiece.model.v1", "sentencepiece.model.v1m1"], "sentencepiece.model.v1m1"),
        (
            [
                "unvalid1.txt",
                "sentencepiece.model.v1",
                "sentencepiece.model.v1m1",
                "sentencepiece.model.v1/tekken.json",
            ],
            "sentencepiece.model.v1/tekken.json",
        ),
        (["unvalid1.txt", "sentencepiece.model", "sentencepiece.model.v1m1"], "sentencepiece.model.v1m1"),
        (["unvalid1.txt", "unvalid2.txt"], None),
    ],
)
def test_download_tokenizer_from_hf_hub(files: list[str], expected: str | None) -> None:
    with patch("huggingface_hub.HfApi.list_repo_files", return_value=files):
        if expected is None:
            with pytest.raises(ValueError):
                download_tokenizer_from_hf_hub(
                    repo_id="mistralai/Mistral-7B-v0.1",
                    token=None,
                    revision=None,
                    local_files_only=False,
                    force_download=False,
                )
        else:
            with patch("huggingface_hub.hf_hub_download", return_value=expected):
                tokenizer = download_tokenizer_from_hf_hub(
                    repo_id="mistralai/Mistral-7B-v0.1",
                    token=None,
                    revision=None,
                    local_files_only=False,
                    force_download=False,
                )
                assert tokenizer == expected

    # Test huggingface_hub not installed
    with patch("mistral_common.tokens.tokenizers.utils._hub_installed", False):
        with pytest.raises(ImportError):
            list_local_hf_repo_files("mistralai/Mistral-7B-v0.1", "test_revision")

    # Test with local_files_only and force_download
    with pytest.raises(
        ValueError, match="You cannot force the download of the tokenizer if you only want to use local files."
    ):
        download_tokenizer_from_hf_hub(
            repo_id="mistralai/Mistral-7B-v0.1", token=None, revision=None, local_files_only=True, force_download=True
        )


@patch("huggingface_hub.HfApi.list_repo_files")
def test_download_tokenizer_from_hf_hub_offline_mode(mock_list_repo_files: MagicMock) -> None:
    mock_list_repo_files.side_effect = huggingface_hub.errors.OfflineModeIsEnabled("offline mode is enabled")

    hf_cache = _create_temporary_hf_model_cache("mistralai/Mistral-7B-v0.1")

    with patch("huggingface_hub.constants.HF_HUB_CACHE", hf_cache):
        tokenizer = download_tokenizer_from_hf_hub(repo_id="mistralai/Mistral-7B-v0.1", token=None, revision="main")
        expected = (hf_cache / "models--mistralai--Mistral-7B-v0.1/snapshots/RANDOM_REVISION/tekken.json").resolve()
        assert tokenizer == str(expected)

import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import huggingface_hub
import pytest
from huggingface_hub.utils import httpx as hub_httpx

from mistral_common.exceptions import MistralCommonException
from mistral_common.tokens.tokenizers.utils import (
    download_tokenizer_from_hf_hub,
    get_one_valid_tokenizer_file,
    list_local_hf_repo_files,
)

_REPO_ID = "mistralai/Mistral-7B-v0.1"
_COMMIT = "RANDOM_REVISION"

_NETWORK_ERRORS: list[Exception] = [
    hub_httpx.ConnectError("no connection"),
    hub_httpx.ReadTimeout("timed out"),
    huggingface_hub.errors.OfflineModeIsEnabled("offline mode is enabled"),
]


def _cached_tokenizer_path(cache_dir: Path) -> Path:
    repo_folder = huggingface_hub.constants.REPO_ID_SEPARATOR.join(["models", *_REPO_ID.split("/")])
    return cache_dir / repo_folder / "snapshots" / _COMMIT / "tekken.json"


def _populate_hf_cache(cache_dir: Path) -> None:
    tokenizer_path = _cached_tokenizer_path(cache_dir=cache_dir)
    tokenizer_path.parent.mkdir(parents=True)
    tokenizer_path.write_text("{}")
    ref_file = tokenizer_path.parents[2] / "refs" / huggingface_hub.constants.DEFAULT_REVISION
    ref_file.parent.mkdir()
    ref_file.write_text(_COMMIT)


@pytest.fixture
def custom_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "custom_cache"
    _populate_hf_cache(cache_dir=cache_dir)
    return cache_dir


@pytest.fixture
def empty_default_cache(tmp_path: Path) -> Iterator[Path]:
    default_cache = tmp_path / "default_cache"
    default_cache.mkdir()
    with patch("huggingface_hub.constants.HF_HUB_CACHE", str(default_cache)):
        yield default_cache


@pytest.mark.usefixtures("empty_default_cache")
@pytest.mark.parametrize("cache_dir_type", [str, Path])
def test_list_local_hf_repo_files_reads_cache_dir(custom_cache: Path, cache_dir_type: type) -> None:
    files = list_local_hf_repo_files(repo_id=_REPO_ID, revision=None, cache_dir=cache_dir_type(custom_cache))
    assert files == ["tekken.json"]


@pytest.mark.usefixtures("custom_cache")
def test_list_local_hf_repo_files_defaults_to_hf_hub_cache(empty_default_cache: Path) -> None:
    assert list_local_hf_repo_files(repo_id=_REPO_ID, revision=None) == []

    _populate_hf_cache(cache_dir=empty_default_cache)
    assert list_local_hf_repo_files(repo_id=_REPO_ID, revision=None) == ["tekken.json"]


@pytest.mark.usefixtures("empty_default_cache")
def test_download_tokenizer_from_hf_hub_local_files_only_uses_cache_dir(custom_cache: Path) -> None:
    tokenizer_path = download_tokenizer_from_hf_hub(repo_id=_REPO_ID, cache_dir=custom_cache, local_files_only=True)
    assert Path(tokenizer_path) == _cached_tokenizer_path(cache_dir=custom_cache)


@pytest.mark.usefixtures("empty_default_cache")
@pytest.mark.parametrize("network_error", _NETWORK_ERRORS)
@patch("huggingface_hub.HfApi.list_repo_files")
def test_download_tokenizer_from_hf_hub_offline_fallback_uses_cache_dir(
    mock_list_repo_files: MagicMock, network_error: Exception, custom_cache: Path
) -> None:
    mock_list_repo_files.side_effect = network_error

    tokenizer_path = download_tokenizer_from_hf_hub(repo_id=_REPO_ID, cache_dir=custom_cache)
    assert Path(tokenizer_path) == _cached_tokenizer_path(cache_dir=custom_cache)


@pytest.mark.usefixtures("empty_default_cache")
def test_download_tokenizer_from_hf_hub_falls_back_to_cache_on_real_connection_error(
    custom_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("huggingface_hub.constants.ENDPOINT", "http://127.0.0.1:9")

    tokenizer_path = download_tokenizer_from_hf_hub(repo_id=_REPO_ID, cache_dir=custom_cache)
    assert Path(tokenizer_path) == _cached_tokenizer_path(cache_dir=custom_cache)


@pytest.mark.parametrize("network_error", _NETWORK_ERRORS)
@patch("huggingface_hub.HfApi.list_repo_files")
def test_download_tokenizer_from_hf_hub_force_download_raises_network_error(
    mock_list_repo_files: MagicMock, network_error: Exception
) -> None:
    mock_list_repo_files.side_effect = network_error

    with pytest.raises(type(network_error), match=str(network_error)):
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID, force_download=True)


@pytest.mark.usefixtures("empty_default_cache")
@pytest.mark.parametrize("network_error", _NETWORK_ERRORS)
@patch("huggingface_hub.HfApi.list_repo_files")
def test_download_tokenizer_from_hf_hub_offline_without_cache_raises(
    mock_list_repo_files: MagicMock, network_error: Exception
) -> None:
    mock_list_repo_files.side_effect = network_error

    with pytest.raises(FileNotFoundError, match="Could not connect to the Hugging Face Hub and no local files"):
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID)

    with pytest.raises(FileNotFoundError, match="No local files found for the repo ID"):
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID, local_files_only=True)


@pytest.mark.parametrize("force_download", [False, True])
@patch("huggingface_hub.HfApi.list_repo_files")
def test_download_tokenizer_from_hf_hub_unexpected_error_raises_general_error(
    mock_list_repo_files: MagicMock, force_download: bool
) -> None:
    mock_list_repo_files.side_effect = RuntimeError("boom")

    with pytest.raises(MistralCommonException, match="Something went wrong") as exc_info:
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID, force_download=force_download)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.usefixtures("empty_default_cache")
@pytest.mark.parametrize(
    ("revision", "expected"),
    [
        (None, ["file2.txt", "tekken.json"]),
        (_COMMIT, ["file2.txt", "tekken.json"]),
        ("main", ["file2.txt", "tekken.json"]),
        ("release-branch", ["file2.txt", "tekken.json"]),
        ("non_existent_revision", []),
        ("", []),
    ],
    ids=["default-ref", "commit-hash", "default-branch-via-refs", "other-branch-via-refs", "unknown-revision", "empty"],
)
def test_list_local_hf_repo_files_resolves_revisions(
    custom_cache: Path, revision: str | None, expected: list[str]
) -> None:
    snapshot = _cached_tokenizer_path(cache_dir=custom_cache).parent
    (snapshot / "file2.txt").write_text("test")
    (_cached_tokenizer_path(cache_dir=custom_cache).parents[2] / "refs" / "release-branch").write_text(_COMMIT)
    assert sorted(list_local_hf_repo_files(repo_id=_REPO_ID, revision=revision, cache_dir=custom_cache)) == expected


def test_list_local_hf_repo_files_requires_huggingface_hub() -> None:
    with patch("mistral_common.tokens.tokenizers.utils._hub_installed", False):
        with pytest.raises(ImportError, match="huggingface_hub"):
            list_local_hf_repo_files(repo_id=_REPO_ID, revision=None)


def test_download_tokenizer_from_hf_hub_requires_huggingface_hub() -> None:
    with patch("mistral_common.tokens.tokenizers.utils._hub_installed", False):
        with pytest.raises(ImportError, match="huggingface_hub"):
            download_tokenizer_from_hf_hub(repo_id=_REPO_ID)


@pytest.mark.parametrize(
    ("files", "expected"),
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
        (["unvalid1.txt", "sentencepiece.model.v99"], None),
    ],
    ids=[
        "empty",
        "tekken-with-dir",
        "tekken",
        "tekken-rooted",
        "spm",
        "spm-v1",
        "spm-v1-and-v1m1",
        "spm-v1-and-v1m1-absolute",
        "tekken-in-subdir-wins",
        "spm-and-v1m1",
        "no-valid-file",
        "unknown-version-suffix",
    ],
)
def test_get_one_valid_tokenizer_file_selects(files: list[str], expected: str | None) -> None:
    if expected is None:
        with pytest.raises(ValueError, match="No tokenizer file found"):
            get_one_valid_tokenizer_file(files=files)
    else:
        assert get_one_valid_tokenizer_file(files=files) == expected


def test_get_one_valid_tokenizer_file_warns_when_multiple_non_tekken_files(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING, logger="mistral_common.tokens.tokenizers.utils"):
        selected = get_one_valid_tokenizer_file(files=["sentencepiece.model", "sentencepiece.model.v1m1"])
    assert selected == "sentencepiece.model.v1m1"
    assert "Multiple valid tokenizer files found. Using sentencepiece.model.v1m1." in caplog.text


def test_get_one_valid_tokenizer_file_prefers_tekken_without_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING, logger="mistral_common.tokens.tokenizers.utils"):
        selected = get_one_valid_tokenizer_file(files=["sentencepiece.model", "tekken.json"])
    assert selected == "tekken.json"
    assert "Multiple valid tokenizer files found" not in caplog.text


@pytest.mark.parametrize(
    ("files", "expected"),
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
    ids=[
        "empty",
        "tekken-with-dir",
        "tekken",
        "spm",
        "spm-v1",
        "spm-v1-and-v1m1",
        "tekken-in-subdir-wins",
        "spm-and-v1m1",
        "no-valid-file",
    ],
)
@patch("huggingface_hub.hf_hub_download")
@patch("huggingface_hub.HfApi.list_repo_files")
def test_download_tokenizer_from_hf_hub_selects_file_and_forwards_arguments(
    mock_list_repo_files: MagicMock, mock_hf_hub_download: MagicMock, files: list[str], expected: str | None
) -> None:
    mock_list_repo_files.return_value = files
    if expected is None:
        with pytest.raises(ValueError, match="No valid tokenizer file found in the repo"):
            download_tokenizer_from_hf_hub(
                repo_id=_REPO_ID, token=None, revision=None, local_files_only=False, force_download=False
            )
        mock_hf_hub_download.assert_not_called()
        return
    mock_hf_hub_download.return_value = expected
    tokenizer = download_tokenizer_from_hf_hub(
        repo_id=_REPO_ID, token=None, revision=None, local_files_only=False, force_download=False
    )
    assert tokenizer == expected
    mock_list_repo_files.assert_called_once_with(_REPO_ID, revision=None, token=None)
    mock_hf_hub_download.assert_called_once_with(
        repo_id=_REPO_ID,
        cache_dir=None,
        filename=expected,
        token=None,
        revision=None,
        local_files_only=False,
        force_download=False,
    )


def test_download_tokenizer_from_hf_hub_force_download_with_local_files_only_raises() -> None:
    with pytest.raises(
        ValueError, match="You cannot force the download of the tokenizer if you only want to use local files."
    ):
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID, local_files_only=True, force_download=True)


@pytest.mark.usefixtures("custom_cache")
@patch("huggingface_hub.HfApi.list_repo_files")
def test_download_tokenizer_from_hf_hub_offline_fallback_uses_default_cache(
    mock_list_repo_files: MagicMock, empty_default_cache: Path
) -> None:
    mock_list_repo_files.side_effect = huggingface_hub.errors.OfflineModeIsEnabled("offline mode is enabled")
    _populate_hf_cache(cache_dir=empty_default_cache)

    tokenizer_path = download_tokenizer_from_hf_hub(repo_id=_REPO_ID, token=None, revision="main")
    assert Path(tokenizer_path) == _cached_tokenizer_path(cache_dir=empty_default_cache).resolve()

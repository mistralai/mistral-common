from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import huggingface_hub
import pytest
from huggingface_hub.utils import httpx as hub_httpx

from mistral_common.exceptions import MistralCommonException
from mistral_common.tokens.tokenizers.utils import download_tokenizer_from_hf_hub, list_local_hf_repo_files

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

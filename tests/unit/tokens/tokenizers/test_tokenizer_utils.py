from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import huggingface_hub
import pytest
import requests

from mistral_common.tokens.tokenizers.utils import download_tokenizer_from_hf_hub, list_local_hf_repo_files

_REPO_ID = "mistralai/Mistral-7B-v0.1"
_COMMIT = "RANDOM_REVISION"


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
@pytest.mark.parametrize(
    "network_error",
    [
        requests.ConnectionError("no connection"),
        huggingface_hub.errors.OfflineModeIsEnabled("offline mode is enabled"),
    ],
)
@patch("huggingface_hub.HfApi.list_repo_files")
def test_download_tokenizer_from_hf_hub_offline_fallback_uses_cache_dir(
    mock_list_repo_files: MagicMock, network_error: Exception, custom_cache: Path
) -> None:
    mock_list_repo_files.side_effect = network_error

    tokenizer_path = download_tokenizer_from_hf_hub(repo_id=_REPO_ID, cache_dir=custom_cache)
    assert Path(tokenizer_path) == _cached_tokenizer_path(cache_dir=custom_cache)

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import huggingface_hub
import pytest
from huggingface_hub.utils import httpx as hub_httpx

from mistral_common.exceptions import MistralCommonException
from mistral_common.tokens.tokenizers.utils import (
    _list_remote_hf_repo_files,
    download_tokenizer_from_hf_hub,
    list_local_hf_repo_files,
)
from mistral_common.utils.download import DownloadTimeout

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
@patch("mistral_common.tokens.tokenizers.utils._list_remote_hf_repo_files")
def test_download_tokenizer_from_hf_hub_offline_fallback_uses_cache_dir(
    mock_list_remote: MagicMock, network_error: Exception, custom_cache: Path
) -> None:
    mock_list_remote.side_effect = network_error

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
@patch("mistral_common.tokens.tokenizers.utils._list_remote_hf_repo_files")
def test_download_tokenizer_from_hf_hub_force_download_raises_network_error(
    mock_list_remote: MagicMock, network_error: Exception
) -> None:
    mock_list_remote.side_effect = network_error

    with pytest.raises(type(network_error), match=str(network_error)):
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID, force_download=True)


@pytest.mark.usefixtures("empty_default_cache")
@pytest.mark.parametrize("network_error", _NETWORK_ERRORS)
@patch("mistral_common.tokens.tokenizers.utils._list_remote_hf_repo_files")
def test_download_tokenizer_from_hf_hub_offline_without_cache_raises(
    mock_list_remote: MagicMock, network_error: Exception
) -> None:
    mock_list_remote.side_effect = network_error

    with pytest.raises(FileNotFoundError, match="Could not connect to the Hugging Face Hub and no local files"):
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID)

    with pytest.raises(FileNotFoundError, match="No local files found for the repo ID"):
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID, local_files_only=True)


@pytest.mark.parametrize("force_download", [False, True])
@patch("mistral_common.tokens.tokenizers.utils._list_remote_hf_repo_files")
def test_download_tokenizer_from_hf_hub_unexpected_error_raises_general_error(
    mock_list_remote: MagicMock, force_download: bool
) -> None:
    mock_list_remote.side_effect = RuntimeError("boom")

    with pytest.raises(MistralCommonException, match="Something went wrong") as exc_info:
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID, force_download=force_download)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.usefixtures("empty_default_cache")
def test_download_tokenizer_from_hf_hub_passes_timeout_to_hub_requests(custom_cache: Path) -> None:
    with (
        patch(
            "mistral_common.tokens.tokenizers.utils._list_remote_hf_repo_files", return_value=["tekken.json"]
        ) as mock_list_remote,
        patch(
            "huggingface_hub.hf_hub_download", return_value=str(_cached_tokenizer_path(cache_dir=custom_cache))
        ) as mock_download,
    ):
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID, cache_dir=custom_cache, timeout=7)

    assert mock_list_remote.call_args.kwargs.get("timeout") == 7.0
    assert mock_download.call_args.kwargs.get("etag_timeout") == 7.0


@pytest.mark.usefixtures("empty_default_cache")
def test_download_tokenizer_from_hf_hub_default_timeout(custom_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mistral_common.utils.download import DOWNLOAD_TIMEOUT_ENV_KEY

    monkeypatch.delenv(DOWNLOAD_TIMEOUT_ENV_KEY, raising=False)
    with (
        patch(
            "mistral_common.tokens.tokenizers.utils._list_remote_hf_repo_files", return_value=["tekken.json"]
        ) as mock_list_remote,
        patch(
            "huggingface_hub.hf_hub_download", return_value=str(_cached_tokenizer_path(cache_dir=custom_cache))
        ) as mock_download,
    ):
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID, cache_dir=custom_cache)

    assert mock_list_remote.call_args.kwargs.get("timeout") == 10.0
    assert mock_download.call_args.kwargs.get("etag_timeout") == 10.0


@pytest.mark.parametrize("timeout", [0, -1.0, "abc"])
def test_download_tokenizer_from_hf_hub_rejects_invalid_timeout(timeout: DownloadTimeout) -> None:
    with (
        patch("mistral_common.tokens.tokenizers.utils._list_remote_hf_repo_files") as mock_list_remote,
        pytest.raises(ValueError, match="timeout must be a positive finite number"),
    ):
        download_tokenizer_from_hf_hub(repo_id=_REPO_ID, timeout=timeout)
    mock_list_remote.assert_not_called()


def _fake_tree_response(entries: list[dict], next_url: str | None = None) -> MagicMock:
    response = MagicMock()
    response.json.return_value = entries
    response.links = {"next": {"url": next_url}} if next_url else {}
    return response


@patch("mistral_common.tokens.tokenizers.utils.http_backoff")
@patch("mistral_common.tokens.tokenizers.utils.get_session")
def test_list_remote_hf_repo_files_passes_timeout_to_every_page(
    mock_get_session: MagicMock, mock_backoff: MagicMock
) -> None:
    mock_get_session.return_value.get.return_value = _fake_tree_response(
        [{"path": "tekken.json", "type": "file"}], next_url="https://next.page"
    )
    mock_backoff.return_value = _fake_tree_response(
        [{"path": "folder", "type": "directory"}, {"path": "other.model", "type": "file"}]
    )

    files = _list_remote_hf_repo_files(repo_id=_REPO_ID, revision=None, token=None, timeout=7.0)

    assert files == ["tekken.json", "other.model"]
    assert mock_get_session.return_value.get.call_args.kwargs.get("timeout") == 7.0
    assert mock_backoff.call_args.kwargs.get("timeout") == 7.0

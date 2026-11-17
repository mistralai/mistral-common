import base64
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests
from PIL import Image, UnidentifiedImageError

from mistral_common.exceptions import ImageDecodeException
from mistral_common.image import download_image, maybe_load_image_from_str_or_bytes
from mistral_common.protocol.instruct.chunk import ImageChunk, ImageURLChunk
from mistral_common.tokens.tokenizers.image import image_from_chunk
from mistral_common.types import TimeoutType
from mistral_common.utils.download import (
    DOWNLOAD_TIMEOUT_ENV_KEY,
    resolve_env_download_timeout,
)


def _create_test_image(size: tuple[int, int], color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", size, color)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _mock_png_response(image: Image.Image | None = None) -> Mock:
    mock_response = Mock()
    mock_response.content = _png_bytes(image if image is not None else _create_test_image((8, 8)))
    mock_response.raise_for_status = Mock()
    return mock_response


@pytest.mark.parametrize(
    "load",
    [
        pytest.param(maybe_load_image_from_str_or_bytes, id="maybe_load_image_from_str_or_bytes"),
        pytest.param(lambda image: ImageChunk(image=image), id="ImageChunk"),
    ],
)
@pytest.mark.parametrize(
    ("image", "match"),
    [
        pytest.param("not base64 at all!!", "starting with 'not base64 at all!!'", id="not-base64"),
        pytest.param("aGVsbG8gd29ybGQ=", "starting with 'aGVsbG8gd29ybGQ='", id="not-an-image"),
        pytest.param(b"notanimage", "from bytes", id="bytes-not-an-image"),
        pytest.param(42, "got int", id="int"),
        pytest.param(None, "got NoneType", id="none"),
    ],
)
def test_load_image_rejects_invalid_input(load: Callable[[object], object], image: object, match: str) -> None:
    # pydantic's ValidationError, raised by ImageChunk, subclasses ValueError.
    with pytest.raises(ValueError, match=match):
        load(image)


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("data:image/png;base64", id="no-payload-separator"),
        pytest.param("data:image/png;base64,", id="empty-payload"),
    ],
)
def test_image_from_chunk_data_url_without_payload(url: str) -> None:
    with pytest.raises(ValueError, match="Invalid base64 data URL"):
        image_from_chunk(chunk=ImageURLChunk(image_url=url))


def test_image_from_chunk_bare_file_name_is_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _create_test_image((8, 8)).save(tmp_path / "file.png")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Unsupported image url scheme"):
        image_from_chunk(chunk=ImageURLChunk(image_url="file.png"))


def test_image_from_chunk_data_url_decodes_image() -> None:
    data_url = f"data:image/png;base64,{base64.b64encode(_png_bytes(_create_test_image((8, 8)))).decode('ascii')}"

    image = image_from_chunk(chunk=ImageURLChunk(image_url=data_url))

    assert image.size == (8, 8)


def test_image_download_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DOWNLOAD_TIMEOUT_ENV_KEY, raising=False)
    assert resolve_env_download_timeout() == 10.0


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("invalid", id="invalid"),
        pytest.param("0", id="zero"),
        pytest.param("-1", id="negative"),
        pytest.param("inf", id="inf"),
        pytest.param("-inf", id="-inf"),
        pytest.param("nan", id="nan"),
    ],
)
def test_image_download_timeout_rejects_invalid_env(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(DOWNLOAD_TIMEOUT_ENV_KEY, value)

    with pytest.raises(ValueError, match="expected a positive finite number of seconds"):
        resolve_env_download_timeout()


def test_download_image_passes_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DOWNLOAD_TIMEOUT_ENV_KEY, raising=False)
    mock_response = _mock_png_response()

    with patch("mistral_common.image.requests.get", return_value=mock_response) as mock_get:
        download_image(url="https://example.com/image.png", timeout=10.0)
    assert mock_get.call_args.kwargs.get("timeout") == 10.0

    timeout_error = requests.exceptions.ReadTimeout("timed out")
    with (
        patch("mistral_common.image.requests.get", side_effect=timeout_error),
        pytest.raises(
            requests.exceptions.Timeout,
            match=r"timed out after 10\.0 seconds\. Pass a larger `timeout` or set the environment variable "
            rf"`{DOWNLOAD_TIMEOUT_ENV_KEY}` to increase the timeout\.",
        ) as exc_info,
    ):
        download_image(url="https://example.com/image.png", timeout=10.0)
    assert exc_info.value.__cause__ is timeout_error


def test_download_image_preserves_request_error() -> None:
    request_error = requests.exceptions.ConnectionError("connection failed")

    with (
        patch("mistral_common.image.requests.get", side_effect=request_error),
        pytest.raises(requests.exceptions.ConnectionError) as exc_info,
    ):
        download_image(url="https://example.com/image.png", timeout=10.0)

    assert exc_info.value is request_error


def test_download_image_preserves_image_error() -> None:
    image_error = UnidentifiedImageError("invalid image")

    with (
        patch("mistral_common.image.requests.get", return_value=_mock_png_response()),
        patch("mistral_common.image.Image.open", side_effect=image_error),
        pytest.raises(ImageDecodeException) as exc_info,
    ):
        download_image(url="https://example.com/image.png", timeout=10.0)

    assert exc_info.value.__cause__ is image_error


def test_download_image_uses_env_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DOWNLOAD_TIMEOUT_ENV_KEY, "30")

    with patch("mistral_common.image.requests.get", return_value=_mock_png_response()) as mock_get:
        download_image(url="https://example.com/image.png", timeout=None)
    assert mock_get.call_args.kwargs.get("timeout") == 30.0


def test_download_image_explicit_timeout_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DOWNLOAD_TIMEOUT_ENV_KEY, "30")

    with patch("mistral_common.image.requests.get", return_value=_mock_png_response()) as mock_get:
        download_image(url="https://example.com/image.png", timeout=5.0)
    assert mock_get.call_args.kwargs.get("timeout") == 5.0


@pytest.mark.parametrize(
    "timeout",
    [
        pytest.param(5, id="int"),
        pytest.param("2.5", id="numeric-string"),
        pytest.param(7.0, id="float"),
    ],
)
def test_download_image_accepts_numbers_as_timeout(timeout: TimeoutType) -> None:
    with patch("mistral_common.image.requests.get", return_value=_mock_png_response()) as mock_get:
        download_image(url="https://example.com/image.png", timeout=timeout)
    assert mock_get.call_args.kwargs.get("timeout") == float(timeout)


@pytest.mark.parametrize(
    "timeout",
    [
        pytest.param(0, id="zero-int"),
        pytest.param(0.0, id="zero-float"),
        pytest.param(-1.0, id="negative"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="-inf"),
        pytest.param(float("nan"), id="nan"),
        pytest.param("abc", id="non-numeric-string"),
        pytest.param(None.__class__, id="type"),
    ],
)
def test_download_image_rejects_invalid_timeout(timeout: object) -> None:
    with (
        patch("mistral_common.image.requests.get") as mock_get,
        pytest.raises(ValueError, match="timeout must be a positive finite number, got timeout="),
    ):
        download_image(url="https://example.com/image.png", timeout=timeout)  # type: ignore[arg-type]
    mock_get.assert_not_called()


_URL1 = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
_URL2 = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"


def _mock_get_by_url(url_to_image: dict[str, Image.Image]) -> Any:
    def mock_get(url: str, headers: Any = None, timeout: Any = None) -> Any:
        if url not in url_to_image:
            raise requests.exceptions.RequestException("Download failed")
        mock_response = Mock()
        mock_response.content = _png_bytes(url_to_image[url])
        mock_response.raise_for_status = Mock()
        return mock_response

    return mock_get


@pytest.mark.parametrize(
    ("url", "size", "color"),
    [
        pytest.param(_URL1, (500, 300), (128, 128, 128), id="url1"),
        pytest.param(_URL2, (400, 600), (100, 150, 200), id="url2"),
    ],
)
def test_download_image_returns_source_image(url: str, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    source_image = _create_test_image(size, color=color)

    with patch("mistral_common.image.requests.get", side_effect=_mock_get_by_url({url: source_image})):
        image = download_image(url=url, timeout=10.0)

    assert image.size == size
    assert np.asarray(image).tolist() == np.asarray(source_image).tolist()


def test_download_image_http_error() -> None:
    response = requests.Response()
    response.status_code = 404
    response._content = b"Not found"

    with patch("mistral_common.image.requests.get", return_value=response):
        with pytest.raises(requests.exceptions.HTTPError, match="404 Client Error"):
            download_image(url="https://invalid.url/image.jpg", timeout=10.0)

import base64
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests
from PIL import Image, UnidentifiedImageError

from mistral_common.exceptions import ImageDecodeException
from mistral_common.image import (
    _IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY,
    _resolve_env_image_download_timeout,
    download_image,
)
from mistral_common.protocol.instruct.chunk import (
    ImageChunk,
    ImageURLChunk,
    TextChunk,
)
from mistral_common.tokens.tokenizers.image import (
    DATASET_MEAN,
    DATASET_STD,
    ImageConfig,
    ImageEncoder,
    SpecialImageIDs,
    image_from_chunk,
    normalize,
    transform_image,
)


def _create_test_image(size: tuple[int, int], color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", size, color)


@pytest.fixture
def special_token_ids() -> SpecialImageIDs:
    return SpecialImageIDs(img=0, img_break=1, img_end=2)


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
def test_image_to_num_tokens(spatial_merge_size: int, special_token_ids: SpecialImageIDs) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size,
        max_image_size=128,
        spatial_merge_size=spatial_merge_size,
    )
    image_encoder = ImageEncoder(image_config, special_token_ids)

    for size, exp in [(4, 1), (16, 1), (128, 8), (512, 8), (2048, 8)]:
        img = Image.new("RGB", (size, size), "red")
        assert image_encoder._image_to_num_tokens(img) == (exp, exp)

    for size1, size2, exp1, exp2 in [
        (4, 2, 1, 1),
        (8, 16, 1, 1),
        (128, 64, 8, 4),
        (512, 1024, 4, 8),
    ]:
        img = Image.new("RGB", (size1, size2), "red")
        assert image_encoder._image_to_num_tokens(img) == (exp1, exp2)


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize("size", [(1, 512), (4, 10000), (10000, 4), (2, 4096)])
def test_image_to_num_tokens_extreme_aspect_ratio(
    special_token_ids: SpecialImageIDs, size: tuple[int, int], spatial_merge_size: int
) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size,
        max_image_size=128,
        spatial_merge_size=spatial_merge_size,
    )
    image_encoder = ImageEncoder(image_config, special_token_ids)

    img = Image.new("RGB", size, "red")
    w_tokens, h_tokens = image_encoder._image_to_num_tokens(img)
    assert w_tokens >= 1 and h_tokens >= 1
    encoding = image_encoder(ImageChunk(image=img))
    assert len(encoding.tokens) == (w_tokens + 1) * h_tokens


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
def test_download_image(spatial_merge_size: int, special_token_ids: SpecialImageIDs) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size,
        max_image_size=128,
        spatial_merge_size=spatial_merge_size,
    )
    image_encoder = ImageEncoder(image_config, special_token_ids)

    test_image1 = _create_test_image((500, 300), color=(128, 128, 128))
    test_image2 = _create_test_image((400, 600), color=(100, 150, 200))

    def mock_get(url: str, headers: Any = None, timeout: Any = None) -> Any:
        mock_response = Mock()

        if url == url1:
            img_byte_arr = BytesIO()
            test_image1.save(img_byte_arr, format="PNG")
            mock_response.content = img_byte_arr.getvalue()
        elif url == url2:
            img_byte_arr = BytesIO()
            test_image2.save(img_byte_arr, format="PNG")
            mock_response.content = img_byte_arr.getvalue()
        else:
            raise requests.exceptions.RequestException("Download failed")

        mock_response.raise_for_status = Mock()
        return mock_response

    url1 = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
    url2 = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"

    with patch("mistral_common.image.requests.get", side_effect=mock_get):
        for url, expected_image in [(url1, test_image1), (url2, test_image2)]:
            content = ImageURLChunk(image_url=url)
            result = image_encoder(content)

            assert result.image is not None, "Image should be processed successfully"

    # Test request error.
    invalid_url = "https://invalid.url/image.jpg"
    response = requests.Response()
    response.status_code = 404
    response._content = b"Not found"

    with patch("mistral_common.image.requests.get", return_value=response):
        with pytest.raises(requests.exceptions.HTTPError, match="404 Client Error"):
            content = ImageURLChunk(image_url=invalid_url)
            image_encoder(content)


@pytest.mark.parametrize("url", ["data:image/png;base64", "data:image/png;base64,"])
def test_image_from_chunk_data_url_without_payload(url: str) -> None:
    with pytest.raises(ValueError, match="expected a base64 payload"):
        image_from_chunk(chunk=ImageURLChunk(image_url=url))


def test_image_from_chunk_bare_file_name_is_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _create_test_image((8, 8)).save(tmp_path / "file.png")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Unsupported image url scheme"):
        image_from_chunk(chunk=ImageURLChunk(image_url="file.png"))


def test_image_from_chunk_file_uri(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _create_test_image((8, 8)).save(image_path)

    image = image_from_chunk(chunk=ImageURLChunk(image_url=f"file://{image_path}"))
    assert image.size == (8, 8)


def _mock_png_response() -> Mock:
    img_byte_arr = BytesIO()
    _create_test_image((8, 8)).save(img_byte_arr, format="PNG")
    mock_response = Mock()
    mock_response.content = img_byte_arr.getvalue()
    mock_response.raise_for_status = Mock()
    return mock_response


def test_image_download_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY, raising=False)
    assert _resolve_env_image_download_timeout() == 10.0


@pytest.mark.parametrize("value", ["invalid", "0", "-1", "inf", "-inf", "nan"])
def test_image_download_timeout_rejects_invalid_env(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY, value)

    with pytest.raises(ValueError, match="expected a positive finite number of seconds"):
        _resolve_env_image_download_timeout()


def test_download_image_passes_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY, raising=False)
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
            rf"`{_IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY}` to increase the timeout\.",
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
    monkeypatch.setenv(_IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY, "30")

    with patch("mistral_common.image.requests.get", return_value=_mock_png_response()) as mock_get:
        download_image(url="https://example.com/image.png", timeout=None)
    assert mock_get.call_args.kwargs.get("timeout") == 30.0


def test_download_image_explicit_timeout_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY, "30")

    with patch("mistral_common.image.requests.get", return_value=_mock_png_response()) as mock_get:
        download_image(url="https://example.com/image.png", timeout=5.0)
    assert mock_get.call_args.kwargs.get("timeout") == 5.0


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_download_image_rejects_invalid_timeout(timeout: float) -> None:
    with (
        patch("mistral_common.image.requests.get") as mock_get,
        pytest.raises(ValueError, match="timeout must be a positive finite float, got timeout="),
    ):
        download_image(url="https://example.com/image.png", timeout=timeout)
    mock_get.assert_not_called()


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
def test_image_encoder(spatial_merge_size: int, special_token_ids: SpecialImageIDs) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size,
        max_image_size=128,
        spatial_merge_size=spatial_merge_size,
    )
    image_encoder = ImageEncoder(image_config, special_token_ids)

    size = 386
    img = Image.new("RGB", (size, size), "red")
    img_chunk = ImageChunk(image=img)
    text_chunk = TextChunk(text="")

    with pytest.raises(AttributeError):
        image_encoder(text_chunk)  # type: ignore[arg-type]

    output = image_encoder(img_chunk)
    tokens, image = output.tokens, output.image

    w, h = image_encoder._image_to_num_tokens(img)
    # max image size 128
    assert image.shape == (3, 128, 128)
    assert (
        w * image_config.image_patch_size * spatial_merge_size,
        h * image_config.image_patch_size * spatial_merge_size,
    ) == (128, 128)
    assert len(tokens) == (w + 1) * h

    size = 111  # nearest multiple of sixteen lower than 128 is 112
    img = Image.new("RGB", (size, size), "red")
    img_chunk = ImageChunk(image=img)
    text_chunk = TextChunk(text="")

    with pytest.raises(AttributeError):
        image_encoder(text_chunk)  # type: ignore[arg-type]

    output = image_encoder(img_chunk)
    tokens, image = output.tokens, output.image
    assert image.shape == (3, 112, 112)
    w, h = image_encoder._image_to_num_tokens(img)
    assert (
        w * image_config.image_patch_size * spatial_merge_size,
        h * image_config.image_patch_size * spatial_merge_size,
    ) == (112, 112)
    assert len(tokens) == (w + 1) * h


@pytest.mark.parametrize(
    "size, spatial_merge_size",
    [
        ((200, 311), 1),
        ((300, 212), 1),
        ((251, 1374), 1),
        ((1475, 477), 1),
        ((1344, 1544), 1),
        ((2133, 3422), 1),
        ((200, 311), 2),
        ((300, 212), 2),
        ((251, 1374), 2),
        ((1475, 477), 2),
        ((1344, 1544), 2),
        ((2133, 3422), 2),
    ],
)
def test_image_processing(special_token_ids: SpecialImageIDs, size: tuple[int, int], spatial_merge_size: int) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size,
        max_image_size=1024,
        spatial_merge_size=spatial_merge_size,
    )
    image_encoder = ImageEncoder(image_config, special_token_ids)

    # all images with w,h >= 1024 should be resized to 1024
    # else round to nearest multiple of 16
    # all while keeping the aspect ratio
    EXP_IMG_SIZES = {
        (200, 311): (208, 320),
        (300, 212): (304, 224),
        (251, 1374): (192, 1024),
        (1475, 477): (1024, 336),
        (1344, 1544): (896, 1024),
        (2133, 3422): (640, 1024),
    }
    # Expected sums for gray test images (RGB: 128, 128, 128)
    # These are manually calculated based on the normalization process and image size
    # The spatial_merge_size doesn't affect image sums, only token generation
    EXP_IMG_SUM = {
        (200, 311): 38949.706477,
        (300, 212): 39848.559487,
        (251, 1374): 115051.500823,
        (1475, 477): 201339.957814,
        (1344, 1544): 536907.040239,
        (2133, 3422): 383504.917753,
    }

    test_image = _create_test_image(size, color=(128, 128, 128))
    content = ImageChunk(image=test_image)

    image = image_encoder(content).image

    expected_sum = EXP_IMG_SUM[size]

    assert image.transpose().shape[:2] == EXP_IMG_SIZES[size], image.transpose().shape[:2]
    assert np.abs(image).sum(dtype=np.float64) - expected_sum < 1e-1, np.abs(image).sum(dtype=np.float64)


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
def test_image_encoder_formats(spatial_merge_size: int, special_token_ids: SpecialImageIDs) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size,
        max_image_size=1024,
        spatial_merge_size=spatial_merge_size,
    )
    image_encoder = ImageEncoder(image_config, special_token_ids)

    pil = _create_test_image((200, 300))
    buffer = BytesIO()
    pil.save(buffer, "PNG")
    img_data = buffer.getvalue()
    data_url = f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}"

    img_pil = ImageChunk(image=pil)
    img_url = ImageURLChunk(image_url="https://url.com")
    img_data_url = ImageURLChunk(image_url=data_url)

    outputs = []
    for content in [img_pil, img_data_url]:
        assert isinstance(content, (ImageChunk, ImageURLChunk))
        outputs.append(image_encoder(content))

    def mock_get(url: str, headers: Any = None, timeout: Any = None) -> Any:
        mock_response = Mock()

        if url == "https://url.com":
            mock_response.content = img_data
        else:
            raise requests.exceptions.RequestException("Download failed")
        mock_response.raise_fort_status = Mock()
        return mock_response

    with patch("mistral_common.image.requests.get", side_effect=mock_get):
        outputs.append(image_encoder(img_url))

    for output in outputs[1:]:
        assert (output.image == outputs[0].image).all()
        assert output.tokens == outputs[0].tokens


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalize_dtype_preservation(dtype: type) -> None:
    image: np.ndarray = np.zeros((64, 64, 3), dtype=dtype)
    normalized = normalize(image, DATASET_MEAN, DATASET_STD)
    assert normalized.dtype == dtype, f"Expected {dtype} but got {normalized.dtype}"


def test_transform_image_returns_float32() -> None:
    pil_img = _create_test_image((128, 128))
    transformed = transform_image(pil_img, (64, 64))
    assert transformed.dtype == np.float32, f"Expected float32 but got {transformed.dtype}"

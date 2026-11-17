import base64
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests
from PIL import Image

from mistral_common.protocol.instruct.chunk import ImageChunk, ImageURLChunk
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

_SPECIAL_IDS = SpecialImageIDs(img=0, img_break=1, img_end=2)


def _save_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (128, 128, 128)).save(path)
    return path


def _create_test_image(size: tuple[int, int], color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", size, color)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _encoder(spatial_merge_size: int, max_image_size: int = 128) -> ImageEncoder:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size,
        max_image_size=max_image_size,
        spatial_merge_size=spatial_merge_size,
    )
    return ImageEncoder(image_config, _SPECIAL_IDS)


def _expected_marker_tokens(width_tokens: int, height_tokens: int) -> list[int]:
    tokens = ([_SPECIAL_IDS.img] * width_tokens + [_SPECIAL_IDS.img_break]) * height_tokens
    tokens[-1] = _SPECIAL_IDS.img_end
    return tokens


@pytest.mark.parametrize("relative_path", ["image.png", "my images/café.png", "100% #1/image.png"])
def test_image_from_chunk_file_uri_is_decoded(tmp_path: Path, relative_path: str) -> None:
    image_path = _save_image(tmp_path / relative_path)

    image = image_from_chunk(chunk=ImageURLChunk(image_url=image_path.as_uri()))

    assert image.size == (8, 8)


def test_image_from_chunk_file_uri_with_unencoded_path(tmp_path: Path) -> None:
    image_path = _save_image(tmp_path / "image.png")

    image = image_from_chunk(chunk=ImageURLChunk(image_url=f"file://{image_path}"))

    assert image.size == (8, 8)


def test_image_from_chunk_file_uri_with_localhost(tmp_path: Path) -> None:
    image_path = _save_image(tmp_path / "my images" / "image.png")
    url = image_path.as_uri().replace("file://", "file://localhost", 1)

    image = image_from_chunk(chunk=ImageURLChunk(image_url=url))

    assert image.size == (8, 8)


def test_image_from_chunk_file_uri_with_remote_host_is_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported file URI host 'remote-host'"):
        image_from_chunk(chunk=ImageURLChunk(image_url="file://remote-host/image.png"))


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize(
    ("size", "expected"),
    [
        pytest.param((4, 4), (1, 1), id="4x4"),
        pytest.param((16, 16), (1, 1), id="16x16"),
        pytest.param((128, 128), (8, 8), id="128x128"),
        pytest.param((512, 512), (8, 8), id="512x512-capped"),
        pytest.param((2048, 2048), (8, 8), id="2048x2048-capped"),
        pytest.param((4, 2), (1, 1), id="4x2"),
        pytest.param((8, 16), (1, 1), id="8x16"),
        pytest.param((128, 64), (8, 4), id="128x64"),
        pytest.param((512, 1024), (4, 8), id="512x1024-capped"),
    ],
)
def test_image_to_num_tokens(spatial_merge_size: int, size: tuple[int, int], expected: tuple[int, int]) -> None:
    image_encoder = _encoder(spatial_merge_size)

    img = Image.new("RGB", size, "red")

    assert image_encoder._image_to_num_tokens(img) == expected


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize(
    "size",
    [
        pytest.param((1, 512), id="1x512"),
        pytest.param((4, 10000), id="4x10000"),
        pytest.param((10000, 4), id="10000x4"),
        pytest.param((2, 4096), id="2x4096"),
    ],
)
def test_image_to_num_tokens_extreme_aspect_ratio(spatial_merge_size: int, size: tuple[int, int]) -> None:
    image_encoder = _encoder(spatial_merge_size)

    img = Image.new("RGB", size, "red")
    w_tokens, h_tokens = image_encoder._image_to_num_tokens(img)
    assert w_tokens >= 1 and h_tokens >= 1
    encoding = image_encoder(ImageChunk(image=img))
    assert len(encoding.tokens) == (w_tokens + 1) * h_tokens
    assert encoding.tokens == _expected_marker_tokens(w_tokens, h_tokens)


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize(
    ("size", "expected_size"),
    [
        pytest.param((386, 386), (3, 128, 128), id="386x386"),
        pytest.param((111, 111), (3, 112, 112), id="111x111"),
    ],
)
def test_image_encoder(spatial_merge_size: int, size: tuple[int, int], expected_size: tuple[int, int, int]) -> None:
    image_encoder = _encoder(spatial_merge_size)

    img = Image.new("RGB", size, "red")
    output = image_encoder(ImageChunk(image=img))

    w, h = image_encoder._image_to_num_tokens(img)
    assert output.image.shape == expected_size
    assert output.image.dtype == np.float32
    assert (
        w * image_encoder.image_config.image_patch_size * spatial_merge_size,
        h * image_encoder.image_config.image_patch_size * spatial_merge_size,
    ) == (expected_size[2], expected_size[1])
    assert output.tokens == _expected_marker_tokens(w, h)
    assert len(output.tokens) == (w + 1) * h


# Expected processed sizes (width, height) after resize, per input size.
_EXPECTED_PROCESSED_SIZES = {
    (200, 311): (208, 320),
    (300, 212): (304, 224),
    (251, 1374): (192, 1024),
    (1475, 477): (1024, 336),
    (1344, 1544): (896, 1024),
    (2133, 3422): (640, 1024),
}


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
@pytest.mark.parametrize(
    "size",
    [
        pytest.param((200, 311), id="200x311"),
        pytest.param((300, 212), id="300x212"),
        pytest.param((251, 1374), id="251x1374"),
        pytest.param((1475, 477), id="1475x477"),
        pytest.param((1344, 1544), id="1344x1544"),
        pytest.param((2133, 3422), id="2133x3422"),
    ],
)
def test_image_processing(spatial_merge_size: int, size: tuple[int, int]) -> None:
    image_encoder = _encoder(spatial_merge_size, max_image_size=1024)

    # Images with w,h >= 1024 are resized to 1024, else rounded to the nearest
    # multiple of 16, keeping the aspect ratio. The spatial merge size does not
    # affect the processed image, only the token grid.
    expected_w, expected_h = _EXPECTED_PROCESSED_SIZES[size]
    test_image = _create_test_image(size, color=(128, 128, 128))

    image = image_encoder(ImageChunk(image=test_image)).image

    assert image.shape == (3, expected_h, expected_w)
    # Two-sided check against the analytic expectation for a uniform gray image:
    # each pixel contributes sum((128/255 - mean) / std) over the three channels.
    per_pixel_sum = sum((128 / 255 - mean) / std for mean, std in zip(DATASET_MEAN, DATASET_STD))
    expected_sum = expected_w * expected_h * per_pixel_sum
    actual_sum = np.abs(image).sum(dtype=np.float64)
    assert abs(actual_sum - expected_sum) < 1e-4 * abs(expected_sum), actual_sum


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
def test_image_encoder_formats(spatial_merge_size: int) -> None:
    image_encoder = _encoder(spatial_merge_size, max_image_size=1024)

    pil = _create_test_image((200, 300))
    img_data = _png_bytes(pil)
    # Correctly labeled PNG bytes; the payload is sniffed by PIL, not the label.
    data_url = f"data:image/png;base64,{base64.b64encode(img_data).decode('utf-8')}"

    img_pil = ImageChunk(image=pil)
    img_data_url = ImageURLChunk(image_url=data_url)
    img_url = ImageURLChunk(image_url="https://url.com")

    outputs = [image_encoder(img_pil), image_encoder(img_data_url)]

    def mock_get(url: str, headers: Any = None, timeout: Any = None) -> Any:
        if url != "https://url.com":
            raise requests.exceptions.RequestException("Download failed")
        mock_response = Mock()
        mock_response.content = img_data
        mock_response.raise_for_status = Mock()
        return mock_response

    with patch("mistral_common.image.requests.get", side_effect=mock_get):
        outputs.append(image_encoder(img_url))

    for output in outputs[1:]:
        np.testing.assert_array_equal(output.image, outputs[0].image)
        assert output.tokens == outputs[0].tokens


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        pytest.param(np.float32, np.float32, id="float32"),
        pytest.param(np.float64, np.float64, id="float64"),
        pytest.param(np.uint8, np.float32, id="uint8"),
        pytest.param(np.int64, np.float32, id="int64"),
    ],
)
def test_normalize_dtype(dtype: type, expected_dtype: type) -> None:
    image: np.ndarray = np.full((64, 64, 3), 255, dtype=dtype)
    normalized = normalize(image, DATASET_MEAN, DATASET_STD)
    assert normalized.dtype == expected_dtype, f"Expected {expected_dtype} but got {normalized.dtype}"
    expected = (1 - np.array(DATASET_MEAN)) / np.array(DATASET_STD)
    np.testing.assert_allclose(normalized[:, 0, 0], expected, rtol=1e-6)


def test_transform_image_returns_float32() -> None:
    pil_img = _create_test_image((128, 128))
    transformed = transform_image(pil_img, (64, 64))
    assert transformed.dtype == np.float32, f"Expected float32 but got {transformed.dtype}"

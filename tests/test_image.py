import base64
from io import BytesIO
from typing import Any

import numpy as np
import pytest
import requests
from PIL import Image

from mistral_common.protocol.instruct.chunk import (
    ImageChunk,
    ImageURLChunk,
    TextChunk,
)
from mistral_common.tokens.tokenizers.image import ImageConfig, ImageEncoder, SpecialImageIDs


def _create_test_image(size: tuple[int, int], color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", size, color)


@pytest.fixture
def special_token_ids() -> SpecialImageIDs:
    return SpecialImageIDs(img=0, img_break=1, img_end=2)


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
def test_image_to_num_tokens(spatial_merge_size: int, special_token_ids: SpecialImageIDs) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size, max_image_size=128, spatial_merge_size=spatial_merge_size
    )
    image_encoder = ImageEncoder(image_config, special_token_ids)

    for size, exp in [(4, 1), (16, 1), (128, 8), (512, 8), (2048, 8)]:
        img = Image.new("RGB", (size, size), "red")
        assert image_encoder._image_to_num_tokens(img) == (exp, exp)

    for size1, size2, exp1, exp2 in [(4, 2, 1, 1), (8, 16, 1, 1), (128, 64, 8, 4), (512, 1024, 4, 8)]:
        img = Image.new("RGB", (size1, size2), "red")
        assert image_encoder._image_to_num_tokens(img) == (exp1, exp2)


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
def test_download_image(spatial_merge_size: int, special_token_ids: SpecialImageIDs, mocker: Any) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size, max_image_size=128, spatial_merge_size=spatial_merge_size
    )
    image_encoder = ImageEncoder(image_config, special_token_ids)

    test_image1 = _create_test_image((500, 300), color=(128, 128, 128))
    test_image2 = _create_test_image((400, 600), color=(100, 150, 200))

    def mock_get(url: str, headers: Any = None) -> Any:
        mock_response = mocker.Mock()

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

        mock_response.raise_for_status = mocker.Mock()
        return mock_response

    url1 = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
    url2 = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"

    mocker.patch("mistral_common.image.requests.get", side_effect=mock_get)

    for url, expected_image in [(url1, test_image1), (url2, test_image2)]:
        content = ImageURLChunk(image_url=url)
        result = image_encoder(content)

        assert result.image is not None, "Image should be processed successfully"

    # Test request error.
    invalid_url = "https://invalid.url/image.jpg"
    response = requests.Response()
    response.status_code = 404
    response._content = b"Not found"

    mocker.patch("mistral_common.image.requests.get", return_value=response)

    with pytest.raises(RuntimeError, match="Error downloading the image"):
        content = ImageURLChunk(image_url=invalid_url)
        image_encoder(content)


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
def test_image_encoder(spatial_merge_size: int, special_token_ids: SpecialImageIDs) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size, max_image_size=128, spatial_merge_size=spatial_merge_size
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
        image_patch_size=16 // spatial_merge_size, max_image_size=1024, spatial_merge_size=spatial_merge_size
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
        (200, 311): 38949.726562,
        (300, 212): 39848.566406,
        (251, 1374): 115051.507812,
        (1475, 477): 201340.125000,
        (1344, 1544): 536907.000000,
        (2133, 3422): 383505.000000,
    }

    test_image = _create_test_image(size, color=(128, 128, 128))
    content = ImageChunk(image=test_image)

    image = image_encoder(content).image

    expected_sum = EXP_IMG_SUM[size]

    assert image.transpose().shape[:2] == EXP_IMG_SIZES[size], image.transpose().shape[:2]
    assert np.abs(image).sum() - expected_sum < 1e-1, np.abs(image).sum()


@pytest.mark.parametrize("spatial_merge_size", [1, 2])
def test_image_encoder_formats(spatial_merge_size: int, special_token_ids: SpecialImageIDs) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size, max_image_size=1024, spatial_merge_size=spatial_merge_size
    )
    image_encoder = ImageEncoder(image_config, special_token_ids)

    url = "https://picsum.photos/id/237/200/300"
    img_data = requests.get(url).content

    pil = Image.open(BytesIO(img_data))
    data_url = f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}"

    img_pil = ImageChunk(image=pil)
    img_url = ImageURLChunk(image_url=url)
    img_data_url = ImageURLChunk(image_url=data_url)

    outputs = []
    for content in [img_pil, img_url, img_data_url]:
        assert isinstance(content, (ImageChunk, ImageURLChunk))

        outputs.append(image_encoder(content))

    for output in outputs[1:]:
        assert (output.image == outputs[0].image).all()
        assert output.tokens == outputs[0].tokens

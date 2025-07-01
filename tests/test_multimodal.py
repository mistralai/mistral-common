import base64
from io import BytesIO
from typing import Any, Tuple

import numpy as np
import pytest
import requests
from PIL import Image

from mistral_common.protocol.instruct.messages import (
    ImageChunk,
    ImageURLChunk,
    TextChunk,
)
from mistral_common.tokens.tokenizers.image import ImageConfig, ImageEncoder, SpecialImageIDs, transform_image


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
def test_download_gated_image(spatial_merge_size: int, special_token_ids: SpecialImageIDs) -> None:
    image_config = ImageConfig(
        image_patch_size=16 // spatial_merge_size, max_image_size=128, spatial_merge_size=spatial_merge_size
    )
    image_encoder = ImageEncoder(image_config, special_token_ids)

    url1 = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
    url2 = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"

    for url in [url1, url2]:
        content = ImageURLChunk(image_url=url)
        image = image_encoder(content).image

        assert image is not None, "Make sure gated wikipedia images can be downloaded"


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
        image_encoder(text_chunk)  # type: ignore

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
        image_encoder(text_chunk)  # type: ignore

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
def test_image_processing(special_token_ids: SpecialImageIDs, size: Tuple[int, int], spatial_merge_size: int) -> None:
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
    # integration test to make sure the img processing stays 100% the same
    EXP_IMG_SUM = {
        (200, 311): 232402.60528341102,
        (300, 212): 183409.6477803542,
        (251, 1374): 727176.6407945724,
        (1475, 477): 987062.1457962373,
        (1344, 1544): 2984206.24160149,
        (2133, 3422): 2305820.5333060464,
    }

    url = f"https://picsum.photos/id/237/{size[0]}/{size[1]}"

    content = ImageURLChunk(image_url=url)

    RETRY = 10  # sometimes the image download fails, so we retry a few times
    for i in range(RETRY):
        try:
            image = image_encoder(content).image
            break
        except RuntimeError as e:
            if i == RETRY - 1:
                raise e
            continue

    assert image.transpose().shape[:2] == EXP_IMG_SIZES[size], image.transpose().shape[:2]
    assert np.abs(image).sum() - EXP_IMG_SUM[size] < 1e-1, np.abs(image).sum()


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


def test_transform_image_missing_cv2(monkeypatch: Any) -> None:
    img = Image.new("RGB", (10, 10), "red")

    monkeypatch.setattr("mistral_common.tokens.tokenizers.image.is_cv2_installed", lambda: False)

    with pytest.raises(ImportError) as exc_info:
        transform_image(img, (16, 16))

    assert "pip install mistral-common[opencv]" in str(exc_info.value)

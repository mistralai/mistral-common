import base64
from io import BytesIO

import pytest
import requests
from mistral_common.protocol.instruct.messages import (
    ImageChunk,
    ImageURLChunk,
    TextChunk,
)
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder, MultimodalConfig, SpecialImageIDs
from PIL import Image


@pytest.fixture
def mm_config() -> MultimodalConfig:
    return MultimodalConfig(image_patch_size=16, max_image_size=128)


@pytest.fixture
def special_token_ids() -> SpecialImageIDs:
    return SpecialImageIDs(img=0, img_break=1, img_end=2)


def test_image_to_num_tokens(mm_config: MultimodalConfig, special_token_ids: SpecialImageIDs) -> None:
    mm_encoder = ImageEncoder(mm_config, special_token_ids)

    for size, exp in [(4, 1), (16, 1), (128, 8), (512, 8), (2048, 8)]:
        img = Image.new("RGB", (size, size), "red")
        assert mm_encoder._image_to_num_tokens(img) == (exp, exp)

    for size1, size2, exp1, exp2 in [(4, 2, 1, 1), (8, 16, 1, 1), (128, 64, 8, 4), (512, 1024, 4, 8)]:
        img = Image.new("RGB", (size1, size2), "red")
        assert mm_encoder._image_to_num_tokens(img) == (exp1, exp2)


def test_image_encoder(mm_config: MultimodalConfig, special_token_ids: SpecialImageIDs) -> None:
    mm_encoder = ImageEncoder(mm_config, special_token_ids)

    size = 386
    img = Image.new("RGB", (size, size), "red")
    img_chunk = ImageChunk(image=img)
    text_chunk = TextChunk(text="")

    with pytest.raises(AttributeError):
        mm_encoder(text_chunk)  # type: ignore

    output = mm_encoder(img_chunk)
    tokens, image = output.tokens, output.image

    w, h = mm_encoder._image_to_num_tokens(img)
    # max image size 128
    assert image.shape == (3, 128, 128)
    assert (w * mm_config.image_patch_size, h * mm_config.image_patch_size) == (128, 128)
    assert len(tokens) == (w + 1) * h

    size = 111  # nearest multiple of sixteen lower than 128 is 112
    img = Image.new("RGB", (size, size), "red")
    img_chunk = ImageChunk(image=img)
    text_chunk = TextChunk(text="")

    with pytest.raises(AttributeError):
        mm_encoder(text_chunk)  # type: ignore

    output = mm_encoder(img_chunk)
    tokens, image = output.tokens, output.image
    assert image.shape == (3, 112, 112)
    w, h = mm_encoder._image_to_num_tokens(img)
    assert (w * mm_config.image_patch_size, h * mm_config.image_patch_size) == (112, 112)
    assert len(tokens) == (w + 1) * h


def test_image_encoder_formats(mm_config: MultimodalConfig, special_token_ids: SpecialImageIDs) -> None:
    mm_encoder = ImageEncoder(mm_config, special_token_ids)

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

        outputs.append(mm_encoder(content))

    for output in outputs[1:]:
        assert (output.image == outputs[0].image).all()
        assert output.tokens == outputs[0].tokens

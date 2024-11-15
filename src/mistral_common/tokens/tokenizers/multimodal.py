import base64
import logging
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Tuple, Union

import numpy as np
from PIL import Image

from mistral_common.multimodal import SerializableImage, download_image
from mistral_common.protocol.instruct.messages import ImageChunk, ImageURLChunk
from mistral_common.tokens.tokenizers.base import (
    ImageEncoding,
    MultiModalEncoder,
    SpecialImageIDs,
)

logger = logging.getLogger(__name__)


_cv2_installed: bool
try:
    import cv2

    _cv2_installed = True
except ImportError:
    _cv2_installed = False
except Exception as e:
    # cv2 has lots of import problems: https://github.com/opencv/opencv-python/issues/884
    # for better UX, let's simply skip all errors that might arise from import for now
    logger.warning(
        f"Warning: Your installation of OpenCV appears to be broken: {e}."
        "Please follow the instructions at https://github.com/opencv/opencv-python/issues/884 "
        "to correct your environment. The import of cv2 has been skipped."
    )


def is_cv2_installed() -> bool:
    return _cv2_installed


def image_from_chunk(chunk: Union[ImageURLChunk, ImageChunk]) -> SerializableImage:
    """Get a serializable image from a chunk."""
    if isinstance(chunk, ImageChunk):
        return chunk.image
    if chunk.get_url().startswith("data:image"):
        data = chunk.get_url().split(",")[1]
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))
    if chunk.get_url().startswith("http"):
        return download_image(chunk.get_url())

    raise RuntimeError(f"Unsupported image url scheme {chunk.get_url()}")


DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)  # RGB
DATASET_STD = (0.26862954, 0.26130258, 0.27577711)  # RGB


# only relevant for spm
class MultiModalVersion(str, Enum):
    m1 = "m1"

    @property
    def config(self) -> "MultimodalConfig":
        if self.name == "m1":
            return MultimodalConfig(16, 1024)

        raise NotImplementedError(f"{self.name}")


@dataclass
class MultimodalConfig:
    image_patch_size: int
    max_image_size: int


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert a PIL image to RGB.
    We ensure transparent background becomes white.
    """
    if image.mode == "RGB":
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    white_bg: Image.Image = Image.new("RGBA", image.size, "WHITE")
    white_bg.paste(image, (0, 0), image)
    return white_bg.convert("RGB")


def normalize(
    np_image: np.ndarray,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> np.ndarray:
    """
    Normalize a tensor image with mean and standard deviation.

    Args:
    image (np.ndarray): Image to be normalized.
    mean (tuple[float, float, float]): Mean for each channel.
    std (tuple[float, float, float]): Standard deviation for each channel.

    Returns:
    np.ndarray: Normalized image with shape (C, H, W).
    """
    np_image = np_image / 255.0

    assert len(np_image.shape) == 3, f"{np_image.shape=}"
    assert np_image.shape[2] == len(mean) == len(std), f"{np_image.shape=}, {mean=}, {std=}"

    np_image = (np_image - mean) / std

    return np_image.transpose(2, 0, 1)


def transform_image(image: Image.Image, new_size: Tuple[int, int]) -> np.ndarray:
    if not is_cv2_installed():
        raise ImportError("OpenCV is required for this function. Install it with 'pip install mistral_common[opencv]'")

    np_image = cv2.resize(np.array(_convert_to_rgb(image), dtype=np.float32), new_size, interpolation=cv2.INTER_CUBIC)
    return normalize(np_image, DATASET_MEAN, DATASET_STD)


class ImageEncoder(MultiModalEncoder):
    def __init__(self, mm_config: MultimodalConfig, special_ids: SpecialImageIDs) -> None:
        self.mm_config = mm_config
        self.special_ids = special_ids

    def _image_to_num_tokens(self, img: Image.Image) -> Tuple[int, int]:
        w: Union[int, float]
        h: Union[int, float]

        w, h = img.size
        ratio = max(h / self.mm_config.max_image_size, w / self.mm_config.max_image_size)
        if ratio > 1:
            w = round(w / ratio)
            h = round(h / ratio)

        width_tokens = (w - 1) // self.mm_config.image_patch_size + 1
        height_tokens = (h - 1) // self.mm_config.image_patch_size + 1

        return width_tokens, height_tokens

    def __call__(self, content: Union[ImageChunk, ImageURLChunk]) -> ImageEncoding:
        """
        Converts ImageChunks to numpy image arrays and image token ids

        Args:
        image (ImageChunk, ImageURLChunk): ImageChunk to be converted

        Returns:
        ImageEncoding containing image token ids and processed image in numpy format
        """
        image = image_from_chunk(content)
        w, h = self._image_to_num_tokens(image)
        assert w > 0
        assert h > 0
        image_tokens = ([self.special_ids.img] * w + [self.special_ids.img_break]) * h
        image_tokens[-1] = self.special_ids.img_end
        new_image_size = (
            w * self.mm_config.image_patch_size,
            h * self.mm_config.image_patch_size,
        )
        processed_image = transform_image(image, new_image_size)
        return ImageEncoding(tokens=image_tokens, image=processed_image)

    @property
    def image_token(self) -> int:
        return self.special_ids.img

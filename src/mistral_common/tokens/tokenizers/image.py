import base64
import logging
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
from PIL import Image

from mistral_common.image import SerializableImage, download_image
from mistral_common.protocol.instruct.messages import ImageChunk, ImageURLChunk

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
    r"""Check if OpenCV is installed."""
    return _cv2_installed


@dataclass
class ImageEncoding:
    """A tokenized image.

    Attributes:
        tokens: The token ids.
        image: The image as a numpy array.

    Examples:
        >>> import numpy as np
        >>> image_encoding = ImageEncoding(tokens=[1, 2, 3], image=np.array([[0., 0.5, 1.]]))
    """

    tokens: List[int]
    image: np.ndarray


@dataclass
class SpecialImageIDs:
    """Special image tokens ids.

    Attributes:
        img: The image token id.
        img_break: The image break token id.
        img_end: The image end token id.

    Examples:
        >>> special_image_ids = SpecialImageIDs(img=1, img_break=2, img_end=3)
    """

    img: int
    img_break: int
    img_end: int


def image_from_chunk(chunk: Union[ImageURLChunk, ImageChunk]) -> SerializableImage:
    r"""Get a serializable image from a chunk.

    Args:
        chunk: The chunk to get the image from.

    Returns:
        The image as a PIL Image object.
    """
    if isinstance(chunk, ImageChunk):
        return chunk.image
    if chunk.get_url().startswith("data:image"):
        data = chunk.get_url().split(",")[1]
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))
    if chunk.get_url().startswith("file"):
        return Image.open(open(chunk.get_url().replace("file://", ""), "rb"))
    if chunk.get_url().startswith("http"):
        return download_image(chunk.get_url())

    raise RuntimeError(f"Unsupported image url scheme {chunk.get_url()}")


DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)  # RGB
DATASET_STD = (0.26862954, 0.26130258, 0.27577711)  # RGB


# only relevant for spm
class MultiModalVersion(str, Enum):
    r"""Version of the image tokenizer."""

    m1 = "m1"

    @property
    def config(self) -> "ImageConfig":
        if self.name == "m1":
            return ImageConfig(16, 1024)

        raise NotImplementedError(f"{self.name}")


@dataclass
class ImageConfig:
    r"""Configuration for the image tokenizers."""

    image_patch_size: int
    max_image_size: int
    spatial_merge_size: int = 1


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    r"""Convert a PIL image to RGB.

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
    r"""Normalize a tensor image with mean and standard deviation.

    Args:
        np_image: Image to be normalized.
        mean: Mean for each channel.
        std: Standard deviation for each channel.

    Returns:
        Normalized image with shape (C, H, W).
    """
    np_image = np_image / 255.0

    assert len(np_image.shape) == 3, f"{np_image.shape=}"
    assert np_image.shape[2] == len(mean) == len(std), f"{np_image.shape=}, {mean=}, {std=}"

    np_image = (np_image - mean) / std

    return np_image.transpose(2, 0, 1)


def transform_image(image: Image.Image, new_size: Tuple[int, int]) -> np.ndarray:
    r"""Transform an image to a numpy array with the given size.

    Args:
        image: Image to be transformed.
        new_size: New size of the image.

    Returns:
        Transformed image with shape (C, H, W).
    """
    if not is_cv2_installed():
        raise ImportError("OpenCV is required for this function. Install it with 'pip install mistral-common[opencv]'")

    np_image = cv2.resize(np.array(_convert_to_rgb(image), dtype=np.float32), new_size, interpolation=cv2.INTER_CUBIC)
    return normalize(np_image, DATASET_MEAN, DATASET_STD)


class ImageEncoder:
    r"""Image encoder for the image tokenizer."""

    def __init__(self, image_config: ImageConfig, special_ids: SpecialImageIDs) -> None:
        r"""Initialize the image encoder.

        Args:
            image_config: Configuration for the image tokenizer.
            special_ids: Special image tokens ids.
        """
        self.image_config = image_config
        self.special_ids = special_ids

    @property
    def mm_config(self) -> ImageConfig:
        # this property is deprecated, use image_config instead
        # TODO(Patrick) - throw deprecation warning once
        # changes implemented into vLLM and transformers
        return self.image_config

    def _image_to_num_tokens(self, img: Image.Image) -> Tuple[int, int]:
        w: Union[int, float]
        h: Union[int, float]

        w, h = img.size
        ratio = max(h / self.image_config.max_image_size, w / self.image_config.max_image_size)
        if ratio > 1:
            w = round(w / ratio)
            h = round(h / ratio)

        width_tokens = (w - 1) // (self.image_config.image_patch_size * self.image_config.spatial_merge_size) + 1
        height_tokens = (h - 1) // (self.image_config.image_patch_size * self.image_config.spatial_merge_size) + 1

        return width_tokens, height_tokens

    def __call__(self, content: Union[ImageChunk, ImageURLChunk]) -> ImageEncoding:
        r"""Converts an image chunk to an image encoding.

        Args:
            content: image chunk to be converted.

        Returns:
            Image encoding.
        """
        image = image_from_chunk(content)
        w, h = self._image_to_num_tokens(image)
        assert w > 0
        assert h > 0
        image_tokens = ([self.special_ids.img] * w + [self.special_ids.img_break]) * h
        image_tokens[-1] = self.special_ids.img_end
        new_image_size = (
            w * self.image_config.image_patch_size * self.image_config.spatial_merge_size,
            h * self.image_config.image_patch_size * self.image_config.spatial_merge_size,
        )
        processed_image = transform_image(image, new_image_size)
        return ImageEncoding(tokens=image_tokens, image=processed_image)

    @property
    def image_token(self) -> int:
        return self.special_ids.img

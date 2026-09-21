import base64
import logging
from dataclasses import dataclass
from enum import Enum
from io import BytesIO

import numpy as np
from PIL import Image

from mistral_common.image import (
    SerializableImage,
    download_image,
)
from mistral_common.imports import assert_opencv_installed, is_opencv_installed
from mistral_common.protocol.instruct.chunk import ImageChunk, ImageURLChunk

logger = logging.getLogger(__name__)


if is_opencv_installed():
    import cv2


@dataclass
class ImageEncoding:
    r"""A tokenized image.

    Attributes:
        tokens: The token IDs representing the image (img/break/end markers).
        image: The processed image as a numpy array of shape (C, H, W),
            normalized and resized for the model.

    Examples:
        >>> import numpy as np
        >>> image_encoding = ImageEncoding(tokens=[1, 2, 3], image=np.array([[0., 0.5, 1.]]))
    """

    tokens: list[int]
    image: np.ndarray


@dataclass
class SpecialImageIDs:
    r"""Special image token IDs used to mark image regions in the token stream.

    Attributes:
        img: The token ID marking a single image patch.
        img_break: The token ID marking the end of an image row.
        img_end: The token ID marking the end of the whole image.

    Examples:
        >>> special_image_ids = SpecialImageIDs(img=1, img_break=2, img_end=3)
    """

    img: int
    img_break: int
    img_end: int


def image_from_chunk(chunk: ImageURLChunk | ImageChunk) -> SerializableImage:
    r"""Load a serializable image from a chunk.

    Resolves the image from the chunk's data, accepting base64 data URLs,
    file URIs, and http(s) URLs.

    Args:
        chunk: The chunk to get the image from.

    Returns:
        The image as a PIL Image object.

    Raises:
        ValueError: If a data URL does not contain a base64 payload or the URL
            scheme is unsupported.
    """
    if isinstance(chunk, ImageChunk):
        return chunk.image
    url = chunk.get_url()
    if url.startswith("data:image"):
        _, _, data = url.partition(",")
        if not data:
            raise ValueError(f"Invalid image data URL {url[:64]}: expected a base64 payload after a comma.")
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))
    if url.startswith("file://"):
        with open(url.removeprefix("file://"), "rb") as file:
            image = Image.open(file)
            image.load()
        return image
    if url.startswith("http"):
        return download_image(url=url, timeout=None)

    raise ValueError(f"Unsupported image url scheme {url}")


DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)  # RGB
DATASET_STD = (0.26862954, 0.26130258, 0.27577711)  # RGB


# only relevant for spm
class MultiModalVersion(str, Enum):
    r"""Version of the image tokenizer.

    Attributes:
        m1: The first multimodal version, patch size 16 and max image size 1024.
    """

    m1 = "m1"

    @property
    def config(self) -> "ImageConfig":
        r"""The image config associated with this version.

        Returns:
            The ImageConfig for this multimodal version.

        Raises:
            NotImplementedError: If the version has no config.
        """
        if self.name == "m1":
            return ImageConfig(16, 1024)

        raise NotImplementedError(f"{self.name}")


@dataclass
class ImageConfig:
    r"""Configuration for image tokenization.

    Attributes:
        image_patch_size: Size of a single image patch in pixels. The image
            grid is made of patches of this size; must be > 0.
        max_image_size: Maximum image dimension (width or height) in pixels
            before downsampling; must be > 0.
        spatial_merge_size: Number of adjacent patches merged into one image
            token along each dimension; must be > 0.
    """

    image_patch_size: int
    max_image_size: int
    spatial_merge_size: int = 1

    def __post_init__(self) -> None:
        assert self.image_patch_size > 0, f"image_patch_size must be > 0, got {self.image_patch_size}"
        assert self.max_image_size > 0, f"max_image_size must be > 0, got {self.max_image_size}"
        assert self.spatial_merge_size > 0, f"spatial_merge_size must be > 0, got {self.spatial_merge_size}"


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    r"""Convert a PIL image to RGB.

    Transparent areas become white: RGBA images are composited over a white
    background before the final RGB conversion.

    Returns:
        The image in RGB mode, or the input unchanged if already RGB.
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
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    r"""Normalize an image array with per-channel mean and standard deviation.

    Args:
        np_image: Image to normalize, with shape (H, W, C) and values in [0, 255].
        mean: Mean for each channel, values in [0, 1].
        std: Standard deviation for each channel, values in [0, 1].

    Returns:
        Normalized image with shape (C, H, W), scaled to unit variance.

    Raises:
        AssertionError: If `np_image` is not (H, W, C) or the channel count does
            not match mean and std.
    """
    np_image = np.divide(np_image, np.float32(255.0), dtype=np_image.dtype)

    assert len(np_image.shape) == 3, f"{np_image.shape=}"
    assert np_image.shape[2] == len(mean) == len(std), f"{np_image.shape=}, {mean=}, {std=}"

    mean_array = np.asarray(mean, dtype=np_image.dtype)
    std_array = np.asarray(std, dtype=np_image.dtype)

    np_image -= mean_array
    np_image /= std_array

    return np_image.transpose(2, 0, 1)


def transform_image(image: Image.Image, new_size: tuple[int, int]) -> np.ndarray:
    r"""Resize and normalize an image for the model.

    Converts to RGB (transparent backgrounds become white), resizes with cubic
    interpolation, then normalizes with the dataset statistics.

    Args:
        image: PIL image to transform.
        new_size: Target (width, height) in pixels.

    Returns:
        Transformed image with shape (C, H, W), normalized.

    Raises:
        ImportError: If opencv is not installed.
    """
    assert_opencv_installed()

    np_image = cv2.resize(np.array(_convert_to_rgb(image), dtype=np.float32), new_size, interpolation=cv2.INTER_CUBIC)
    return normalize(np_image, DATASET_MEAN, DATASET_STD)


class ImageEncoder:
    r"""Encodes images into tokens and processed arrays.

    An image is turned into a grid of patch tokens: each row is `width_tokens`
    img tokens terminated by an `img_break` token, and the last row ends with an
    `img_end` token. The image array is resized so its dimensions are multiples
    of the patch and merge sizes.
    """

    def __init__(self, image_config: ImageConfig, special_ids: SpecialImageIDs) -> None:
        r"""Initialize the image encoder.

        Args:
            image_config: Configuration controlling patch size, max image size,
                and spatial merging.
            special_ids: Token IDs used to mark image patches, row breaks, and
                the image end.
        """
        self.image_config = image_config
        self.special_ids = special_ids

    @property
    def mm_config(self) -> ImageConfig:
        r"""Deprecated alias for `image_config`.

        Returns:
            The image config.
        """
        # this property is deprecated, use image_config instead
        # TODO(Patrick) - throw deprecation warning once
        # changes implemented into vLLM and transformers
        return self.image_config

    def _image_to_num_tokens(self, img: Image.Image) -> tuple[int, int]:
        w: int | float
        h: int | float

        w, h = img.size
        ratio = max(h / self.image_config.max_image_size, w / self.image_config.max_image_size)
        if ratio > 1:
            # an extreme aspect ratio would otherwise round the shorter side to 0 pixels
            w = max(round(w / ratio), 1)
            h = max(round(h / ratio), 1)

        width_tokens = (w - 1) // (self.image_config.image_patch_size * self.image_config.spatial_merge_size) + 1
        height_tokens = (h - 1) // (self.image_config.image_patch_size * self.image_config.spatial_merge_size) + 1

        return width_tokens, height_tokens

    def __call__(self, content: ImageChunk | ImageURLChunk) -> ImageEncoding:
        r"""Convert an image chunk into an image encoding.

        The image is loaded, resized so its token grid covers the image, and
        encoded as img/`img_break`/`img_end` marker tokens plus the processed
        pixel array.

        Args:
            content: Image chunk to be converted.

        Returns:
            An ImageEncoding with the marker tokens and the processed image
            of shape (C, H, W).
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
        r"""The token ID marking a single image patch.

        Returns:
            The img special token ID.
        """
        return self.special_ids.img

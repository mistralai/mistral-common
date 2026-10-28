import base64
import io

import requests
from PIL import ExifTags, Image, ImageOps
from pydantic import BeforeValidator, PlainSerializer, SerializationInfo
from typing_extensions import Annotated

from mistral_common import __version__


def apply_exif_orientation(image: Image.Image) -> Image.Image:
    r"""Apply EXIF orientation transposition to an image if present.

    Transposes the image according to its EXIF orientation tag (e.g. from mobile cameras)
    so pixels match user visual orientation. The orientation tag is removed from the
    returned image, so applying this function more than once is safe.

    Note:
        This also applies to `PIL.Image.Image` objects passed in directly, matching vLLM's
        behavior. An image whose pixels were already rotated manually but that still
        carries an orientation tag will therefore be rotated again.

    Args:
        image: The PIL Image to transpose.

    Returns:
        The oriented PIL Image, with its original `format` preserved. If the image has no
        orientation tag, or its EXIF metadata is malformed, the same object is returned
        unchanged (no copy and no pixel decoding).

    Raises:
        OSError: If the image has an orientation tag but its pixel data cannot be decoded
            (e.g. a truncated file).
    """
    try:
        orientation = image.getexif().get(ExifTags.Base.Orientation, 1)
    except Exception:
        # Malformed EXIF metadata only; pixel decoding errors surface below.
        return image
    if orientation == 1:
        return image
    image_format = image.format
    oriented = ImageOps.exif_transpose(image)
    oriented.format = image_format
    return oriented


def download_image(url: str) -> Image.Image:
    r"""Download an image from a URL and return it as a PIL Image.

    Args:
        url: The URL of the image to download.

    Returns:
       The downloaded image as a PIL Image object.
    """
    headers = {"User-Agent": f"mistral-common/{__version__}"}
    try:
        # Make a request to download the image
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        # Convert the image content to a PIL Image
        img = Image.open(io.BytesIO(response.content))
        return apply_exif_orientation(img)

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error downloading the image from {url}: {e}.")
    except Exception as e:
        raise RuntimeError(f"Error converting to PIL image: {e}")


def maybe_load_image_from_str_or_bytes(x: Image.Image | str | bytes) -> Image.Image:
    r"""Load an image from a string or bytes.

    If the input is already a PIL Image, return it as is.

    Args:
        x: The input to load the image from. Can be a PIL Image, a string, or
            bytes. If it's a string, it's assumed to be a base64 encoded string
            of bytes; if it's bytes, raw encoded image data (e.g., PNG).

    Returns:
       The loaded image as a PIL Image object.

    Raises:
        RuntimeError: If the input cannot be decoded into an image.
    """
    if isinstance(x, Image.Image):
        return apply_exif_orientation(x)
    if isinstance(x, bytes):
        try:
            return apply_exif_orientation(Image.open(io.BytesIO(x)))
        except Exception:
            raise RuntimeError("Encountered an error when loading image from bytes.")

    try:
        image = Image.open(io.BytesIO(base64.b64decode(x.encode("ascii"))))
        return apply_exif_orientation(image)
    except Exception as e:
        raise RuntimeError(
            f"Encountered an error when loading image from bytes starting "
            f"with '{x[:20]}'. Expected either a PIL.Image.Image or a base64 "
            f"encoded string of bytes."
        ) from e


def serialize_image_to_byte_str(im: Image.Image, info: SerializationInfo) -> str:
    r"""Serialize an image to a base64 encoded string of bytes.

    The output honors two context keys from info, when present:
    `max_image_b64_len` truncates the base64 string for display, and
    `add_format_prefix` prepends a data:...;base64, prefix.

    Args:
        im: The image to serialize. Its format is used, defaulting to PNG
            when unset.
        info: The pydantic serialization info carrying optional context.

    Returns:
        The serialized image as a base64 encoded string of bytes.
    """
    if hasattr(info, "context"):
        context = info.context or {}
    else:
        context = {}

    stream = io.BytesIO()
    im_format = im.format or "PNG"
    im.save(stream, format=im_format)
    im_b64 = base64.b64encode(stream.getvalue()).decode("ascii")
    if context and (max_image_b64_len := context.get("max_image_b64_len")):
        return im_b64[:max_image_b64_len] + "..."
    if context and context.get("add_format_prefix"):
        im_b64 = f"data:image/{im_format.lower()};base64," + im_b64
    return im_b64


SerializableImage = Annotated[
    Image.Image,
    BeforeValidator(maybe_load_image_from_str_or_bytes),
    PlainSerializer(serialize_image_to_byte_str),
    "A normal PIL image that supports serialization to b64 bytes string.",
]

import base64
import io
from typing import Annotated

import requests
from PIL import Image
from pydantic import BeforeValidator, PlainSerializer, SerializationInfo

from mistral_common import __version__
from mistral_common.exceptions import ImageDecodeException
from mistral_common.types import TimeoutType
from mistral_common.utils.data_url import maybe_strip_base64_data_url_prefix
from mistral_common.utils.download import DOWNLOAD_TIMEOUT_ENV_KEY, resolve_download_timeout


def download_image(url: str, timeout: TimeoutType | None) -> Image.Image:
    r"""Download an image from a URL and return it as a PIL Image.

    Args:
        url: The URL of the image to download.
        timeout: Seconds to wait for the server response, converted to float.
            If None, uses the `MISTRAL_COMMON_DOWNLOAD_TIMEOUT` environment
            variable.

    Returns:
       The downloaded image as a PIL Image object.
    """
    timeout = resolve_download_timeout(timeout)

    headers = {"User-Agent": f"mistral-common/{__version__}"}
    try:
        # Make a request to download the image
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        # Convert the image content to a PIL Image
        img = Image.open(io.BytesIO(response.content))
        return img

    except requests.exceptions.Timeout as e:
        raise requests.exceptions.Timeout(
            f"Error downloading the image from {url}: timed out after {timeout} seconds. "
            f"Pass a larger `timeout` or set the environment variable `{DOWNLOAD_TIMEOUT_ENV_KEY}` "
            "to increase the timeout.",
            request=e.request,
            response=e.response,
        ) from e
    except Exception as e:
        if isinstance(e, requests.exceptions.RequestException):
            raise
        raise ImageDecodeException(f"Error converting to PIL image: {e}") from e


def maybe_load_image_from_str_or_bytes(x: Image.Image | str | bytes) -> Image.Image:
    r"""Load an image from a string or bytes.

    If the input is already a PIL Image, return it as is.

    Args:
        x: The input to load the image from. Can be a PIL Image, a string, or
            bytes. If it's a string, it's assumed to be a base64 encoded string
            of bytes, optionally with a `data:<mime>;base64,` prefix; if it's
            bytes, raw encoded image data (e.g., PNG).

    Returns:
       The loaded image as a PIL Image object.

    Raises:
        ValueError: If the input is a malformed data URL, cannot be decoded
            into an image, or is not a PIL Image, a string or bytes.
    """
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, bytes):
        try:
            return Image.open(io.BytesIO(x))
        except Exception as e:
            raise ValueError("Encountered an error when loading image from bytes.") from e
    if not isinstance(x, str):
        raise ValueError(
            f"Expected either a PIL.Image.Image, bytes or a base64 encoded string of bytes, got {type(x).__name__}."
        )

    x = maybe_strip_base64_data_url_prefix(x)
    try:
        image = Image.open(io.BytesIO(base64.b64decode(x.encode("ascii"))))
        return image
    except Exception as e:
        raise ValueError(
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

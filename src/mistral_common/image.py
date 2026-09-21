import base64
import io
import math
import os
from typing import Annotated

import requests
from PIL import Image
from pydantic import BeforeValidator, PlainSerializer, SerializationInfo

from mistral_common import __version__
from mistral_common.exceptions import ImageDecodeException

_IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY = "MISTRAL_COMMON_IMAGE_DOWNLOAD_TIMEOUT"
_DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_S = 10.0


def _validate_timeout(timeout: object, *, error_message: str) -> float:
    r"""Return a positive finite floating-point timeout."""
    if not isinstance(timeout, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(error_message)
    return timeout


def _resolve_env_image_download_timeout() -> float:
    r"""Return the HTTP timeout used when downloading images.

    Reads `MISTRAL_COMMON_IMAGE_DOWNLOAD_TIMEOUT` (seconds). Unset or empty uses 10.
    """
    raw = os.getenv(_IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY)
    if raw is None or raw.strip() == "":
        return _DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_S
    error_message = (
        f"Invalid environment variable {_IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY}: expected a positive finite number of seconds."
    )
    try:
        timeout = float(raw)
    except ValueError as e:
        raise ValueError(error_message) from e
    return _validate_timeout(timeout, error_message=error_message)


def download_image(url: str, timeout: float | None) -> Image.Image:
    r"""Download an image from a URL and return it as a PIL Image.

    Args:
        url: The URL of the image to download.
        timeout: Seconds to wait for the server response. If None, uses the
            `MISTRAL_COMMON_IMAGE_DOWNLOAD_TIMEOUT` environment variable.

    Returns:
       The downloaded image as a PIL Image object.
    """
    if timeout is None:
        timeout = _resolve_env_image_download_timeout()
    else:
        timeout = _validate_timeout(
            timeout,
            error_message=f"timeout must be a positive finite float, got {timeout=}",
        )

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
            f"Pass a larger `timeout` or set the environment variable `{_IMAGE_DOWNLOAD_TIMEOUT_ENV_KEY}` "
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
            of bytes; if it's bytes, raw encoded image data (e.g., PNG).

    Returns:
       The loaded image as a PIL Image object.

    Raises:
        RuntimeError: If the input cannot be decoded into an image.
    """
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, bytes):
        try:
            return Image.open(io.BytesIO(x))
        except Exception:
            raise RuntimeError("Encountered an error when loading image from bytes.")

    try:
        image = Image.open(io.BytesIO(base64.b64decode(x.encode("ascii"))))
        return image
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

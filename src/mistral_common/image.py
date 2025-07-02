import base64
import io
from typing import Union

import requests
from PIL import Image
from pydantic import BeforeValidator, PlainSerializer, SerializationInfo
from typing_extensions import Annotated

from mistral_common import __version__


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
        return img

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error downloading the image from {url}: {e}.")
    except Exception as e:
        raise RuntimeError(f"Error converting to PIL image: {e}")


def maybe_load_image_from_str_or_bytes(x: Union[Image.Image, str, bytes]) -> Image.Image:
    r"""Load an image from a string or bytes.

    If the input is already a PIL Image, return it as is.

    Args:
        x: The input to load the image from. Can be a PIL Image, a string, or bytes.
            If it's a string, it's assumed to be a base64 encoded string of bytes.

    Returns:
       The loaded image as a PIL Image object.
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

    Args:
        im: The image to serialize.
        info: The serialization info.

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

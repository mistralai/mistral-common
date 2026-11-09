from collections.abc import Callable

import pytest

from mistral_common.image import maybe_load_image_from_str_or_bytes
from mistral_common.protocol.instruct.chunk import ImageChunk


@pytest.mark.parametrize(
    "load",
    [
        pytest.param(maybe_load_image_from_str_or_bytes, id="maybe_load_image_from_str_or_bytes"),
        pytest.param(lambda image: ImageChunk(image=image), id="ImageChunk"),
    ],
)
@pytest.mark.parametrize(
    ("image", "match"),
    [
        pytest.param("not base64 at all!!", "starting with 'not base64 at all!!'", id="not-base64"),
        pytest.param("aGVsbG8gd29ybGQ=", "starting with 'aGVsbG8gd29ybGQ='", id="not-an-image"),
        pytest.param(b"notanimage", "from bytes", id="bytes-not-an-image"),
        pytest.param(42, "got int", id="int"),
        pytest.param(None, "got NoneType", id="none"),
    ],
)
def test_load_image_rejects_invalid_input(load: Callable[[object], object], image: object, match: str) -> None:
    # pydantic's ValidationError, raised by ImageChunk, subclasses ValueError.
    with pytest.raises(ValueError, match=match):
        load(image)

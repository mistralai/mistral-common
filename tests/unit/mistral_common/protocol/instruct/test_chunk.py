import base64
import io

import PIL
import pytest

from mistral_common.protocol.instruct.chunk import ImageChunk


def _encoded_image_bytes(format: str) -> bytes:
    image = PIL.Image.new("RGB", (8, 8), color=(1, 2, 3))
    stream = io.BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


@pytest.mark.parametrize("subtype", ["png", "jpeg", "svg+xml", "x-icon", "vnd.microsoft.icon"])
def test_image_chunk_from_openai_strips_data_url_prefix(subtype: str) -> None:
    """A `data:image/<subtype>;base64,` prefix is stripped for every MIME subtype, not only word-character ones."""
    raw_bytes = _encoded_image_bytes("PNG")
    encoded = base64.b64encode(raw_bytes).decode("ascii")

    chunk = ImageChunk.from_openai(
        {"type": "image_url", "image_url": {"url": f"data:image/{subtype};base64,{encoded}"}}
    )

    stream = io.BytesIO()
    chunk.image.save(stream, format="PNG")
    assert stream.getvalue() == raw_bytes


@pytest.mark.parametrize("url", ["https://example.com/image.png", "data:image/png;base64,AAAA"])
def test_image_chunk_from_openai_rejects_undecodable_image(url: str) -> None:
    """Plain URLs and truncated payloads keep raising the image loader error."""
    with pytest.raises(RuntimeError, match="Expected either a PIL.Image.Image or a base64 encoded string of bytes"):
        ImageChunk.from_openai({"type": "image_url", "image_url": {"url": url}})

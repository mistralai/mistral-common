import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from mistral_common.exceptions import ImageDecodeException
from mistral_common.image import download_image, maybe_load_image_from_str_or_bytes
from mistral_common.protocol.instruct.chunk import ImageURLChunk
from mistral_common.tokens.tokenizers.image import image_from_chunk


def _truncated_png() -> bytes:
    r"""Return a PNG with an intact header and incomplete pixel data."""
    stream = BytesIO()
    Image.effect_noise(size=(256, 256), sigma=64).convert("RGB").save(stream, format="PNG")
    data = stream.getvalue()
    return data[: len(data) // 2]


@pytest.mark.parametrize("as_base64", [False, True])
def test_load_image_rejects_truncated_data(as_base64: bool) -> None:
    data = _truncated_png()
    value = base64.b64encode(data).decode("ascii") if as_base64 else data

    with pytest.raises(RuntimeError, match="Encountered an error when loading image"):
        maybe_load_image_from_str_or_bytes(value)


def test_image_data_url_rejects_truncated_data() -> None:
    data = base64.b64encode(_truncated_png()).decode("ascii")
    chunk = ImageURLChunk(image_url=f"data:image/png;base64,{data}")

    with pytest.raises(ValueError, match="payload is not a decodable image"):
        image_from_chunk(chunk=chunk)


def test_download_image_rejects_truncated_data() -> None:
    response = Mock(content=_truncated_png())
    with (
        patch("mistral_common.image.requests.get", return_value=response),
        pytest.raises(ImageDecodeException, match="Error converting to PIL image"),
    ):
        download_image(url="https://example.com/image.png", timeout=10.0)


def test_download_image_accepts_integer_timeout() -> None:
    import io

    stream = io.BytesIO()
    Image.new("RGB", (4, 4)).save(stream, format="PNG")
    response = Mock(content=stream.getvalue())
    with patch("mistral_common.image.requests.get", return_value=response) as mock_get:
        image = download_image(url="https://example.com/image.png", timeout=10)

    assert mock_get.call_args.kwargs["timeout"] == 10.0
    assert image.size == (4, 4)


@pytest.mark.parametrize("timeout", [0, -1.5, float("inf"), float("nan"), True, "10"])
def test_download_image_rejects_invalid_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout must be a positive finite"):
        download_image(url="https://example.com/image.png", timeout=timeout)  # type: ignore[arg-type]


def test_image_file_url_decodes_percent_encoding(tmp_path: Path) -> None:
    r"""A file URI from Path.as_uri() (percent-encoded) must load like the plain path."""
    path = tmp_path / "my image.png"
    Image.new("RGB", (4, 4), color="blue").save(path)

    from_file_uri = image_from_chunk(chunk=ImageURLChunk(image_url=path.as_uri()))

    assert path.as_uri().endswith("/my%20image.png")
    assert from_file_uri.size == (4, 4)

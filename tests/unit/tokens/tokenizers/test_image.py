from pathlib import Path

import pytest
from PIL import Image

from mistral_common.protocol.instruct.chunk import ImageURLChunk
from mistral_common.tokens.tokenizers.image import image_from_chunk


def _save_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (128, 128, 128)).save(path)
    return path


@pytest.mark.parametrize("relative_path", ["image.png", "my images/café.png", "100% #1/image.png"])
def test_image_from_chunk_file_uri_is_decoded(tmp_path: Path, relative_path: str) -> None:
    image_path = _save_image(tmp_path / relative_path)

    image = image_from_chunk(chunk=ImageURLChunk(image_url=image_path.as_uri()))

    assert image.size == (8, 8)


def test_image_from_chunk_file_uri_with_unencoded_path(tmp_path: Path) -> None:
    image_path = _save_image(tmp_path / "image.png")

    image = image_from_chunk(chunk=ImageURLChunk(image_url=f"file://{image_path}"))

    assert image.size == (8, 8)


def test_image_from_chunk_file_uri_with_localhost(tmp_path: Path) -> None:
    image_path = _save_image(tmp_path / "my images" / "image.png")
    url = image_path.as_uri().replace("file://", "file://localhost", 1)

    image = image_from_chunk(chunk=ImageURLChunk(image_url=url))

    assert image.size == (8, 8)


def test_image_from_chunk_file_uri_with_remote_host_is_unsupported() -> None:
    with pytest.raises(ValueError, match="Unsupported file URI host 'remote-host'"):
        image_from_chunk(chunk=ImageURLChunk(image_url="file://remote-host/image.png"))

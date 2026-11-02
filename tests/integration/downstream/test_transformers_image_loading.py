r"""Downstream check with latest Transformers: its image loader also applies EXIF orientation.

Mirrors the vLLM test: transformers.image_utils.load_image must produce the same
upright orientation that mistral-common now produces, so all three stacks agree.

Runs only under scripts/run_downstream_tests.sh (marker: downstream + transformers).
"""

import base64
import io

import pytest
from PIL import Image

pytestmark = [pytest.mark.downstream, pytest.mark.transformers]

transformers = pytest.importorskip("transformers")

from transformers.image_utils import load_image  # noqa: E402

from mistral_common.protocol.instruct.chunk import ImageURLChunk  # noqa: E402
from mistral_common.tokens.tokenizers.image import image_from_chunk  # noqa: E402


def _jpeg_with_orientation(width: int, height: int, orientation: int) -> bytes:
    img = Image.new("RGB", (width, height), (30, 30, 200))
    exif = img.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_exif_orientation_matches_transformers(orientation: int) -> None:
    raw = _jpeg_with_orientation(200, 100, orientation)
    data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

    hf_img = load_image(data_url)
    mc_img = image_from_chunk(ImageURLChunk(image_url=data_url))

    assert mc_img.size == hf_img.size, (
        f"orientation={orientation}: mistral-common {mc_img.size} != transformers {hf_img.size}"
    )

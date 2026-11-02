r"""E2E parity test: mistral-common prompt tokens vs vLLM media loader for EXIF-rotated images.

This is the test requested in mistral-common#331 review: prove that when a client
sends a smartphone photo (EXIF orientation tag 6/8), the [IMG]/[IMG_BREAK]
placeholder grid that mistral-common writes into prompt_token_ids has the same
width/height orientation as the patch grid vLLM's own media loader computes for
the vision encoder. Before the EXIF fix, mistral-common produced the transposed
(landscape) grid while vLLM produced the portrait one.

Runs only under scripts/run_downstream_tests.sh (marker: downstream + vllm).
"""

import base64
import io

import pytest
from PIL import Image

pytestmark = [pytest.mark.downstream, pytest.mark.vllm]

vllm = pytest.importorskip("vllm")

from vllm.multimodal.image import normalize_image  # noqa: E402

from mistral_common.protocol.instruct.chunk import ImageURLChunk, TextChunk  # noqa: E402
from mistral_common.protocol.instruct.messages import UserMessage  # noqa: E402
from mistral_common.protocol.instruct.request import ChatCompletionRequest  # noqa: E402
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # noqa: E402

PATCH_SIZE = 16


def _jpeg_with_orientation(width: int, height: int, orientation: int) -> bytes:
    img = Image.new("RGB", (width, height), (200, 30, 30))
    exif = img.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def _grid_from_tokens(tokens: list[int], img_id: int, break_id: int, end_id: int) -> tuple[int, int]:
    """Count [IMG] columns and rows in the image placeholder block."""
    assert img_id in tokens and end_id in tokens
    start = tokens.index(img_id)
    end = tokens.index(end_id)
    block = tokens[start : end + 1]
    rows = block.count(break_id) or 1
    cols = block.count(img_id) // rows
    return cols, rows


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_exif_orientation_matches_vllm_media_loader(orientation: int) -> None:
    raw = _jpeg_with_orientation(200, 100, orientation)

    # (A) vLLM's serving media loader: bytes -> PIL -> normalize_image (exif_transpose)
    vllm_img = normalize_image(Image.open(io.BytesIO(raw)))
    expected_w = (vllm_img.width - 1) // PATCH_SIZE + 1
    expected_h = (vllm_img.height - 1) // PATCH_SIZE + 1

    # (B) mistral-common prompt token ids for the same bytes sent as image_url
    tokenizer = MistralTokenizer.v7(is_mm=True)
    assert tokenizer.instruct_tokenizer.image_encoder is not None
    special = tokenizer.instruct_tokenizer.image_encoder.special_ids
    data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    request = ChatCompletionRequest(
        messages=[UserMessage(content=[TextChunk(text="describe"), ImageURLChunk(image_url=data_url)])]
    )
    tokenized = tokenizer.encode_chat_completion(request)
    cols, rows = _grid_from_tokens(tokenized.tokens, special.img, special.img_break, special.img_end)

    assert (cols, rows) == (expected_w, expected_h), (
        f"orientation={orientation}: mistral-common grid {cols}w x {rows}h != "
        f"vLLM media loader grid {expected_w}w x {expected_h}h (image {vllm_img.size})"
    )

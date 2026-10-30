import base64
import io

from PIL import Image

from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.fixtures.audio import get_dummy_audio_chunk
from tests.test_tokenizer_v7_audio import get_tekkenizer_with_audio


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color="red").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_truncation_drops_image_of_dropped_message() -> None:
    r"""A message removed by truncation must not leave an orphan image behind."""
    tokenizer = MistralTokenizer.v7(is_mm=True)
    messages = [
        UserMessage(content=[TextChunk(text="describe"), ImageChunk(image=_png_b64())]),
        AssistantMessage(content="answer"),
        UserMessage(content="question"),
    ]
    full = tokenizer.instruct_tokenizer.encode_instruct(InstructRequest(messages=messages))
    assert len(full.images) == 1

    truncated = tokenizer.instruct_tokenizer.encode_instruct(
        InstructRequest(messages=messages, truncate_at_max_tokens=len(full.tokens) - 2)
    )

    assert len(truncated.tokens) < len(full.tokens)
    assert len(truncated.images) == 0


def test_truncation_keeps_media_of_kept_messages() -> None:
    r"""Media of messages that survive truncation must stay in the output."""
    tokenizer = MistralTokenizer.v7(is_mm=True)
    messages = [
        UserMessage(content="filler " * 20),
        AssistantMessage(content="filler"),
        UserMessage(content=[TextChunk(text="describe"), ImageChunk(image=_png_b64())]),
    ]
    full = tokenizer.instruct_tokenizer.encode_instruct(InstructRequest(messages=messages))
    assert len(full.images) == 1

    truncated = tokenizer.instruct_tokenizer.encode_instruct(
        InstructRequest(messages=messages, truncate_at_max_tokens=len(full.tokens) - 2)
    )

    assert len(truncated.tokens) < len(full.tokens)
    assert len(truncated.images) == 1


def test_truncation_drops_audio_of_dropped_message() -> None:
    r"""A message removed by truncation must not leave an orphan audio behind."""
    tokenizer = get_tekkenizer_with_audio()
    messages = [
        UserMessage(content=[TextChunk(text="a"), get_dummy_audio_chunk()]),
        AssistantMessage(content="b"),
        UserMessage(content="c"),
    ]
    full = tokenizer.encode_instruct(InstructRequest(messages=messages))
    assert len(full.audios) == 1

    truncated = tokenizer.encode_instruct(
        InstructRequest(messages=messages, truncate_at_max_tokens=len(full.tokens) - 2)
    )

    assert len(truncated.tokens) < len(full.tokens)
    assert len(truncated.audios) == 0

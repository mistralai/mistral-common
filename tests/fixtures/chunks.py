from PIL import Image

from mistral_common.protocol.instruct.chunk import (
    ContentChunk,
    ImageChunk,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
)
from tests.fixtures.audio import get_dummy_audio_chunk, get_dummy_audio_url_chunk


def get_content_chunk(name: str) -> ContentChunk:
    r"""Return a single instance of the requested content chunk type.

    Args:
        name: One of "text", "image", "image_url", "audio", "audio_url" or "think".

    Returns:
        A content chunk of the requested type.
    """
    chunks: dict[str, ContentChunk] = {
        "text": TextChunk(text="hello"),
        "image": ImageChunk(image=Image.new("RGB", (4, 4), "red")),
        "image_url": ImageURLChunk(image_url="data:image/png;base64,iVBORw0"),
        "audio": get_dummy_audio_chunk(),
        "audio_url": get_dummy_audio_url_chunk(),
        "think": ThinkChunk(thinking="reasoning"),
    }
    return chunks[name]


def get_content_chunks(names: tuple[str, ...]) -> list[ContentChunk]:
    r"""Return a list of content chunks for the requested type names."""
    return [get_content_chunk(name) for name in names]

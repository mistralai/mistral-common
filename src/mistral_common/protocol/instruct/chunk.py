import base64
import io
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator
from typing_extensions import Annotated

from mistral_common.base import MistralBase
from mistral_common.deprecation import warn_once
from mistral_common.image import SerializableImage
from mistral_common.imports import assert_soundfile_installed, is_soundfile_installed

if is_soundfile_installed():
    import soundfile as sf
if TYPE_CHECKING:
    from mistral_common.tokens.tokenizers.audio import Audio


def _detect_audio_format(data: str | bytes) -> str:
    r"""Detect audio format from base64-encoded string or raw bytes.

    Uses soundfile to read only the file header, avoiding full audio decoding.

    Args:
        data: Base64-encoded audio string or raw audio bytes.

    Returns:
        The detected audio format as a lowercase string (e.g. "wav", "mp3").

    Raises:
        ValueError: If the audio format cannot be detected.
    """
    assert_soundfile_installed()

    if isinstance(data, str):
        audio_bytes = base64.b64decode(data)
    else:
        audio_bytes = data

    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except RuntimeError as e:
        raise ValueError("Failed to detect audio format. Verify that the given file is valid wav or mp3.") from e
    fmt: str = info.format.lower()
    return fmt


class ChunkTypes(str, Enum):
    r"""Enum for the types of chunks that can be sent to the model.

    Attributes:
       text: A text chunk.
       image: An image chunk.
       image_url: An image url chunk.
       input_audio: An input audio chunk.
       audio_url: An audio url chunk.

    Examples:
        >>> from mistral_common.protocol.instruct.chunk import ChunkTypes
        >>> chunk_type = ChunkTypes.text
    """

    text = "text"
    image = "image"
    image_url = "image_url"
    input_audio = "input_audio"
    audio_url = "audio_url"
    thinking = "thinking"


class BaseContentChunk(MistralBase):
    r"""Base class for all content chunks.

    Content chunks are used to send different types of content to the model.

    Attributes:
       type: The type of the chunk.
    """

    type: Literal[
        ChunkTypes.text,
        ChunkTypes.image,
        ChunkTypes.image_url,
        ChunkTypes.input_audio,
        ChunkTypes.audio_url,
        ChunkTypes.thinking,
    ]

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the chunk to the OpenAI format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, Any]) -> "BaseContentChunk":
        r"""Converts the OpenAI chunk to the Mistral format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}")


class ImageChunk(BaseContentChunk):
    r"""Image chunk.

    Attributes:
       image: The image to be sent to the model.

    Examples:
        >>> from PIL import Image
        >>> image_chunk = ImageChunk(image=Image.new('RGB', (200, 200), color='blue'))
    """

    type: Literal[ChunkTypes.image] = ChunkTypes.image
    image: SerializableImage
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the chunk to the OpenAI format."""
        base64_image = self.model_dump(include={"image"}, context={"add_format_prefix": True})["image"]
        return {"type": "image_url", "image_url": {"url": base64_image}}

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, Any]) -> "ImageChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        assert openai_chunk.get("type") == "image_url", openai_chunk

        image_url_dict = openai_chunk["image_url"]
        assert isinstance(image_url_dict, dict) and "url" in image_url_dict, image_url_dict

        url = image_url_dict["url"]
        if re.match(r"^data:image/\w+;base64,", url):  # Remove the prefix if it exists
            url = url.split(",")[1]

        return cls.model_validate({"image": url})


class ImageURL(MistralBase):
    r"""Image URL or a base64 encoded image.

    Attributes:
       url: The URL of the image.
       detail: The detail of the image.

    Examples:
       >>> image_url = ImageURL(url="https://example.com/image.png")
    """

    url: str
    detail: str | None = None


class ImageURLChunk(BaseContentChunk):
    r"""Image URL chunk.

    Attributes:
       image_url: The URL of the image or a base64 encoded image to be sent to the model.

    Examples:
        >>> image_url_chunk = ImageURLChunk(image_url="data:image/png;base64,iVBORw0")
    """

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: ImageURL | str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the chunk to the OpenAI format."""
        image_url_dict = {"url": self.get_url()}
        if isinstance(self.image_url, ImageURL) and self.image_url.detail is not None:
            image_url_dict["detail"] = self.image_url.detail

        out_dict: dict[str, Any] = {
            "type": "image_url",
            "image_url": image_url_dict,
        }
        return out_dict

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, Any]) -> "ImageURLChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        return cls.model_validate({"image_url": openai_chunk["image_url"]})


class RawAudio(MistralBase):
    r"""Deprecated: Use `str | bytes` directly. Will be removed in 1.13.0."""

    data: str | bytes
    format: str

    def model_post_init(self, __context: Any) -> None:
        warn_once(
            "RawAudio",
            "RawAudio is deprecated. Use str | bytes directly for audio data. Will be removed in 1.13.0.",
            DeprecationWarning,
            stacklevel=2,
        )

    @classmethod
    def from_audio(cls, audio: "Audio") -> "RawAudio":
        r"""Create a RawAudio instance from an Audio object.

        Args:
            audio: An Audio object containing audio data, format, and duration.

        Returns:
            A RawAudio instance initialized with the audio data.
        """
        format = audio.format
        data = audio.to_base64(format, False)
        return cls(data=data, format=format)

    @field_validator("format")
    def should_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValidationError("`format` should not be empty")

        return v


class AudioURL(MistralBase):
    r"""Audio URL.

    Attributes:
        url: The URL of the audio file.
    """

    url: str


class AudioURLType(str, Enum):
    r"""Enum for the types of audio URLs.

    Attributes:
        url: A URL.
        base64: A base64 encoded audio. Can be prefixed with `data:audio/<format>;base64,`.
        file: A file path.
        file_uri: A file URI (eg. `file:///path/to/file`).
    """

    url = "url"
    base64 = "base64"
    file = "file"
    file_uri = "file_uri"


class AudioURLChunk(BaseContentChunk):
    r"""Audio URL chunk.

    Attributes:
        type: The type of the chunk, which is always `ChunkTypes.audio_url`.
        audio_url: The URL of the audio file.
    """

    type: Literal[ChunkTypes.audio_url] = ChunkTypes.audio_url
    audio_url: str | AudioURL

    @property
    def url(self) -> str:
        if isinstance(self.audio_url, AudioURL):
            return self.audio_url.url
        return self.audio_url

    def get_url_type(self) -> AudioURLType:
        r"""Returns the type of the audio URL.

        Note:
            URLs should be either:
            - a valid URL (http:// or https://)
            - a valid file path (e.g. /path/to/file)
            - a valid file URI (e.g. file:///path/to/file)
            - a base64 encoded audio. It is assumed to be base64 encoded if it is not a valid URL or file path.

        Returns:
            The type of the audio URL.
        """
        url_scheme = urlparse(self.url).scheme
        if url_scheme in {"http", "https"}:
            return AudioURLType.url
        elif url_scheme == "data":
            return AudioURLType.base64
        elif url_scheme == "file":
            return AudioURLType.file_uri

        try:
            url_path = Path(self.url)
            exist_path = url_path.exists()
        except OSError:  # File name too long
            exist_path = False

        if exist_path:
            return AudioURLType.file

        return AudioURLType.base64

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the chunk to the OpenAI format."""
        if isinstance(self.audio_url, AudioURL):
            return self.model_dump()
        else:
            return {"type": self.type, "audio_url": {"url": self.audio_url}}

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, Any]) -> "AudioURLChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        return cls.model_validate_ignore_extra(openai_chunk)


class AudioChunk(BaseContentChunk):
    r"""Audio chunk containing raw audio data.

    Attributes:
        type: The type of the chunk, which is always ChunkTypes.input_audio.
        input_audio: The audio data as a base64-encoded string or raw bytes.

    Examples:
        >>> audio_chunk = AudioChunk(input_audio="base64_encoded_audio_data")
    """

    type: Literal[ChunkTypes.input_audio] = ChunkTypes.input_audio
    input_audio: str | bytes

    @model_validator(mode="before")
    @classmethod
    def _flatten_audio_dict(cls, values: dict[str, Any]) -> dict[str, Any]:
        r"""Extract audio data from a nested dict or legacy RawAudio payload.

        Handles the OpenAI format where `input_audio` is a dict with a
        `data` key (e.g. `{"data": "...", "format": "wav"}`) as well as
        deprecated `RawAudio` instances, flattening them to a plain
        `str | bytes` value.
        """
        if not isinstance(values, dict):
            return values
        raw = values.get("input_audio")
        if isinstance(raw, MistralBase):
            raw = raw.model_dump()
        if isinstance(raw, dict) and "data" in raw:
            values["input_audio"] = raw["data"]
        return values

    @field_validator("input_audio")
    @classmethod
    def should_not_be_empty(cls, v: str | bytes) -> str | bytes:
        r"""Validate that the audio data is not empty."""
        if isinstance(v, str) and not v.strip():
            raise ValidationError("`input_audio` should not be empty.")
        if isinstance(v, bytes) and not v:
            raise ValidationError("`input_audio` should not be empty.")
        return v

    @classmethod
    def from_audio(cls, audio: "Audio") -> "AudioChunk":
        r"""Create an AudioChunk instance from an Audio object.

        Args:
            audio: An Audio object containing audio data.

        Returns:
            An AudioChunk instance initialized with the audio data.
        """
        return cls(input_audio=audio.to_base64(audio.format, False))

    def to_openai(self) -> dict[str, Any]:
        r"""Convert the chunk to the OpenAI format.

        Returns:
            A dictionary representing the audio chunk in the OpenAI format.
        """
        content = self.input_audio.decode("utf-8") if isinstance(self.input_audio, bytes) else self.input_audio
        fmt = _detect_audio_format(self.input_audio)
        return {
            "type": self.type,
            "input_audio": {
                "data": content,
                "format": fmt,
            },
        }

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, Any]) -> "AudioChunk":
        r"""Convert the OpenAI chunk to the Mistral format.

        Args:
            openai_chunk: A dictionary representing the audio chunk in the OpenAI format.

        Returns:
            An AudioChunk instance initialized with the data from the OpenAI chunk.
        """
        return cls.model_validate_ignore_extra(openai_chunk)


class TextChunk(BaseContentChunk):
    r"""Text chunk.

    Attributes:
      text: The text to be sent to the model.

    Examples:
        >>> text_chunk = TextChunk(text="Hello, how can I help you?")
    """

    type: Literal[ChunkTypes.text] = ChunkTypes.text
    text: str

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the chunk to the OpenAI format."""
        return self.model_dump()

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, Any]) -> "TextChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        return cls.model_validate_ignore_extra(openai_chunk)


class ThinkChunk(BaseContentChunk):
    r"""Thinking chunk.

    Attributes:
        type: The type of the chunk, which is always ChunkTypes.thinking.
        thinking: The list of text chunks of the thinking.
        closed: Whether the thinking chunk is closed or not.
    """

    type: Literal[ChunkTypes.thinking] = ChunkTypes.thinking
    thinking: str
    closed: bool = Field(default=True, description="Whether the thinking chunk is closed or not.")

    def to_openai(self) -> dict[str, Any]:
        r"""Converts the chunk to the OpenAI format."""
        return self.model_dump()

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, Any]) -> "ThinkChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        return cls.model_validate_ignore_extra(openai_chunk)


ContentChunk = Annotated[
    TextChunk | ImageChunk | ImageURLChunk | AudioChunk | AudioURLChunk | ThinkChunk, Field(discriminator="type")
]
UserContentChunk = Annotated[
    TextChunk | ImageChunk | ImageURLChunk | AudioChunk | AudioURLChunk, Field(discriminator="type")
]


# The OpenAI Responses API spells text/image content types differently from the
# Chat Completions API that ``ChunkTypes`` mirrors. Map the Responses spellings
# onto their Chat Completions equivalents so structured Responses input/output
# is accepted instead of raising e.g. "'input_text' is not a valid ChunkTypes".
# (``input_image`` carries ``image_url`` as a bare string, which
# ``ImageURLChunk`` already accepts.)
_RESPONSES_CONTENT_TYPE_ALIASES: dict[str, ChunkTypes] = {
    "input_text": ChunkTypes.text,
    "output_text": ChunkTypes.text,
    "input_image": ChunkTypes.image_url,
}


def _convert_openai_content_chunks(openai_content_chunks: dict[str, Any]) -> ContentChunk:
    content_type_str = openai_content_chunks.get("type")

    if content_type_str is None:
        raise ValueError("Content chunk must have a type field.")

    # Normalize OpenAI Responses content types to the Chat Completions spellings.
    # The discriminated chunk models require the canonical ``type``, so rewrite
    # the field too.
    aliased_type = _RESPONSES_CONTENT_TYPE_ALIASES.get(content_type_str)
    if aliased_type is not None:
        openai_content_chunks = {**openai_content_chunks, "type": aliased_type.value}
        content_type_str = aliased_type.value

    content_type = ChunkTypes(content_type_str)

    if content_type == ChunkTypes.text:
        return TextChunk.from_openai(openai_content_chunks)
    elif content_type == ChunkTypes.image_url:
        return ImageURLChunk.from_openai(openai_content_chunks)
    elif content_type == ChunkTypes.image:
        return ImageChunk.from_openai(openai_content_chunks)
    elif content_type == ChunkTypes.input_audio:
        return AudioChunk.from_openai(openai_content_chunks)
    elif content_type == ChunkTypes.audio_url:
        return AudioURLChunk.from_openai(openai_content_chunks)
    elif content_type == ChunkTypes.thinking:
        return ThinkChunk.from_openai(openai_content_chunks)
    else:
        raise ValueError(f"Unknown content chunk type: {content_type}")

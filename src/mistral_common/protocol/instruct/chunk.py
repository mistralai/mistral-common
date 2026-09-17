import base64
import io
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Annotated

from mistral_common.base import MistralBase
from mistral_common.deprecation import warn_once
from mistral_common.image import SerializableImage
from mistral_common.imports import assert_soundfile_installed, is_soundfile_installed

if is_soundfile_installed():
    import soundfile as sf
if TYPE_CHECKING:
    from mistral_common.tokens.tokenizers.audio import Audio


def _strip_audio_data_url_prefix(data: str) -> str:
    r"""Remove the optional base64 audio data URL prefix."""
    if re.match(r"^data:audio/\w+;base64,", data):
        return data.split(",", 1)[1]
    return data


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
        audio_bytes = base64.b64decode(_strip_audio_data_url_prefix(data))
    else:
        audio_bytes = data

    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except RuntimeError as e:
        raise ValueError("Failed to detect audio format. Verify that the given file is valid wav or mp3.") from e
    fmt: str = info.format.lower()
    return fmt


class ChunkTypes(str, Enum):
    r"""Enum of the types of content chunks that can be sent to the model.

    Attributes:
        text: A plain text chunk.
        image: An image provided as raw data (PIL image, base64, URL or path).
        image_url: An image referenced by URL or base64 data URL.
        input_audio: Audio provided inline as base64 or raw bytes.
        audio_url: Audio referenced by URL, file path, file URI or base64.
        thinking: A reasoning/thinking chunk from the assistant.

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

    Content chunks represent a piece of multimodal content (text, image, audio,
    thinking) inside a message. A message's content can be a string or a list
    of these chunks.

    Attributes:
        type: The chunk type, used as the pydantic discriminator.
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
        r"""Convert this chunk to the OpenAI format.

        Must be implemented by concrete subclasses.

        Returns:
            Dictionary matching OpenAI's content chunk schema.

        Raises:
            NotImplementedError: Always, as this is an abstract method.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, Any]) -> "BaseContentChunk":
        r"""Create a chunk instance from OpenAI format.

        Must be implemented by concrete subclasses.

        Args:
            openai_chunk: Dictionary matching OpenAI's content chunk schema.

        Returns:
            Chunk instance of the appropriate subclass.

        Raises:
            NotImplementedError: Always, as this is an abstract method.
        """
        raise NotImplementedError(f"from_openai method not implemented for {cls.__name__}")


class ImageChunk(BaseContentChunk):
    r"""Image content provided as raw data.

    The image is processed (resized, serialized) at tokenization time by the
    image encoder.

    Attributes:
        image: The image to send to the model. Accepts a PIL image, a URL
            (http/https), a base64-encoded string (optionally with a
            data:...;base64, prefix), or a local file path.

    Examples:
        >>> from PIL import Image
        >>> image_chunk = ImageChunk(image=Image.new('RGB', (200, 200), color='blue'))
    """

    type: Literal[ChunkTypes.image] = ChunkTypes.image
    image: SerializableImage
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_openai(self) -> dict[str, Any]:
        r"""Convert this chunk to the OpenAI `image_url` format.

        Returns:
            Dictionary with "type" set to `image_url` and the image as a
            base64 data URL under `image_url` -> "url".
        """
        base64_image = self.model_dump(include={"image"}, context={"add_format_prefix": True})["image"]
        return {"type": "image_url", "image_url": {"url": base64_image}}

    @classmethod
    def from_openai(cls, openai_chunk: dict[str, Any]) -> "ImageChunk":
        r"""Create an ImageChunk from an OpenAI `image_url` chunk.

        Args:
            openai_chunk: Dictionary with "type" set to `image_url` and an
                `image_url` -> "url" entry (URL or base64 data URL).

        Returns:
            An ImageChunk with the base64 prefix stripped if present.

        Raises:
            AssertionError: If the chunk type is not `image_url` or the
                `image_url` entry is malformed.
        """
        assert openai_chunk.get("type") == "image_url", openai_chunk

        image_url_dict = openai_chunk["image_url"]
        assert isinstance(image_url_dict, dict) and "url" in image_url_dict, image_url_dict

        url = image_url_dict["url"]
        if re.match(r"^data:image/\w+;base64,", url):  # Remove the prefix if it exists
            url = url.split(",")[1]

        return cls.model_validate({"image": url})


class ImageURL(MistralBase):
    r"""Image URL with optional detail level.

    Attributes:
        url: The URL of the image, or a base64-encoded image (optionally with
            a data:...;base64, prefix).
        detail: Optional detail level hint for image processing (e.g., "high",
            "low", "auto"). If `None`, the default is used.

    Examples:
       >>> image_url = ImageURL(url="https://example.com/image.png")
    """

    url: str
    detail: str | None = None


class ImageURLChunk(BaseContentChunk):
    r"""Image content referenced by URL or base64 data URL.

    Unlike ImageChunk, the image is not loaded at request construction time;
    the URL is resolved at tokenization time.

    Attributes:
        image_url: The image reference. Either an ImageURL instance (allowing a
            detail hint) or a plain URL/base64 string.

    Examples:
        >>> image_url_chunk = ImageURLChunk(image_url="data:image/png;base64,iVBORw0")
    """

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: ImageURL | str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        r"""Return the underlying image URL string.

        Returns:
            The URL regardless of whether `image_url` is an ImageURL or a plain string.
        """
        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url

    def to_openai(self) -> dict[str, Any]:
        r"""Convert this chunk to the OpenAI format.

        Returns:
            Dictionary with "type" set to `image_url` and the URL (plus
            optional "detail") under `image_url`.
        """
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
        r"""Create an ImageURLChunk from an OpenAI chunk.

        Args:
            openai_chunk: Dictionary with an `image_url` entry (dict or string).

        Returns:
            An ImageURLChunk with the `image_url` parsed from the OpenAI format.
        """
        return cls.model_validate({"image_url": openai_chunk["image_url"]})


class RawAudio(MistralBase):
    r"""Audio data with an explicit format.

    Deprecated: Use `str | bytes` directly. Will be removed in 1.13.0.

    Attributes:
        data: The audio data as raw bytes or a base64-encoded string.
        format: The audio format (e.g., "wav", "mp3"). Must not be empty.
    """

    data: str | bytes
    format: str

    def model_post_init(self, __context: Any) -> None:
        r"""Emit a one-shot deprecation warning for `RawAudio`."""
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
        r"""Reject empty format strings.

        Returns:
            The unchanged format string if non-empty.

        Raises:
            ValueError: If `format` is empty or whitespace.
        """
        if not v.strip():
            raise ValueError("`format` should not be empty")

        return v


class AudioURL(MistralBase):
    r"""Audio URL reference.

    Attributes:
        url: The URL of the audio file.
    """

    url: str


class AudioURLType(str, Enum):
    r"""Enum for the kinds of audio URL references.

    Attributes:
        url: An http(s) URL.
        base64: A base64-encoded audio string. May be prefixed with
            `data:audio/<format>;base64,`.
        file: A local file path (e.g., /path/to/file).
        file_uri: A file URI (e.g., `file:///path/to/file`).
    """

    url = "url"
    base64 = "base64"
    file = "file"
    file_uri = "file_uri"


class AudioURLChunk(BaseContentChunk):
    r"""Audio content referenced by URL, path, file URI or base64.

    The URL kind is resolved lazily by `get_url_type`; the audio itself is
    loaded at tokenization time.

    Attributes:
        audio_url: The audio reference. Either an AudioURL instance or a
            plain string (URL, file path, file URI, or base64).

    Examples:
        >>> audio_url_chunk = AudioURLChunk(audio_url="https://example.com/audio.mp3")
    """

    type: Literal[ChunkTypes.audio_url] = ChunkTypes.audio_url
    audio_url: str | AudioURL

    @property
    def url(self) -> str:
        r"""The audio URL string.

        Returns:
            The URL regardless of whether `audio_url` is an `AudioURL` or a plain string.
        """
        if isinstance(self.audio_url, AudioURL):
            return self.audio_url.url
        return self.audio_url

    def get_url_type(self) -> AudioURLType:
        r"""Detect the kind of the referenced audio URL.

        Note:
            URLs should be either:
            - a valid URL (http:// or https://)
            - a valid file path (e.g. /path/to/file)
            - a valid file URI (e.g. file:///path/to/file)
            - a base64 encoded audio. It is assumed to be base64 encoded if it is not a valid URL or file path.

        Returns:
            The detected AudioURLType for this chunk's URL.
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
    r"""Audio content provided inline as base64 or raw bytes.

    Attributes:
        input_audio: The audio data as a base64-encoded string (optionally
            with a data:audio/<format>;base64, prefix) or raw bytes. The format
            is detected at serialization time.

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

        Returns:
            The values with a flattened `input_audio` key, or the input unchanged
            if there is nothing to flatten.
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
            raise ValueError("`input_audio` should not be empty.")
        if isinstance(v, bytes) and not v:
            raise ValueError("`input_audio` should not be empty.")
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
        if isinstance(self.input_audio, bytes):
            content = base64.b64encode(self.input_audio).decode("utf-8")
        else:
            content = _strip_audio_data_url_prefix(self.input_audio)
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
    r"""Reasoning/thinking content from the assistant.

    ThinkChunks represent chain-of-thought text the model produces before the
    final answer. In AssistantMessage content they must appear before any
    other chunk.

    Attributes:
        thinking: The thinking text content.
        closed: If `True` (default), the thinking section is complete. If `False`,
            the thinking is ongoing (e.g., mid-stream during generation).
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


def _convert_openai_content_chunks(openai_content_chunks: dict[str, Any]) -> ContentChunk:
    r"""Convert an OpenAI content chunk dict to the matching ContentChunk subclass.

    Args:
        openai_content_chunks: Dictionary with a "type" key matching a
            ChunkTypes value.

    Returns:
        The ContentChunk instance corresponding to the OpenAI chunk type.

    Raises:
        ValueError: If the chunk has no "type" field or the type is unknown.
    """
    content_type_str = openai_content_chunks.get("type")

    if content_type_str is None:
        raise ValueError("Content chunk must have a type field.")

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

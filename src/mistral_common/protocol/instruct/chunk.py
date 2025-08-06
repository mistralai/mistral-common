import re
from enum import Enum
from pathlib import Path
from typing import Dict, Literal, Optional, Union
from urllib.parse import urlparse

from pydantic import ConfigDict, Field, ValidationError, field_validator
from typing_extensions import Annotated

from mistral_common.audio import EXPECTED_FORMAT_VALUES, Audio
from mistral_common.base import MistralBase
from mistral_common.image import SerializableImage


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

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format.

        Should be implemented by subclasses.
        """
        raise NotImplementedError(f"to_openai method not implemented for {type(self).__name__}")

    @classmethod
    def from_openai(cls, openai_chunk: Dict[str, Union[str, Dict[str, str]]]) -> "BaseContentChunk":
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

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format."""
        base64_image = self.model_dump(include={"image"}, context={"add_format_prefix": True})["image"]
        return {"type": "image_url", "image_url": {"url": base64_image}}

    @classmethod
    def from_openai(cls, openai_chunk: Dict[str, Union[str, Dict[str, str]]]) -> "ImageChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        assert openai_chunk.get("type") == "image_url", openai_chunk

        image_url_dict = openai_chunk["image_url"]
        assert isinstance(image_url_dict, dict) and "url" in image_url_dict, image_url_dict

        if re.match(r"^data:image/\w+;base64,", image_url_dict["url"]):  # Remove the prefix if it exists
            image_url_dict["url"] = image_url_dict["url"].split(",")[1]

        return cls.model_validate({"image": image_url_dict["url"]})


class ImageURL(MistralBase):
    r"""Image URL or a base64 encoded image.

    Attributes:
       url: The URL of the image.
       detail: The detail of the image.

    Examples:
       >>> image_url = ImageURL(url="https://example.com/image.png")
    """

    url: str
    detail: Optional[str] = None


class ImageURLChunk(BaseContentChunk):
    r"""Image URL chunk.

    Attributes:
       image_url: The URL of the image or a base64 encoded image to be sent to the model.

    Examples:
        >>> image_url_chunk = ImageURLChunk(image_url="data:image/png;base64,iVBORw0")
    """

    type: Literal[ChunkTypes.image_url] = ChunkTypes.image_url
    image_url: Union[ImageURL, str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_url(self) -> str:
        if isinstance(self.image_url, ImageURL):
            return self.image_url.url
        return self.image_url

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format."""
        image_url_dict = {"url": self.get_url()}
        if isinstance(self.image_url, ImageURL) and self.image_url.detail is not None:
            image_url_dict["detail"] = self.image_url.detail

        out_dict: Dict[str, Union[str, Dict[str, str]]] = {
            "type": "image_url",
            "image_url": image_url_dict,
        }
        return out_dict

    @classmethod
    def from_openai(cls, openai_chunk: Dict[str, Union[str, Dict[str, str]]]) -> "ImageURLChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        return cls.model_validate({"image_url": openai_chunk["image_url"]})


class RawAudio(MistralBase):
    r"""Base64 encoded audio data.

    This class represents raw audio data encoded in base64 format.

    Attributes:
        data: The base64 encoded audio data, which can be a string or bytes.
        format: The format of the audio data.

    Examples:
        >>> audio = RawAudio(data="base64_encoded_audio_data", format="mp3")
    """

    data: Union[str, bytes]
    format: str

    @classmethod
    def from_audio(cls, audio: Audio) -> "RawAudio":
        """Creates a RawAudio instance from an Audio object.

        Args:
            audio: An Audio object containing audio data, format, and duration.

        Returns:
            An AudioChunk instance initialized with the audio data.
        """
        format = audio.format
        data = audio.to_base64(format, False)

        return cls(data=data, format=format)

    @field_validator("format")
    def should_not_be_empty(cls, v: str) -> str:
        if v not in EXPECTED_FORMAT_VALUES:
            raise ValidationError(f"`format` should be one of {EXPECTED_FORMAT_VALUES}. Got: {v}`")

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
    audio_url: Union[str, AudioURL]

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

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format."""
        if isinstance(self.audio_url, AudioURL):
            return self.model_dump()
        else:
            return {"type": self.type, "audio_url": {"url": self.audio_url}}

    @classmethod
    def from_openai(cls, openai_chunk: Dict[str, Union[str, Dict[str, str]]]) -> "AudioURLChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        return cls.model_validate(openai_chunk)


class AudioChunk(BaseContentChunk):
    r"""Audio chunk containing raw audio data.

    This class represents a chunk of audio data that can be used as input.

    Attributes:
        type: The type of the chunk, which is always ChunkTypes.input_audio.
        input_audio: The RawAudio object containing the audio data.

    Examples:
        >>> audio_chunk = AudioChunk(input_audio=RawAudio(data="base64_encoded_audio_data", format="mp3"))
    """

    type: Literal[ChunkTypes.input_audio] = ChunkTypes.input_audio
    input_audio: RawAudio

    @field_validator("input_audio")
    def should_not_be_empty(cls, v: RawAudio) -> RawAudio:
        if not v.data.strip():
            raise ValidationError(f"`InputAudio` should not be empty. Got: {v}`")

        return v

    @classmethod
    def from_audio(cls, audio: Audio) -> "AudioChunk":
        r"""Creates an AudioChunk instance from an Audio object.

        Args:
            audio: An Audio object containing audio data.

        Returns:
            An AudioChunk instance initialized with the audio data.
        """
        return cls(input_audio=RawAudio.from_audio(audio))

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format.

        Returns:
            A dictionary representing the audio chunk in the OpenAI format.
        """
        content = (
            self.input_audio.data.decode("utf-8") if isinstance(self.input_audio.data, bytes) else self.input_audio.data
        )
        return {
            "type": self.type,
            "input_audio": RawAudio(data=content, format=self.input_audio.format).model_dump(),
        }

    @classmethod
    def from_openai(cls, openai_chunk: Dict[str, Union[str, Dict[str, str]]]) -> "AudioChunk":
        r"""Converts the OpenAI chunk to the Mistral format.

        Args:
            openai_chunk: A dictionary representing the audio chunk in the OpenAI format.

        Returns:
            An AudioChunk instance initialized with the data from the OpenAI chunk.
        """
        return cls.model_validate(openai_chunk)


class TextChunk(BaseContentChunk):
    r"""Text chunk.

    Attributes:
      text: The text to be sent to the model.

    Examples:
        >>> text_chunk = TextChunk(text="Hello, how can I help you?")
    """

    type: Literal[ChunkTypes.text] = ChunkTypes.text
    text: str

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format."""
        return self.model_dump()

    @classmethod
    def from_openai(cls, openai_chunk: Dict[str, Union[str, Dict[str, str]]]) -> "TextChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        return cls.model_validate(openai_chunk)


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

    def to_openai(self) -> Dict[str, Union[str, Dict[str, str]]]:
        r"""Converts the chunk to the OpenAI format."""
        return self.model_dump()

    @classmethod
    def from_openai(cls, openai_chunk: Dict[str, Union[str, Dict[str, str]]]) -> "ThinkChunk":
        r"""Converts the OpenAI chunk to the Mistral format."""
        return cls.model_validate(openai_chunk)


ContentChunk = Annotated[
    Union[TextChunk, ImageChunk, ImageURLChunk, AudioChunk, AudioURLChunk, ThinkChunk], Field(discriminator="type")
]
UserContentChunk = Annotated[
    Union[TextChunk, ImageChunk, ImageURLChunk, AudioChunk, AudioURLChunk], Field(discriminator="type")
]


def _convert_openai_content_chunks(openai_content_chunks: Dict[str, Union[str, Dict[str, str]]]) -> ContentChunk:
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

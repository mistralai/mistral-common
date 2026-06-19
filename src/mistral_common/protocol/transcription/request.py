import io
from enum import Enum
from typing import Any

from pydantic import Field, model_validator
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.base import MistralBase
from mistral_common.imports import assert_soundfile_installed, is_soundfile_installed
from mistral_common.protocol.base import BaseCompletionRequest
from mistral_common.protocol.instruct.chunk import _detect_audio_format
from mistral_common.tokens.tokenizers.audio import Audio

if is_soundfile_installed():
    import soundfile as sf


class StreamingMode(str, Enum):
    DISABLED = "disabled"
    ONLINE = "online"
    OFFLINE = "offline"


# Fields of `TranscriptionRequest` that are specific to mistral-common / Mistral's
# transcription API and are not part of the OpenAI transcription API. They are dropped
# by `TranscriptionRequest.to_openai`.
_MISTRAL_TRANSCRIPTION_ONLY_FIELDS: tuple[str, ...] = (
    "id",
    "max_tokens",
    "strict_audio_validation",
    "streaming",
    "target_streaming_delay_ms",
)


class TranscriptionRequest(BaseCompletionRequest):
    r"""A class representing a request for audio transcription.

    This class handles the conversion of audio data into a format suitable for transcription
    using the OpenAI API. It includes methods to convert the request to and from the OpenAI format.

    Attributes:
        id: An optional identifier for the transcription request.
        model: The model to be used for transcription.
        audio: The audio data to be transcribed.
        language: The language of the input audio in ISO-639-1 format (optional).
        strict_audio_validation: A flag indicating whether to perform strict validation of the audio data.
    """

    id: str | None = None
    model: str | None = None
    audio: str | bytes
    language: LanguageAlpha2 | None = Field(
        None,
        description=(
            "The language of the input audio. Supplying the input language "
            "in ISO-639-1 format will improve language adherence."
        ),
    )
    strict_audio_validation: bool = True
    streaming: StreamingMode = Field(
        default=StreamingMode.DISABLED,
        description=(
            "Whether to enable streaming for the transcription request. Online "
            "streaming means the audio is streamed to the server and the transcription is "
            "streamed back. Offline streaming means the audio is passed in one go to the server."
        ),
    )
    target_streaming_delay_ms: int | None = Field(
        None,
        description=(
            "When streaming is enabled, the target streaming delay (in milli-seconds). "
            "This controls how much latency will the model be requested to target after "
            "it hears a word. Note: this is not supported by all models and model targets "
            "the target but may not strictly meet it."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _flatten_audio_dict(cls, values: dict[str, Any]) -> dict[str, Any]:
        r"""Extract audio data from a nested dict or legacy RawAudio payload."""
        if not isinstance(values, dict):
            return values
        raw = values.get("audio")
        if isinstance(raw, MistralBase):
            raw = raw.model_dump()
        if isinstance(raw, dict) and "data" in raw:
            values["audio"] = raw["data"]
        return values

    def to_openai(self, exclude: tuple = (), **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        r"""Convert the transcription request into the OpenAI format.

        This method prepares the transcription request data into the OpenAI-compatible transcription
        format consumed by OpenAI-compatible servers. It handles the conversion of audio data and
        additional parameters into the required format.

        Note that "OpenAI" here refers to the de facto request format rather than OpenAI's hosted API:
        the payload may include extension fields (e.g. ``seed`` and ``top_p``) that are supported by
        OpenAI-compatible servers such as vLLM but are not part of OpenAI's hosted transcription API.
        Pass them via ``exclude`` if you need a strictly OpenAI-valid payload.

        Fields that are specific to mistral-common / Mistral's transcription API and are not part of
        the OpenAI-compatible transcription format are always dropped (see
        ``_MISTRAL_TRANSCRIPTION_ONLY_FIELDS``).

        Args:
            exclude: Additional fields to exclude from the conversion, on top of the mistral-specific
                fields that are always dropped.
            kwargs: Additional parameters to be added to the request.

        Returns:
            The request in the OpenAI-compatible transcription format.

        Raises:
            ImportError: If the required soundfile library is not installed.
        """
        openai_request: dict[str, Any] = self.model_dump(exclude={"audio"})

        assert_soundfile_installed()

        if isinstance(self.audio, bytes):
            buffer = io.BytesIO(self.audio)
            fmt = _detect_audio_format(self.audio)
            buffer.seek(0)
        else:
            assert isinstance(self.audio, str)
            audio = Audio.from_base64(self.audio)
            assert audio.format is not None
            fmt = audio.format.lower()

            buffer = io.BytesIO()
            sf.write(buffer, audio.audio_array, audio.sampling_rate, format=audio.format)
            # reset cursor to beginning
            buffer.seek(0)

        # OpenAI's client uses the filename extension from .name to set the Content-Type.
        buffer.name = f"audio.{fmt}"

        openai_request["file"] = buffer
        openai_request["seed"] = openai_request.pop("random_seed")
        openai_request.update(kwargs)

        # Drop fields that are specific to mistral-common / Mistral's transcription API
        # and are not part of the OpenAI transcription API.
        default_exclude = _MISTRAL_TRANSCRIPTION_ONLY_FIELDS + exclude
        for k in default_exclude:
            openai_request.pop(k, None)

        return openai_request

    @classmethod
    def from_openai(cls, openai_request: dict[str, Any], strict: bool = False) -> "TranscriptionRequest":
        r"""Create a TranscriptionRequest instance from an OpenAI request dictionary.

        This method converts an OpenAI request dictionary into a TranscriptionRequest instance,
        handling the conversion of audio data and other parameters.

        Args:
            openai_request: The OpenAI request dictionary.
            strict: A flag indicating whether to perform strict validation of the audio data.

        Returns:
           An instance of TranscriptionRequest.
        """
        file = openai_request.get("file")
        seed = openai_request.get("seed")
        converted_dict = {
            k: v
            for k, v in openai_request.items()
            if (k in cls.model_fields and not (v is None and k in ["temperature", "top_p"]))
        }

        assert file is not None, file
        if isinstance(file, io.BytesIO):
            audio_bytes = file.getvalue()
        else:
            # for example if file is UploadFile, this should work
            audio_bytes = file.file.read()

        audio = Audio.from_bytes(audio_bytes, strict=strict)
        audio_str = audio.to_base64(audio.format)

        converted_dict["audio"] = audio_str
        converted_dict["random_seed"] = seed
        return cls(**converted_dict)

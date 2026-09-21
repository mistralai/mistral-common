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
    r"""Streaming behavior for a transcription request.

    Attributes:
        DISABLED: No streaming; the full transcript is returned once complete.
        ONLINE: Audio is streamed to the server and the transcription is streamed
            back incrementally.
        OFFLINE: Audio is sent in one go and the transcription is streamed back.
    """

    DISABLED = "disabled"
    ONLINE = "online"
    OFFLINE = "offline"


class TranscriptionRequest(BaseCompletionRequest):
    r"""A request for audio transcription.

    Attributes:
        id: Optional identifier for this transcription request.
        model: The model to use for transcription. If `None`, the serving side
            default transcription model is used.
        audio: Audio data to transcribe. Either raw audio bytes or a base64-encoded
            string (decoded automatically).
        language: Language of the input audio in ISO-639-1 format (e.g., "en").
            If provided, improves language adherence of the transcript.
        strict_audio_validation: If `True` (default), audio data is validated against
            the expected format and raises on invalid input. If `False`, best-effort
            decoding is attempted.
        streaming: The streaming mode for the request. See StreamingMode.
        target_streaming_delay_ms: When streaming is enabled, the target delay in
            milliseconds between hearing a word and producing its transcript. This
            is a request, not a guarantee; unsupported models ignore it.
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
        r"""Extract audio data from a nested dict or legacy RawAudio payload.

        Accepts audio provided as {"audio": {"data": ...}} or a legacy RawAudio
        model and flattens it to a plain audio value.

        Args:
            values: The raw input values being validated.

        Returns:
            The values with a flattened "audio" key, or the input unchanged if
            there is nothing to flatten.
        """
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

        Audio is converted into an in-memory file buffer with the correct format
        extension, and mistral-specific fields are dropped.

        Args:
            exclude: Extra field names to exclude from the output, in addition to
                the mistral-specific defaults ("id", `max_tokens`,
                `strict_audio_validation`, "streaming").
            kwargs: Additional OpenAI parameters merged into the output.

        Returns:
            The request in the OpenAI format, with the audio under the "file" key
            and `random_seed` renamed to "seed".

        Raises:
            ImportError: If soundfile is not installed.
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

        # remove mistral-specific
        # TODO: revisit which fields to expose in the OpenAI format
        default_exclude = ("id", "max_tokens", "strict_audio_validation", "streaming")
        default_exclude += exclude
        for k in default_exclude:
            openai_request.pop(k, None)

        return openai_request

    @classmethod
    def from_openai(cls, openai_request: dict[str, Any], strict: bool = False) -> "TranscriptionRequest":
        r"""Create a TranscriptionRequest from an OpenAI request dictionary.

        The "file" entry is read (BytesIO or file-like object with a .file
        attribute), decoded via Audio, and re-encoded as a base64 string.

        Args:
            openai_request: Dictionary matching OpenAI's transcription request
                schema. Must contain a "file" entry.
            strict: If `True`, audio data is strictly validated during decoding.

        Returns:
            A TranscriptionRequest instance with the audio base64-encoded.

        Raises:
            AssertionError: If no "file" entry is present.
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

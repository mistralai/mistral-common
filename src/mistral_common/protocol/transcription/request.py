import io
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.audio import Audio
from mistral_common.imports import assert_soundfile_installed, is_soundfile_installed
from mistral_common.protocol.base import BaseCompletionRequest
from mistral_common.protocol.instruct.chunk import RawAudio

if is_soundfile_installed():
    import soundfile as sf


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

    id: Optional[str] = None
    model: Optional[str] = None
    audio: RawAudio
    language: Optional[LanguageAlpha2] = Field(
        ...,
        description=(
            "The language of the input audio. Supplying the input language "
            "in ISO-639-1 format will improve language adherence."
        ),
    )
    strict_audio_validation: bool = True

    def to_openai(self, exclude: tuple = (), **kwargs: Any) -> Dict[str, List[Dict[str, Any]]]:
        r"""Convert the transcription request into the OpenAI format.

        This method prepares the transcription request data for compatibility with the OpenAI API.
        It handles the conversion of audio data and additional parameters into the required format.

        Args:
            exclude: Fields to exclude from the conversion.
            kwargs: Additional parameters to be added to the request.

        Returns:
            The request in the OpenAI format.

        Raises:
            ImportError: If the required soundfile library is not installed.
        """
        openai_request: Dict[str, Any] = self.model_dump(exclude={"audio"})

        assert_soundfile_installed()

        if isinstance(self.audio.data, bytes):
            buffer = io.BytesIO(self.audio.data)
        else:
            assert isinstance(self.audio.data, str)
            audio = Audio.from_base64(self.audio.data)

            buffer = io.BytesIO()
            sf.write(buffer, audio.audio_array, audio.sampling_rate, format=audio.format)
            # reset cursor to beginning
            buffer.seek(0)

        openai_request["file"] = buffer
        openai_request["seed"] = openai_request.pop("random_seed")
        openai_request.update(kwargs)

        # remove mistral-specific
        default_exclude = ("id", "max_tokens", "strict_audio_validation")
        default_exclude += exclude
        for k in default_exclude:
            openai_request.pop(k, None)

        return openai_request

    @classmethod
    def from_openai(cls, openai_request: Dict[str, Any], strict: bool = False) -> "TranscriptionRequest":
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
        raw_audio = RawAudio(data=audio_str, format=audio.format)

        converted_dict["audio"] = raw_audio
        converted_dict["random_seed"] = seed
        return cls(**converted_dict)

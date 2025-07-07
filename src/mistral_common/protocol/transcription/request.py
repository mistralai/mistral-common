from typing import Optional, Any, Dict, List
import base64
import io

from pydantic import Field
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.protocol.base import BaseCompletionRequest
from mistral_common.audio import Audio, is_soundfile_installed
from mistral_common.protocol.instruct.messages import AudioChunk


class TranscriptionRequest(BaseCompletionRequest):
    id: Optional[str] = None
    model: str
    audio: AudioChunk
    language: Optional[LanguageAlpha2] = Field(
        ...,
        description=(
            "The language of the input audio. Supplying the input language "
            "in ISO-639-1 format will improve accuracy and latency."
        ),
    )

    def to_openai(self, **kwargs: Any) -> Dict[str, List[Dict[str, Any]]]:
        r"""Convert the ranscription request into the OpenAI format.

        Args:
            kwargs: Additional parameters to be added to the request.

        Returns:
            The request in the OpenAI format.
        """  # noqa: E501
        if not is_soundfile_installed():
            raise ImportError(
                "soundfile is required for this function. Install it with 'pip install mistral-common[soundfile]'"
            )

        import soundfile as sf

        openai_request: Dict[str, Any] = self.model_dump(exclude="audio")
        audio = Audio.from_base64(self.audio.input_audio.data)

        buffer = io.BytesIO()
        sf.write(buffer, audio.audio_array, audio.sampling_rate, format=audio.format)
        # reset cursor to beginning
        buffer.seek(0)

        openai_request["file"] = buffer
        openai_request["seed"] = openai_request.pop("random_seed")
        openai_request.update(kwargs)

        # remove mistral-specific
        openai_request.pop("id", None)
        openai_request.pop("max_tokens", None)

        return openai_request

    @classmethod
    def from_openai(cls, openai_request: Dict[str, Any], strict: bool = False) -> "TranscriptionRequest":
        file = openai_request.get("file")
        seed = openai_request.get("seed")
        converted_dict = {k: v for k,v in openai_request.items() if (k in cls.model_fields and not (v is None and k in ["temperature", "top_p"]))}

        

        if isinstance(file, io.BytesIO):
            _bytes = file.getvalue()
        else:
            # for example if file is UploadFile, this should work
            _bytes = file.file.read()

        _audio = Audio._from_bytes(_bytes, strict=strict)
        audio_chunk = AudioChunk.from_audio(_audio)

        converted_dict["audio"] = audio_chunk
        converted_dict["random_seed"] = seed
        return cls(**converted_dict)


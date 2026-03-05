import io
from typing import Any

from mistral_common.audio import Audio
from mistral_common.imports import assert_soundfile_installed, is_soundfile_installed
from mistral_common.protocol.base import BaseCompletionRequest

if is_soundfile_installed():
    import soundfile as sf


class SpeechRequest(BaseCompletionRequest):
    r"""Request for text-to-speech synthesis.

    Supports both preset voices and voice cloning via reference audio.

    Attributes:
        id: Optional unique identifier for the speech request.
        model: Optional model identifier for the speech synthesis.
        input: Text input to be converted to speech.
        voice: Optional preset voice identifier (e.g., 'Neutral Male', 'Neutral Female') to use for speech synthesis.
        ref_audio: Optional reference audio for voice cloning, provided as a base64-encoded string or raw bytes.
            Takes precedence over voice when both are provided.
    """

    id: str | None = None
    model: str | None = None
    input: str
    voice: str | None = None
    ref_audio: str | bytes | None = None

    def to_openai(self, **kwargs: Any) -> dict[str, Any]:
        r"""Convert this SpeechRequest to an OpenAI-compatible request dictionary.

        Args:
            **kwargs: Additional key-value pairs to include in the request dictionary.

        Returns:
            An OpenAI-compatible request dictionary.
        """
        openai_request: dict[str, Any] = self.model_dump(exclude={"ref_audio"})

        assert_soundfile_installed()

        if self.ref_audio is not None:
            if isinstance(self.ref_audio, bytes):
                buffer = io.BytesIO(self.ref_audio)
            else:
                audio = Audio.from_base64(self.ref_audio)

                buffer = io.BytesIO()
                sf.write(buffer, audio.audio_array, audio.sampling_rate, format=audio.format)
                buffer.seek(0)

            openai_request["ref_audio"] = buffer

        openai_request.update(kwargs)

        return openai_request

    @classmethod
    def from_openai(cls, openai_request: dict[str, Any], strict: bool = False) -> "SpeechRequest":
        r"""Create a SpeechRequest instance from an OpenAI-compatible request dictionary.

        Args:
            openai_request: The OpenAI request dictionary.
            strict: A flag indicating whether to perform strict validation of the audio data.

        Returns:
            An instance of SpeechRequest.
        """
        converted_dict: dict[str, Any] = {k: v for k, v in openai_request.items() if k in cls.model_fields}

        if (ref_audio := openai_request.get("ref_audio")) is not None:
            if isinstance(ref_audio, io.BytesIO):
                audio_bytes = ref_audio.getvalue()
            elif hasattr(ref_audio, "file"):
                audio_bytes = ref_audio.file.read()
            else:
                # Already a string (base64) or bytes
                audio_bytes = ref_audio

            if isinstance(audio_bytes, bytes):
                audio = Audio.from_bytes(audio_bytes, strict=strict)
                assert audio.format is not None, f"Audio format must be set, got {audio.format=}"
                converted_dict["ref_audio"] = audio.to_base64(audio.format)
            else:
                converted_dict["ref_audio"] = audio_bytes

        # OAI uses "voice" as a string or object with "id"; normalize to string
        voice = openai_request.get("voice")
        if isinstance(voice, dict):
            converted_dict["voice"] = voice["id"]

        return cls(**converted_dict)

import io
from typing import Any

from mistral_common.imports import assert_soundfile_installed, is_soundfile_installed
from mistral_common.protocol.base import BaseCompletionRequest
from mistral_common.protocol.instruct.chunk import _detect_audio_format
from mistral_common.tokens.tokenizers.audio import Audio

if is_soundfile_installed():
    import soundfile as sf


class SpeechRequest(BaseCompletionRequest):
    r"""Request for text-to-speech synthesis.

    Supports both preset voices and voice cloning via reference audio.

    Attributes:
        id: Optional unique identifier for the speech request.
        model: Optional model identifier for the speech synthesis. If `None`, the
            serving side default speech model is used.
        input: Text to convert to speech.
        voice: Optional preset voice identifier (e.g., 'Neutral Male', 'Neutral
            Female'). Ignored when `ref_audio` is provided.
        ref_audio: Optional reference audio for voice cloning, provided as a
            base64-encoded string or raw bytes. Takes precedence over voice when
            both are provided.
    """

    id: str | None = None
    model: str | None = None
    input: str
    voice: str | None = None
    ref_audio: str | bytes | None = None

    def to_openai(self, **kwargs: Any) -> dict[str, Any]:
        r"""Convert this SpeechRequest to an OpenAI-compatible request dictionary.

        Reference audio is converted into an in-memory file buffer with the
        correct format extension, and `random_seed` is renamed to "seed".

        Args:
            **kwargs: Additional key-value pairs merged into the output.

        Returns:
            An OpenAI-compatible request dictionary.

        Raises:
            ImportError: If soundfile is not installed and `ref_audio` is provided.
        """
        openai_request: dict[str, Any] = self.model_dump(exclude={"ref_audio"})

        assert_soundfile_installed()

        if self.ref_audio is not None:
            if isinstance(self.ref_audio, bytes):
                buffer = io.BytesIO(self.ref_audio)
                fmt = _detect_audio_format(self.ref_audio)
            else:
                audio = Audio.from_base64(self.ref_audio)
                fmt = audio.format.lower()

                buffer = io.BytesIO()
                sf.write(buffer, audio.audio_array, audio.sampling_rate, format=audio.format)
                buffer.seek(0)

            # OpenAI's client uses the filename extension from .name to set the Content-Type.
            buffer.name = f"audio.{fmt}"
            openai_request["ref_audio"] = buffer

        openai_request["seed"] = openai_request.pop("random_seed")
        openai_request.update(kwargs)

        return openai_request

    @classmethod
    def from_openai(cls, openai_request: dict[str, Any], strict: bool = False) -> "SpeechRequest":
        r"""Create a SpeechRequest from an OpenAI-compatible request dictionary.

        Reference audio can be a BytesIO, a file-like object with a .file
        attribute, raw bytes (decoded via Audio), or an already base64-encoded
        string (used as-is). A dict voice ({"id": ...}) is normalized to a string.

        Args:
            openai_request: Dictionary matching OpenAI's speech request schema.
            strict: If `True`, reference audio bytes are strictly validated during
                decoding.

        Returns:
            A SpeechRequest instance with "seed" mapped to `random_seed`.

        Raises:
            AssertionError: If decoded reference audio has no detectable format.
        """
        seed = openai_request.get("seed")
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

        converted_dict["random_seed"] = seed

        return cls(**converted_dict)

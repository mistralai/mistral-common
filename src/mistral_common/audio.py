import base64
import io
import logging
from enum import Enum
from typing import Type

import numpy as np

logger = logging.getLogger(__name__)
_soundfile_installed: bool

try:
    import soundfile  # noqa: F401

    _soundfile_installed = True
except ImportError:
    _soundfile_installed = False

try:
    import soxr  # noqa: F401

    _soxr_installed = True
except ImportError:
    _soxr_installed = False


def is_soundfile_installed() -> bool:
    return _soundfile_installed


def is_soxr_installed() -> bool:
    return _soxr_installed


AudioFormat: Type[Enum]
if is_soundfile_installed():
    import soundfile as sf

    # Get the available formats from soundfile
    available_formats = sf.available_formats()

    # Create an Enum dynamically
    AudioFormat = Enum("AudioFormat", {format_name: format_name for format_name in available_formats})
else:
    AudioFormat = Enum("AudioFormat", {"NONE": "NONE"})


class Audio:
    def __init__(self, audio_array: np.ndarray, sampling_rate: int) -> None:
        self.audio_array = audio_array
        self.sampling_rate = sampling_rate
        self._check_valid()

    def __repr__(self) -> str:
        return (
            f"Audio - sampling_rate={self.sampling_rate} Hz, "
            f"duration={len(self.audio_array) / self.sampling_rate:.2f}s, "
            f"shape={self.audio_array.shape}"
        )

    def _check_valid(self) -> None:
        assert isinstance(self.audio_array, np.ndarray), type(np.ndarray)
        assert self.audio_array.ndim == 1, f"{self.audio_array.ndim=}"

    @property
    def duration(self) -> float:
        # in seconds
        return self.audio_array.shape[0] / self.sampling_rate

    @staticmethod
    def from_base64(audio_base64: str) -> "Audio":
        if not is_soundfile_installed():
            raise ImportError(
                "soundfile is required for this function. Install it with 'pip install mistral-common[soundfile]'"
            )

        audio_bytes = base64.b64decode(audio_base64)

        # Read the bytes into an audio file.
        with io.BytesIO(audio_bytes) as audio_file:
            audio_array, sampling_rate = sf.read(audio_file)

        return Audio(audio_array=audio_array, sampling_rate=sampling_rate)

    def to_base64(self, format: AudioFormat) -> str:
        if not is_soundfile_installed():
            raise ImportError(
                "soundfile is required for this function. Install it with 'pip install mistral-common[soundfile]'"
            )

        with io.BytesIO() as audio_file:
            sf.write(audio_file, self.audio_array, self.sampling_rate, format=format.value)
            audio_file.seek(0)
            return base64.b64encode(audio_file.read()).decode("utf-8")

    def resample(self, new_sampling_rate: int) -> None:
        """Resample audio data to a new sampling rate."""
        if self.sampling_rate == new_sampling_rate:
            return

        if not is_soxr_installed():
            raise ImportError("soxr is required for this function. Install it with 'pip install mistral-common[soxr]'")

        self.audio_array = soxr.resample(self.audio_array, self.sampling_rate, new_sampling_rate, quality="HQ")
        self.sampling_rate = new_sampling_rate

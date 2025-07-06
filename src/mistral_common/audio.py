import base64
import io
import logging
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Type, Optional

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
    def __init__(self, audio_array: np.ndarray, sampling_rate: int, format: AudioFormat) -> None:
        self.audio_array = audio_array
        self.sampling_rate = sampling_rate
        self.format = format
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
        return Audio._from_bytes(audio_bytes)

    @staticmethod
    def from_file(file: str) -> "Audio":
        assert Path(file).exists(), f"{file=} does not exist"

        with open(file, "rb") as f:
            audio_bytes = f.read()

        return Audio._from_bytes(audio_bytes)

    @staticmethod
    def _from_bytes(audio_bytes: bytes) -> "Audio":
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


def hertz_to_mel(freq: float | np.ndarray) -> float | np.ndarray:
    """
    Convert frequency from hertz to mels using the "slaney" mel-scale.
    Args:
        freq (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in hertz (Hz).
    Returns:
        `float` or `np.ndarray`: The frequencies on the mel scale.
    """
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0

    if isinstance(freq, np.ndarray):
        assert isinstance(mels, np.ndarray)
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

    return mels


def mel_to_hertz(mels: np.ndarray) -> np.ndarray:
    """
    Convert frequency from mels to hertz using the "slaney" mel-scale.
    Args:
        mels (`np.ndarray`):
            The frequency, or multiple frequencies, in mels.
    Returns:
        `float` or `np.ndarray`: The frequencies in hertz.
    """
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0

    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    """
    Creates a triangular filter bank.
    Adapted from *torchaudio* and *librosa*.
    Args:
        fft_freqs (`np.ndarray` of shape `(num_frequency_bins,)`):
            Discrete frequencies of the FFT bins in Hz.
        filter_freqs (`np.ndarray` of shape `(num_mel_filters,)`):
            Center frequencies of the triangular filters to create, in Hz.
    Returns:
        `np.ndarray` of shape `(num_frequency_bins, num_mel_filters)`
    """
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


@cache
def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_bins: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    if num_frequency_bins < 2:
        raise ValueError(f"Require num_frequency_bins: {num_frequency_bins} >= 2")

    if min_frequency > max_frequency:
        raise ValueError(f"Require min_frequency: {min_frequency} <= max_frequency: {max_frequency}")

    # center points of the triangular mel filters
    mel_min = hertz_to_mel(min_frequency)
    mel_max = hertz_to_mel(max_frequency)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_bins + 2)
    filter_freqs = mel_to_hertz(mel_freqs)

    # frequencies of FFT bins in Hz
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (filter_freqs[2 : num_mel_bins + 2] - filter_freqs[:num_mel_bins])
    mel_filters *= np.expand_dims(enorm, 0)

    if (mel_filters.max(axis=0) == 0.0).any():
        raise ValueError(
            "At least one mel filter has all zero values. "
            f"The value for `num_mel_filters` ({num_mel_bins}) "
            "may be set too high. "
            "Or, the value for `num_frequency_bins` "
            f"({num_frequency_bins}) may be set too low."
        )
    return mel_filters

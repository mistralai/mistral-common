import base64
import io
import logging
import re
from enum import Enum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import requests

from mistral_common.imports import (
    assert_soundfile_installed,
    assert_soxr_installed,
    is_soundfile_installed,
    is_soxr_installed,
)

if TYPE_CHECKING:
    from mistral_common.protocol.instruct.chunk import RawAudio

logger = logging.getLogger(__name__)

if is_soundfile_installed():
    import soundfile as sf

    # Get the available formats from soundfile
    available_formats = sf.available_formats()

    # Create an Enum dynamically
    AudioFormat = Enum("AudioFormat", {format_name: format_name for format_name in available_formats})  # type: ignore[misc]
else:
    AudioFormat = Enum("AudioFormat", {"none": "none"})  # type: ignore[no-redef]

if is_soxr_installed():
    import soxr

EXPECTED_FORMAT_VALUES = [v.value.lower() for v in AudioFormat.__members__.values()]


class Audio:
    def __init__(self, audio_array: np.ndarray, sampling_rate: int, format: str) -> None:
        r"""Initialize an Audio instance with audio data, sampling rate, and format.

        Args:
            audio_array: The audio data as a numpy array.
            sampling_rate: The sampling rate of the audio in Hz.
            format: The format of the audio file.
        """
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
        assert_soundfile_installed()
        assert self.format in EXPECTED_FORMAT_VALUES, f"{self.format=} not in {EXPECTED_FORMAT_VALUES=}"

    @property
    def duration(self) -> float:
        r"""Calculate the duration of the audio in seconds.

        Returns:
           The duration of the audio in seconds.
        """
        # in seconds
        duration: float = self.audio_array.shape[0] / self.sampling_rate
        return duration

    @staticmethod
    def from_url(url: str, strict: bool = True) -> "Audio":
        r"""Create an Audio instance from a URL.

        Args:
            url: The URL of the audio file.
            strict: Whether to strictly enforce mono audio.

        Returns:
            An instance of the Audio class.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return Audio.from_bytes(response.content, strict=strict)
        except requests.RequestException as e:  # Something went wrong with the request.
            raise ValueError(f"Failed to download audio from URL: {url}") from e
        except Exception as e:  # Something went wrong with the audio file.
            raise ValueError(f"Failed to create Audio instance from URL: {url} .") from e

    @staticmethod
    def from_base64(audio_base64: str, strict: bool = True) -> "Audio":
        r"""Create an Audio instance from a base64 encoded string.

        Args:
            audio_base64: The base64 encoded audio data.
            strict: Whether to strictly enforce mono audio. Defaults to True.

        Returns:
            An instance of the Audio class.
        """
        assert_soundfile_installed()

        if re.match(r"^data:audio/\w+;base64,", audio_base64):  # Remove the prefix if it exists
            audio_base64 = audio_base64.split(",")[1]

        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError("base64 decoding failed. Please check the input string is a valid base64.") from e

        return Audio.from_bytes(audio_bytes, strict=strict)

    @staticmethod
    def from_file(file: str, strict: bool = True) -> "Audio":
        r"""Create an Audio instance from an audio file.

        Args:
            file: Path to the audio file.
            strict: Whether to strictly enforce mono audio. Defaults to True.

        Returns:
            An instance of the Audio class.
        """
        assert_soundfile_installed()

        if isinstance(file, str) and file.startswith("file://"):
            file = file[7:]

        if not Path(file).exists():
            raise FileNotFoundError(f"{file=} does not exist")

        with open(file, "rb") as f:
            audio_bytes = f.read()

        return Audio.from_bytes(audio_bytes, strict=strict)

    @staticmethod
    def from_bytes(audio_bytes: bytes, strict: bool = True) -> "Audio":
        r"""Create an Audio instance from bytes.

        Args:
            audio_bytes: The audio data as bytes.
            strict: Whether to strictly enforce mono audio. Defaults to True.

        Returns:
            An instance of the Audio class.
        """
        # Read the bytes into an audio file.
        with io.BytesIO(audio_bytes) as audio_file:
            with sf.SoundFile(audio_file) as f:
                # Read the entire audio data
                audio_array = f.read(dtype="float32")
                sampling_rate = f.samplerate
                audio_format = f.format

        format_enum = AudioFormat(audio_format)
        format = format_enum.value.lower()

        if audio_array.ndim != 1:
            if strict:
                raise ValueError(f"{audio_array.ndim=}")
            else:
                audio_array = audio_array.mean(axis=1)

        return Audio(audio_array=audio_array, sampling_rate=sampling_rate, format=format)

    def to_base64(self, format: str, prefix: bool = False) -> str:
        r"""Convert the audio data to a base64 encoded string.

        Args:
            format: The format to encode the audio in.
            prefix: Whether to add a data prefix to the base64 encoded string.

        Returns:
            The base64 encoded audio data.
        """
        assert_soundfile_installed()

        assert format in EXPECTED_FORMAT_VALUES, f"{format=} not in {EXPECTED_FORMAT_VALUES=}"

        with io.BytesIO() as audio_file:
            sf.write(audio_file, self.audio_array, self.sampling_rate, format=format.upper())
            audio_file.seek(0)
            base64_str = base64.b64encode(audio_file.read()).decode("utf-8")
        if prefix:
            base64_str = f"data:audio/{format.lower()};base64,{base64_str}"
        return base64_str

    @staticmethod
    def from_raw_audio(audio: "RawAudio") -> "Audio":
        r"""Create an Audio instance from a RawAudio object.

        Args:
            audio: The RawAudio object containing audio data.

        Returns:
            An instance of the Audio class.
        """
        if isinstance(audio.data, bytes):
            return Audio.from_bytes(audio.data)
        elif isinstance(audio.data, str):
            return Audio.from_base64(audio.data)
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio.data)}")

    def resample(self, new_sampling_rate: int) -> None:
        r"""Resample audio data to a new sampling rate.

        Args:
            new_sampling_rate: The new sampling rate to resample the audio to.
        """
        if self.sampling_rate == new_sampling_rate:
            return

        assert_soxr_installed()

        self.audio_array = soxr.resample(self.audio_array, self.sampling_rate, new_sampling_rate, quality="HQ")
        self.sampling_rate = new_sampling_rate


def hertz_to_mel(freq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""Convert frequency from hertz to mels using the "slaney" mel-scale.

    Args:
        freq: The frequency, or multiple frequencies, in hertz (Hz).

    Returns:
        The frequencies on the mel scale.
    """
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0

    if isinstance(freq, np.ndarray):
        assert isinstance(mels, np.ndarray), type(mels)
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

    return mels


def mel_to_hertz(mels: np.ndarray) -> np.ndarray:
    r"""Convert frequency from mels to hertz using the "slaney" mel-scale.

    Args:
        mels: The frequency, or multiple frequencies, in mels.

    Returns:
        The frequencies in hertz.
    """
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0

    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    r"""Creates a triangular filter bank.

    Adapted from *torchaudio* and *librosa*.

    Args:
        fft_freqs: Discrete frequencies of the FFT bins in Hz.
        filter_freqs: Center frequencies of the triangular filters to create, in Hz.

    Returns:
        array of shape `(num_frequency_bins, num_mel_filters)`
    """
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    filter_bank: np.ndarray = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
    return filter_bank


@cache
def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_bins: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    r"""Create a Mel filter bank matrix for converting frequency bins to the Mel scale.

    This function generates a filter bank matrix that can be used to transform a
    spectrum represented in frequency bins to the Mel scale. The Mel scale is a
    perceptual scale of pitches judged by listeners to be equal in distance from one another.

    Args:
        num_frequency_bins: The number of frequency bins in the input spectrum.
        num_mel_bins: The number of desired Mel bins in the output.
        min_frequency: The minimum frequency (in Hz) to consider.
        max_frequency: The maximum frequency (in Hz) to consider.
        sampling_rate: The sampling rate of the audio signal.

    Returns:
        A filter bank matrix of shape (num_mel_bins, num_frequency_bins)
        that can be used to project frequency bin energies onto Mel bins.
    """
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

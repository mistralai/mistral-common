import logging
from functools import cache

import numpy as np

from mistral_common.deprecation import deprecated_import

logger = logging.getLogger(__name__)


def hertz_to_mel(freq: float | np.ndarray) -> float | np.ndarray:
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


def __getattr__(name: str) -> object:
    if name in ("Audio", "AudioFormat", "EXPECTED_FORMAT_VALUES"):
        return deprecated_import(
            old_path="mistral_common.audio",
            new_module="mistral_common.tokens.tokenizers.audio",
            name=name,
            version="1.13.0",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

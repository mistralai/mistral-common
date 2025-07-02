import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioSpectrogramConfig:
    # Number of mel bins, typically 80 or 128
    num_mel_bins: int
    # Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients, typically 160
    hop_length: int
    # Window size of the Fourier transform, typically 400
    window_size: int

    def __post_init__(self) -> None:
        assert self.num_mel_bins > 0
        assert self.hop_length > 0
        assert self.window_size > 0


@dataclass
class AudioConfig:
    sampling_rate: int
    # number of frames per second accepted by the tokenizer model.
    frame_rate: float
    audio_encoding_config: AudioSpectrogramConfig
    # Whether to pad an audio into multiples of chunk_length_s seconds
    chunk_length_s: Optional[float] = None

    def __post_init__(self) -> None:
        assert self.frame_rate > 0
        assert self.sampling_rate > 0

        if self.chunk_length_s is not None:
            assert self.chunk_length_s > 0
            assert self.chunk_frames > 0, (
                f"chunk_length_s and sampling_rate must both be > 0, got {self.chunk_length_s} and {self.sampling_rate}"
            )

    @property
    def chunk_frames(self) -> int:
        assert self.chunk_length_s is not None, f"Can't call chunk_frames if {self.chunk_length_s=}."
        return int(self.chunk_length_s * self.sampling_rate)

    @property
    def audio_length_per_tok(self) -> int:
        downsample_factor = self.sampling_rate // self.frame_rate
        downsample_factor /= self.audio_encoding_config.hop_length
        return int(downsample_factor)

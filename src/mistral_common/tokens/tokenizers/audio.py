import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from mistral_common.audio import Audio
from mistral_common.protocol.instruct.messages import AudioChunk

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


@dataclass
class AudioEncoding:
    # Text tokens corresponding to this audio chunk
    tokens: List[int]
    # Original audio waveform data.
    audio: Audio


@dataclass
class SpecialAudioIDs:
    """Special text tokens corresponding to audio token sequence."""

    audio: int
    begin_audio: int


class AudioEncoder:
    def __init__(self, audio_config: AudioConfig, special_ids: SpecialAudioIDs) -> None:
        self.audio_config = audio_config
        self.encoding_config = audio_config.audio_encoding_config
        self.special_ids = special_ids

    def _pad(self, audio_array: np.ndarray) -> np.ndarray:
        if self.audio_config.chunk_length_s:
            # pad the audio to a multiple of chunk_length_s seconds
            # padding token is zero (equivalent to silence in the audio space)
            next_multiple_of_chunk_frames = (
                math.ceil(audio_array.shape[-1] / self.audio_config.chunk_frames) * self.audio_config.chunk_frames
            )
            audio_array = np.pad(
                audio_array, (0, next_multiple_of_chunk_frames - audio_array.shape[-1])
            )
        elif audio_array.shape[-1] < self.encoding_config.window_size:
            # minimum length for audios is at least one spectrogram frame
            audio_array = np.pad(
                audio_array, (0, self.encoding_config.window_size - audio_array.shape[-1])
            )

        return audio_array


    def _encode_audio_chunk(self, content: AudioChunk) -> AudioEncoding:
        audio = Audio.from_base64(content.input_audio.data)
        audio.resample(self.audio_config.sampling_rate)

        audio.audio_array = self._pad(audio.audio_array)
        signal_length = audio.audio_array.shape[0]

        # for spectrogram-based models, the waveform is downsampled by the hop_length when computing the log-mel
        if signal_length % self.encoding_config.hop_length != 0:
            signal_length = math.ceil(signal_length / self.encoding_config.hop_length - 1)
        else:
            signal_length = signal_length // self.encoding_config.hop_length

        num_audio_tokens = math.ceil(signal_length / self.audio_config.audio_length_per_tok)
        audio_tokens = [self.begin_audio_token] + [self.audio_token] * num_audio_tokens

        return AudioEncoding(
            tokens=audio_tokens,
            audio=audio,
        )

    def __call__(self, content: AudioChunk) -> AudioEncoding:
        return self._encode_audio_chunk(content)

    @property
    def audio_token(self) -> int:
        return self.special_ids.audio

    @property
    def begin_audio_token(self) -> int:
        return self.special_ids.begin_audio

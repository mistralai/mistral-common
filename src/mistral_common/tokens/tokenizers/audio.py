from enum import Enum
import logging
import math
from dataclasses import dataclass

import numpy as np

from mistral_common.audio import Audio
from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURLChunk, AudioURLType

logger = logging.getLogger(__name__)

# for offline streaming we're encoding the whole audio at once.
# Because the model is delayed by <transcription_delay_ms> + word_length
# We must always add a buffer of max world length to the audio in the end
# For now we assume that there is no word that requires more than 10 tokens (0.8s)
# which it might it practice but it won't affect any eval results
OFFLINE_STREAMING_BUFFER_TOKENS = 10


class TranscriptionFormat(str, Enum):
    """Transcription format
    Should be set by the tokenizer for correct encoding.
    """

    INSTRUCT = "instruct"
    STREAMING = "streaming"


@dataclass
class AudioSpectrogramConfig:
    r"""Configuration for generating an audio spectrogram.

    Attributes:
        num_mel_bins: Number of mel bins, typically 80 or 128.
        hop_length: Length of the overlapping windows for
            the STFT used to obtain the Mel Frequency coefficients, typically 160.
        window_size: Window size of the Fourier transform, typically 400.
    """

    # Number of mel bins, typically 80 or 128
    num_mel_bins: int
    # Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients, typically 160
    hop_length: int
    # Window size of the Fourier transform, typically 400
    window_size: int

    def __post_init__(self) -> None:
        assert self.num_mel_bins > 0, self.num_mel_bins
        assert self.hop_length > 0, self.hop_length
        assert self.window_size > 0, self.window_size


@dataclass
class AudioConfig:
    r"""Configuration for audio processing.

    Attributes:
        sampling_rate: Sampling rate of the audio.
        frame_rate: Number of frames per second accepted by the tokenizer model.
        encoding_config: Configuration for audio spectrogram.
        chunk_length_s: Whether to pad an audio into multiples of chunk_length_s seconds (optional).
    """

    sampling_rate: int
    # number of frames per second accepted by the tokenizer model.
    frame_rate: float
    encoding_config: AudioSpectrogramConfig
    # Whether to pad an audio into multiples of chunk_length_s seconds
    chunk_length_s: float | None = None

    # delay between the audio stream and text stream
    transcription_delay_ms: float | None = None

    # If we're in streaming or non-streaming
    transcription_format: TranscriptionFormat | None = TranscriptionFormat.INSTRUCT

    def __post_init__(self) -> None:
        assert self.frame_rate > 0, self.frame_rate
        assert self.sampling_rate > 0, self.sampling_rate

        if self.chunk_length_s is not None:
            assert self.chunk_length_s > 0, self.chunk_length_s
            assert self.chunk_frames > 0, (
                f"chunk_length_s and sampling_rate must both be > 0, got {self.chunk_length_s} and {self.sampling_rate}"
            )

        assert self.is_streaming == (self.transcription_delay_ms is not None), (
            f"{self.is_streaming=} and {self.transcription_delay_ms=} must be both set or both unset"
        )

        if self.transcription_delay_ms is not None:
            frame_duration_ms = 1000.0 / self.frame_rate

            assert self.transcription_delay_ms > 0, "{self.transcription_delay_ms=} must be > 0"
            assert self.transcription_delay_ms % frame_duration_ms == 0, (
                f"{self.transcription_delay_ms=} must be a multiple of {frame_duration_ms=}"
            )
            assert self.chunk_length_s is None, f"{self.chunk_length_s=} cannot be set in streaming."

    @property
    def is_streaming(self) -> bool:
        return self.transcription_format == TranscriptionFormat.STREAMING

    def num_audio_tokens(self, audio_len: int) -> int:
        if audio_len % self.encoding_config.hop_length != 0:
            audio_len = math.ceil(audio_len / self.encoding_config.hop_length - 1)
        else:
            audio_len = audio_len // self.encoding_config.hop_length

        return math.ceil(audio_len / self.audio_length_per_tok)

    @property
    def num_delay_tokens(self) -> int:
        assert self.is_streaming, f"Can't call num_delay_tokens if {self.is_streaming=}."
        # streaming pad tokens
        assert self.transcription_delay_ms is not None
        delay_len = int(self.transcription_delay_ms / 1000.0 * self.sampling_rate)

        return self.num_audio_tokens(delay_len)

    @property
    def chunk_frames(self) -> int:
        r"""Calculate the number of frames per chunk."""
        assert self.chunk_length_s is not None, f"Can't call chunk_frames if {self.chunk_length_s=}."
        return int(self.chunk_length_s * self.sampling_rate)

    @property
    def audio_length_per_tok(self) -> int:
        r"""Calculate the length of audio per token."""
        downsample_factor = self.sampling_rate // self.frame_rate
        downsample_factor /= self.encoding_config.hop_length
        return int(downsample_factor)


@dataclass
class AudioEncoding:
    r"""Encapsulates the tokens and audio data for an audio chunk.

    Attributes:
        tokens: Text tokens corresponding to this audio chunk.
        audio: Original audio waveform data.
    """

    # Text tokens corresponding to this audio chunk
    tokens: list[int]
    # Original audio waveform data.
    audio: Audio


@dataclass
class SpecialAudioIDs:
    r"""Special text tokens corresponding to audio token sequence.

    Attributes:
        audio: Token representing audio.
        begin_audio: Token representing the beginning of audio.
    """

    audio: int
    begin_audio: int


class AudioEncoder:
    r"""Encodes audio chunks into a format suitable for further processing.

    Attributes:
        audio_config: Configuration for audio processing.
        encoding_config: Configuration for audio spectrogram.
        special_ids: Special tokens for audio encoding.
    """

    def __init__(self, audio_config: AudioConfig, special_ids: SpecialAudioIDs) -> None:
        self.audio_config = audio_config
        self.encoding_config = audio_config.encoding_config
        self.special_ids = special_ids

    def pad(self, audio_array: np.ndarray, sampling_rate: int, is_online_streaming: bool) -> np.ndarray:
        r"""Pad the audio array to the desired length.

        Args:
            audio_array: Audio data as a numpy array.
            sampling_rate: Sampling rate of the audio.
            is_online_streaming: Whether the audio is being streamed online.

        Returns:
            Padded audio array.
        """
        if self.audio_config.chunk_length_s:
            next_multiple_of_chunk_frames = self.next_multiple_of_chunk_frames(audio_array.shape[-1], sampling_rate)
            audio_array = np.pad(audio_array, (0, next_multiple_of_chunk_frames - audio_array.shape[-1]))
        elif (
            isinstance(self.encoding_config, AudioSpectrogramConfig)
            and audio_array.shape[-1] < self.encoding_config.window_size
        ):
            # minimum length for audios is at least one spectrogram frame
            audio_array = np.pad(audio_array, (0, self.encoding_config.window_size - audio_array.shape[-1]))
        elif self.audio_config.is_streaming:
            pad = self._get_streaming_pad(audio_array.shape[-1], is_online_streaming)
            audio_array = np.pad(audio_array, (0, pad))

        return audio_array

    def _get_streaming_pad(self, num_samples: int, is_online: bool) -> int:
        # let's make sure the audio is a multiple of one "frame" token
        mult_of = self.audio_config.audio_length_per_tok
        pad = int((mult_of - (num_samples % mult_of)) % mult_of)

        if not is_online:
            # in offline streaming we're appending an extra pad to simulate
            # a whole streaming session

            #  then add delay tokens + BOS token + buffer approx
            _extra_pad_tokens = (self.audio_config.num_delay_tokens + 1) + OFFLINE_STREAMING_BUFFER_TOKENS
            extra_pad_samples = int(mult_of * _extra_pad_tokens)
            assert extra_pad_samples % mult_of == 0, f"{extra_pad_samples=} must be a multiple of {mult_of=}"
            pad += extra_pad_samples

        return pad

    def next_multiple_of_chunk_frames(self, audio_array_len: int, sampling_rate: int) -> int:
        r"""Calculate the next multiple of chunk frames.

        Args:
            audio_array_len: Length of the audio array.
            sampling_rate: Sampling rate of the audio.

        Returns:
            The next multiple of chunk frames.
        """
        assert sampling_rate == self.audio_config.sampling_rate, (
            f"Expected {sampling_rate=} to be {self.audio_config.sampling_rate=}"
        )
        assert self.audio_config.chunk_length_s is not None, (
            f"Can't call next_multiple_of_chunk_frames if {self.audio_config.chunk_length_s=}."
        )

        return math.ceil(audio_array_len / self.audio_config.chunk_frames) * self.audio_config.chunk_frames

    def encode_audio(self, audio: Audio, is_online_streaming: bool) -> AudioEncoding:
        audio.resample(self.audio_config.sampling_rate)

        audio.audio_array = self.pad(audio.audio_array, self.audio_config.sampling_rate, is_online_streaming)
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

    def _encode_audio_chunk(self, content: AudioChunk) -> AudioEncoding:
        audio = Audio.from_raw_audio(content.input_audio)
        return self.encode_audio(audio, False)

    def _encode_audio_url_chunk(self, content: AudioURLChunk) -> AudioEncoding:
        url_type = content.get_url_type()

        if url_type in {AudioURLType.file, AudioURLType.file_uri}:
            audio = Audio.from_file(content.url)
        elif url_type == AudioURLType.url:
            audio = Audio.from_url(content.url)
        else:
            audio = Audio.from_base64(content.url)

        return self._encode_audio(audio)

    def __call__(self, content: AudioChunk | AudioURLChunk) -> AudioEncoding:
        r"""Call the encoder on an audio chunk or URL chunk.

        Args:
            content: Audio or URL chunk to encode.

        Returns:
            Encoded audio data and tokens.
        """
        if isinstance(content, AudioURLChunk):
            return self._encode_audio_url_chunk(content)
        elif isinstance(content, AudioChunk):
            return self._encode_audio_chunk(content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    @property
    def audio_token(self) -> int:
        r"""Get the audio token."""
        return self.special_ids.audio

    @property
    def begin_audio_token(self) -> int:
        r"""Get the begin audio token."""
        return self.special_ids.begin_audio

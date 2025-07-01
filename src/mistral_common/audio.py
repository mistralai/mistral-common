from enum import Enum
import logging
import math
import io
import numpy as np
import base64
from typing import Tuple, Type

logger = logging.getLogger(__name__)
_soundfile_installed: bool

try:
    import soundfile

    _soundfile_installed = True
except ImportError:
    _soundfile_installed = False
except Exception as e:
    # cv2 has lots of import problems: https://github.com/opencv/opencv-python/issues/884
    # for better UX, let's simply skip all errors that might arise from import for now
    logger.warning(
        f"Warning: Your installation of OpenCV appears to be broken: {e}."
        "Please follow the instructions at https://github.com/opencv/opencv-python/issues/884 "
        "to correct your environment. The import of cv2 has been skipped."
    )

AudioFormat: Type[Enum]
if _soundfile_installed:
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
        audio_bytes = base64.b64decode(audio_base64)

        # Read the bytes into an audio file.
        with io.BytesIO(audio_bytes) as audio_file:
            audio_array, sampling_rate = sf.read(audio_file)

        return Audio(audio_array=audio_array, sampling_rate=sampling_rate)

    def to_base64(self, format: AudioFormat) -> str:
        assert _soundfile_installed, "soundfile has to be installed to use this function"
        with io.BytesIO() as audio_file:
            sf.write(audio_file, self.audio_array, self.sampling_rate, format=format.value)
            audio_file.seek(0)
            return base64.b64encode(audio_file.read()).decode("utf-8")

    @staticmethod
    def _get_resample_kernel(
        orig_freq: int,
        new_freq: int,
        gcd: int,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
    ) -> Tuple[np.ndarray, int]:
        """Creates kernel for resampling audio signal from one sampling rate to another

        Args:
            orig_freq (int): Original sampling rate
            new_freq (int): New sampling rate
            gcd (int): Greatest common divisor of orig_freq and new_freq
            device (torch.device): Device to create the kernel on
            lowpass_filter_width (int): Controls sharpness of the filter - higher means sharper
            rolloff (float): Roll-off frequency for the filter
        Returns:
            tuple[torch.Tensor, int]: The kernel and width of the kernel
        """
        orig_freq = int(orig_freq) // gcd
        new_freq = int(new_freq) // gcd

        if lowpass_filter_width <= 0:
            raise ValueError("Low pass filter width should be positive.")
        base_freq = min(orig_freq, new_freq)
        base_freq = max(1, int(base_freq * rolloff))
        width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
        idx = np.arange(-width, width + orig_freq, dtype=np.float64)[None] / orig_freq
        t = np.arange(0, -new_freq, -1, dtype=np.float64)[:, None] / new_freq + idx
        t *= base_freq
        t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)

        window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2

        t *= math.pi

        scale = base_freq / orig_freq
        eps = 1e-20
        kernels = np.where(t == 0, np.ones_like(t), np.sin(t) / (t + eps))
        kernels *= window * scale

        kernels = kernels.astype(dtype=np.float32)

        return kernels, width

    @staticmethod
    def _apply_resample_kernel(
        waveform: np.ndarray,
        orig_freq: int,
        new_freq: int,
        gcd: int,
        kernel: np.ndarray,
        width: int,
    ) -> np.ndarray:
        """Resamples audio signal from one sampling rate to another using a kernel

        Args:
            waveform (np.ndarray): Input waveform [batch, channels, length]
            orig_freq (int): Original sampling rate
            new_freq (int): New sampling rate
            gcd (int): Greatest common divisor of orig_freq and new_freq
            kernel (np.ndarray): Kernel for resampling
            width (int): Width of the kernel
        Returns:
            np.ndarray: Resampled waveform [batch, channels, length * new_freq // orig_freq]
        """
        orig_freq = int(orig_freq) // gcd
        new_freq = int(new_freq) // gcd

        # pack batch
        length = waveform.shape[0]
        padded_waveform = np.pad(waveform, (width, width + orig_freq))
        resampled: np.ndarray = np.stack(
            [np.convolve(padded_waveform, kernel[i][::-1], "valid") for i in range(kernel.shape[0])]
        )

        resampled = resampled[:, ::orig_freq]  # apply stride
        resampled = resampled.T.reshape(1, -1)
        target_length = np.ceil(np.array(new_freq * length / orig_freq)).astype(np.int64)
        resampled = np.reshape(resampled[..., :target_length], [-1])

        return resampled

    def resample(self, new_sampling_rate: int) -> None:
        """Resample audio data to a new sampling rate."""
        if self.sampling_rate == new_sampling_rate:
            return

        gcd = math.gcd(int(self.sampling_rate), int(new_sampling_rate))
        kernel, width = self._get_resample_kernel(self.sampling_rate, new_sampling_rate, gcd)
        self.audio_array = self._apply_resample_kernel(
            self.audio_array,
            self.sampling_rate,
            new_sampling_rate,
            gcd,
            kernel,
            width,
        )
        self.sampling_rate = new_sampling_rate

    @staticmethod
    def _get_resample_kernel(
        orig_freq: int,
        new_freq: int,
        gcd: int,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
    ) -> Tuple[np.ndarray, int]:
        """Creates kernel for resampling audio signal from one sampling rate to another

        Args:
            orig_freq (int): Original sampling rate
            new_freq (int): New sampling rate
            gcd (int): Greatest common divisor of orig_freq and new_freq
            device (torch.device): Device to create the kernel on
            lowpass_filter_width (int): Controls sharpness of the filter - higher means sharper
            rolloff (float): Roll-off frequency for the filter
        Returns:
            tuple[torch.Tensor, int]: The kernel and width of the kernel
        """
        orig_freq = int(orig_freq) // gcd
        new_freq = int(new_freq) // gcd

        if lowpass_filter_width <= 0:
            raise ValueError("Low pass filter width should be positive.")
        base_freq = min(orig_freq, new_freq)
        base_freq = max(1, int(base_freq * rolloff))
        width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
        idx = np.arange(-width, width + orig_freq, dtype=np.float64)[None] / orig_freq
        t = np.arange(0, -new_freq, -1, dtype=np.float64)[:, None] / new_freq + idx
        t *= base_freq
        t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)

        window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2

        t *= math.pi

        scale = base_freq / orig_freq
        eps = 1e-20
        kernels = np.where(t == 0, np.ones_like(t), np.sin(t) / (t + eps))
        kernels *= window * scale

        # downcast: problematic?
        kernels = kernels.astype(dtype=np.float32)

        return kernels, width

    @staticmethod
    def _apply_resample_kernel(
        waveform: np.ndarray,
        orig_freq: int,
        new_freq: int,
        gcd: int,
        kernel: np.ndarray,
        width: int,
    ) -> np.ndarray:
        """Resamples audio signal from one sampling rate to another using a kernel

        Args:
            waveform (np.ndarray): Input waveform [batch, channels, length]
            orig_freq (int): Original sampling rate
            new_freq (int): New sampling rate
            gcd (int): Greatest common divisor of orig_freq and new_freq
            kernel (np.ndarray): Kernel for resampling
            width (int): Width of the kernel
        Returns:
            np.ndarray: Resampled waveform [batch, channels, length * new_freq // orig_freq]
        """
        orig_freq = int(orig_freq) // gcd
        new_freq = int(new_freq) // gcd

        # pack batch
        length = waveform.shape[0]
        padded_waveform = np.pad(waveform, (width, width + orig_freq))
        resampled: np.ndarray = np.stack(
            [np.convolve(padded_waveform, kernel[i][::-1], "valid") for i in range(kernel.shape[0])]
        )

        resampled = resampled[:, ::orig_freq]  # apply stride
        resampled = resampled.T.reshape(1, -1)
        target_length = np.ceil(np.array(new_freq * length / orig_freq)).astype(np.int64)
        resampled = np.reshape(resampled[..., :target_length], [-1])

        return resampled

    def resample(self, new_sampling_rate: int) -> None:
        """Resample audio data to a new sampling rate."""
        if self.sampling_rate == new_sampling_rate:
            return

        gcd = math.gcd(int(self.sampling_rate), int(new_sampling_rate))
        kernel, width = Audio._get_resample_kernel(self.sampling_rate, new_sampling_rate, gcd)
        self.audio_array = Audio._apply_resample_kernel(
            self.audio_array,
            self.sampling_rate,
            new_sampling_rate,
            gcd,
            kernel,
            width,
        )
        self.sampling_rate = new_sampling_rate

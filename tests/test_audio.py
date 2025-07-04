import numpy as np

from mistral_common.audio import Audio, AudioFormat


def sin_wave(sampling_rate: int, duration: float) -> np.ndarray:
    return np.sin(np.ones([int(duration * sampling_rate)]))


def test_audio_resample() -> None:
    sampling_rate = 44_000
    original_array = sin_wave(sampling_rate, 1)

    audio = Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
    )

    audio.resample(sampling_rate)
    assert np.allclose(audio.audio_array, original_array, atol=1e-5)

    audio.resample(sampling_rate // 2)
    assert audio.sampling_rate == sampling_rate // 2
    assert len(audio.audio_array) == len(original_array) // 2
    # fmt: off
    expected_resampled_array = [0.622, 0.907, 0.802, 0.870, 0.819, 0.860, 0.826, 0.854, 0.831, 0.850]
    # fmt: on
    assert np.allclose(audio.audio_array[:10], expected_resampled_array, atol=1e-3)


def test_audio_base64() -> None:
    sampling_rate = 16_000
    original_array = sin_wave(sampling_rate, 3.3)

    audio = Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
    )

    for format in [AudioFormat.MP3, AudioFormat.WAV]:
        base64_str = audio.to_base64(format)
        new_audio = Audio.from_base64(base64_str)

    assert audio.sampling_rate == new_audio.sampling_rate
    assert np.allclose(audio.audio_array, new_audio.audio_array, atol=1e-5)

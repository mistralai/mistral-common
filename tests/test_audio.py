import tempfile

import numpy as np
import pytest
import soundfile as sf

from mistral_common.audio import Audio, hertz_to_mel, mel_filter_bank


def sin_wave(sampling_rate: int, duration: float) -> np.ndarray:
    return np.sin(np.ones([int(duration * sampling_rate)]))


def test_audio_resample() -> None:
    sampling_rate = 44_000
    original_array = sin_wave(sampling_rate, 1)

    audio = Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
        format="wav",
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


def test_from_file() -> None:
    sampling_rate = 44100
    original_array = sin_wave(sampling_rate, 1)

    # Test with a local path
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        with sf.SoundFile(tmp.name, "w", samplerate=sampling_rate, channels=1) as f:
            f.write(original_array)
        audio_local = Audio.from_file(tmp.name)
    assert isinstance(audio_local, Audio)
    assert np.allclose(audio_local.audio_array, original_array, atol=1e-5)
    assert audio_local.sampling_rate == sampling_rate


def test_from_url() -> None:
    # Test with an invalid URL
    with pytest.raises(ValueError, match=("Failed to download audio from URL: https://example.com/invalid_audio.wav")):
        Audio.from_url("https://example.com/invalid_audio.wav")

    # Test with an invalid content
    with pytest.raises(ValueError, match="Failed to create Audio instance from URL: https://example.com ."):
        Audio.from_url("https://example.com")

    # Test valid URL
    url = "https://download.samplelib.com/mp3/sample-3s.mp3"
    audio_url = Audio.from_url(url, strict=False)
    assert isinstance(audio_url, Audio)


@pytest.mark.parametrize("prefix", [True, False])
def test_audio_base64(prefix: bool) -> None:
    sampling_rate = 16_000
    original_array = sin_wave(sampling_rate, 3.3)

    LOSSY = ["mp3"]
    LOSSLESS = ["wav"]

    for format in LOSSY + LOSSLESS:
        audio = Audio(
            audio_array=original_array,
            sampling_rate=sampling_rate,
            format=format,
        )

        base64_str = audio.to_base64(format, prefix)
        new_audio = Audio.from_base64(base64_str)

        if prefix:
            assert base64_str.startswith(f"data:audio/{format};base64,")

        assert audio.sampling_rate == new_audio.sampling_rate
        if format in LOSSLESS:
            assert np.allclose(audio.audio_array, new_audio.audio_array, atol=1e-5)
        elif format in LOSSY:

            def rmse(a: np.ndarray, b: np.ndarray) -> float:
                return float(np.sqrt(np.mean((a - b) ** 2)))

            assert rmse(audio.audio_array, new_audio.audio_array) < 5e-3
        else:
            raise ValueError(f"Unknown format {format}")


@pytest.mark.parametrize(
    "freq, expected_mel",
    [
        (100.0, 1.5),  # Linear region
        (1000.0, 15.0),  # Boundary of linear and log regions
        (4000.0, 15.0 + np.log(4000.0 / 1000.0) * 27.0 / np.log(6.4)),  # Log region
        (10000.0, 15.0 + np.log(10000.0 / 1000.0) * 27.0 / np.log(6.4)),  # Log region
    ],
)
def test_hertz_to_mel_single_value(freq: float, expected_mel: float) -> None:
    mel = hertz_to_mel(freq)
    assert isinstance(mel, float)
    assert abs(mel - expected_mel) < 1e-6


def test_hertz_to_mel_array() -> None:
    # Test with an array of frequency values
    freq_array = np.array([100.0, 1000.0, 4000.0])
    expected_mel_array = np.array(
        [
            3.0 * 100.0 / 200.0,  # Linear region
            15.0,  # min_log_mel
            15.0 + np.log(4000.0 / 1000.0) * 27.0 / np.log(6.4),  # Log region
        ]
    )
    np.testing.assert_array_equal(hertz_to_mel(freq_array), expected_mel_array)


def test_mel_filter_bank() -> None:
    # Test mel_filter_bank function
    num_frequency_bins = 256
    num_mel_bins = 20
    min_frequency = 0
    max_frequency = 8000
    sampling_rate = 16000

    mel_filters = mel_filter_bank(num_frequency_bins, num_mel_bins, min_frequency, max_frequency, sampling_rate)

    # Check the shape of the output
    assert mel_filters.shape == (num_frequency_bins, num_mel_bins)

    # Check that all filters sum to 1 (approximately)
    assert np.allclose(mel_filters.sum(axis=0), 0.032, atol=0.005)

    # Check that there are no negative values
    assert (mel_filters >= 0).all()

    # integration test
    expected_array = np.array(
        [
            0.03160475,
            0.03200728,
            0.03184327,
            0.03176877,
            0.03208179,
            0.03160917,
            0.0320748,
            0.03180101,
            0.03186299,
            0.03190694,
            0.0319008,
            0.03182839,
            0.03191571,
            0.03184625,
            0.03188972,
            0.03185817,
            0.0318928,
            0.03186744,
            0.03187269,
            0.0318759,
        ]
    )
    diff = np.abs(mel_filters.sum(0) - expected_array)
    assert (diff < 1e-5).all()

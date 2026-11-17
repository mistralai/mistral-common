import numpy as np
import pytest

from mistral_common.audio import hertz_to_mel, mel_filter_bank, mel_to_hertz


@pytest.mark.parametrize(
    ("num_frequency_bins", "num_mel_bins", "min_frequency", "max_frequency", "match"),
    [
        (1, 128, 0.0, 8_000.0, r"Require num_frequency_bins: 1 >= 2"),
        (201, 128, 8_000.0, 0.0, r"Require min_frequency: 8000\.0 <= max_frequency: 0\.0"),
        # 1 + 64 // 2 frequency bins (64-sample window) are too few for 128 mel bins
        (33, 128, 0.0, 8_000.0, r"At least one mel filter has all zero values\..*\(128\).*\(33\)"),
    ],
    ids=["too_few_frequency_bins", "min_above_max", "all_zero_filter"],
)
def test_mel_filter_bank_rejects_invalid_arguments(
    num_frequency_bins: int, num_mel_bins: int, min_frequency: float, max_frequency: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        mel_filter_bank(num_frequency_bins, num_mel_bins, min_frequency, max_frequency, sampling_rate=16_000)


@pytest.mark.parametrize(
    ("freq", "expected_mel"),
    [
        (100.0, 1.5),  # linear region
        (1000.0, 15.0),  # boundary of the linear and log regions
        (4000.0, 15.0 + np.log(4000.0 / 1000.0) * 27.0 / np.log(6.4)),  # log region
        (10000.0, 15.0 + np.log(10000.0 / 1000.0) * 27.0 / np.log(6.4)),  # log region
    ],
    ids=["linear-100hz", "boundary-1000hz", "log-4000hz", "log-10000hz"],
)
def test_hertz_to_mel_single_value(freq: float, expected_mel: float) -> None:
    mel = hertz_to_mel(freq)
    assert isinstance(mel, float)
    assert abs(mel - expected_mel) < 1e-6


def test_hertz_to_mel_array() -> None:
    freq_array = np.array([100.0, 1000.0, 4000.0])
    expected_mel_array = np.array(
        [
            3.0 * 100.0 / 200.0,  # linear region
            15.0,  # min_log_mel
            15.0 + np.log(4000.0 / 1000.0) * 27.0 / np.log(6.4),  # log region
        ]
    )
    np.testing.assert_array_equal(hertz_to_mel(freq_array), expected_mel_array)


@pytest.mark.parametrize(
    "freq",
    [0.0, 100.0, 999.999, 1000.0, 4000.0, 10000.0],
    ids=["0hz", "linear-100hz", "just-below-boundary", "boundary-1000hz", "log-4000hz", "log-10000hz"],
)
def test_mel_scale_round_trip(freq: float) -> None:
    # hertz_to_mel and mel_to_hertz are exact inverses in both the linear and log regions.
    np.testing.assert_allclose(mel_to_hertz(np.array([hertz_to_mel(freq)])), [freq], rtol=1e-12, atol=1e-9)


def test_mel_filter_bank() -> None:
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

import pytest

from mistral_common.audio import mel_filter_bank


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

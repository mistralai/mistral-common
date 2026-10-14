r"""Test the comparison helpers shared by tests."""

from collections.abc import Callable

import numpy as np
import pytest

from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import Tokenized
from tests.utils import _TOKENIZED_FIELDS, assert_tokenized_equal

_TOKEN_VALUE_MESSAGE = r"tokens.*index 1.*9.*2"
_TOKEN_LENGTH_MESSAGE = r"tokens.*index 2.*<end>.*3"
_TOKEN_LENGTH_EXTRA_MESSAGE = r"tokens.*index 2.*3.*<end>"
_PREFIX_IDS_MESSAGE = r"prefix_ids.*index 0"
_PREFIX_IDS_VALUE_MESSAGE = r"prefix_ids.*index 1.*3.*4"
_PREFIX_IDS_LENGTH_MESSAGE = r"prefix_ids.*index 1.*<end>.*2"
_PREFIX_IDS_LENGTH_EXTRA_MESSAGE = r"prefix_ids.*index 1.*2.*<end>"
_IMAGE_COUNT_MESSAGE = r"images.*index 0"
_IMAGE_SHAPE_MESSAGE = r"images.*index 0.*shape"
_IMAGE_DTYPE_MESSAGE = r"images.*index 0.*dtype"
_IMAGE_VALUE_MESSAGE = r"images.*index 0.*values"
_IMAGE_SECOND_VALUE_MESSAGE = r"images.*index 1.*values"
_AUDIO_COUNT_MESSAGE = r"audios.*index 0"
_AUDIO_SAMPLING_RATE_MESSAGE = r"audios.*index 0.*sampling_rate"
_AUDIO_FORMAT_MESSAGE = r"audios.*index 0.*format"
_AUDIO_SHAPE_MESSAGE = r"audios.*index 0.*shape"
_AUDIO_DTYPE_MESSAGE = r"audios.*index 0.*dtype"
_AUDIO_VALUE_MESSAGE = r"audios.*index 0.*values"
_AUDIO_SECOND_VALUE_MESSAGE = r"audios.*index 1.*values"
_AUDIO_ATTRIBUTE_SETS_MESSAGE = r"audios.*index 0.*extra"


def _build_tokenized_pair(
    *,
    actual_tokens: list[int],
    expected_tokens: list[int],
    actual_prefix_ids: list[int] | None,
    expected_prefix_ids: list[int] | None,
    actual_images: list[np.ndarray],
    expected_images: list[np.ndarray],
    actual_audios: list[Audio],
    expected_audios: list[Audio],
) -> tuple[Tokenized, Tokenized]:
    r"""Build fresh actual and expected tokenized values for one comparison."""
    return (
        Tokenized(
            tokens=actual_tokens,
            prefix_ids=actual_prefix_ids,
            images=actual_images,
            audios=actual_audios,
        ),
        Tokenized(
            tokens=expected_tokens,
            prefix_ids=expected_prefix_ids,
            images=expected_images,
            audios=expected_audios,
        ),
    )


def _build_audio(*, dtype: type[np.float32] | type[np.float64], sampling_rate: int, format: str) -> Audio:
    r"""Build an eight-sample audio value with the requested metadata."""
    return Audio(
        audio_array=np.array(object=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=dtype),
        sampling_rate=sampling_rate,
        format=format,
    )


def _build_identical_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build identical values for every compared field."""
    return _build_tokenized_pair(
        actual_tokens=[1, 2],
        expected_tokens=[1, 2],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[],
        expected_audios=[],
    )


def _build_different_private_text_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build values whose private text differs but public fields match."""
    actual, expected = _build_identical_pair()
    actual._text = "actual text"
    return actual, expected


def _build_image_within_tolerance_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build image values differing by less than the default tolerance."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[np.full(shape=(2, 2, 3), fill_value=0.0, dtype=np.float32)],
        expected_images=[np.full(shape=(2, 2, 3), fill_value=9e-5, dtype=np.float32)],
        actual_audios=[],
        expected_audios=[],
    )


def _build_audio_within_tolerance_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build audio values differing by less than the default tolerance."""
    actual_audio = _build_audio(dtype=np.float32, sampling_rate=16000, format="wav")
    expected_audio = Audio(
        audio_array=np.array(object=[9e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
        sampling_rate=16000,
        format="wav",
    )
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[actual_audio],
        expected_audios=[expected_audio],
    )


def _build_token_value_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build values with one differing token ID."""
    return _build_tokenized_pair(
        actual_tokens=[1, 9, 3],
        expected_tokens=[1, 2, 3],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[],
        expected_audios=[],
    )


def _build_token_length_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build values whose token sequences have different lengths."""
    return _build_tokenized_pair(
        actual_tokens=[1, 2],
        expected_tokens=[1, 2, 3],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[],
        expected_audios=[],
    )


def _build_token_length_extra_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build token sequences where actual has one extra ID."""
    return _build_tokenized_pair(
        actual_tokens=[1, 2, 3],
        expected_tokens=[1, 2],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[],
        expected_audios=[],
    )


def _build_prefix_ids_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build values with unequal prefix IDs."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=[3],
        actual_images=[],
        expected_images=[],
        actual_audios=[],
        expected_audios=[],
    )


def _build_prefix_ids_empty_vs_none_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build values with empty and absent prefix IDs."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=[],
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[],
        expected_audios=[],
    )


def _build_prefix_ids_value_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build prefix ID lists that differ at one value."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=[1, 3],
        expected_prefix_ids=[1, 4],
        actual_images=[],
        expected_images=[],
        actual_audios=[],
        expected_audios=[],
    )


def _build_prefix_ids_length_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build prefix ID lists that differ by one trailing ID."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=[1],
        expected_prefix_ids=[1, 2],
        actual_images=[],
        expected_images=[],
        actual_audios=[],
        expected_audios=[],
    )


def _build_prefix_ids_length_extra_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build prefix ID lists where actual has one extra ID."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=[1, 2],
        expected_prefix_ids=[1],
        actual_images=[],
        expected_images=[],
        actual_audios=[],
        expected_audios=[],
    )


def _build_image_count_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build values with different image counts."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[np.zeros(shape=(2, 2, 3), dtype=np.float32)],
        actual_audios=[],
        expected_audios=[],
    )


def _build_image_count_extra_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build values where actual has one extra image."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[np.zeros(shape=(2, 2, 3), dtype=np.float32)],
        expected_images=[],
        actual_audios=[],
        expected_audios=[],
    )


def _build_image_shape_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build image values with different shapes."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[np.zeros(shape=(2, 2, 3), dtype=np.float32)],
        expected_images=[np.zeros(shape=(2, 3, 2), dtype=np.float32)],
        actual_audios=[],
        expected_audios=[],
    )


def _build_image_dtype_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build image values with different dtypes."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[np.zeros(shape=(2, 2, 3), dtype=np.float32)],
        expected_images=[np.zeros(shape=(2, 2, 3), dtype=np.float64)],
        actual_audios=[],
        expected_audios=[],
    )


def _build_image_value_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build image values differing beyond the default tolerance."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[np.zeros(shape=(2, 2, 3), dtype=np.float32)],
        expected_images=[np.full(shape=(2, 2, 3), fill_value=1.1e-4, dtype=np.float32)],
        actual_audios=[],
        expected_audios=[],
    )


def _build_image_value_relative_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build large image values that differ beyond absolute tolerance."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[np.full(shape=(2, 2, 3), fill_value=1000.0, dtype=np.float32)],
        expected_images=[np.full(shape=(2, 2, 3), fill_value=1000.005, dtype=np.float32)],
        actual_audios=[],
        expected_audios=[],
    )


def _build_image_second_value_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build two image lists whose second arrays differ beyond tolerance."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[
            np.zeros(shape=(2, 2, 3), dtype=np.float32),
            np.zeros(shape=(2, 2, 3), dtype=np.float32),
        ],
        expected_images=[
            np.zeros(shape=(2, 2, 3), dtype=np.float32),
            np.full(shape=(2, 2, 3), fill_value=1.1e-4, dtype=np.float32),
        ],
        actual_audios=[],
        expected_audios=[],
    )


def _build_audio_count_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build values with different audio counts."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[],
        expected_audios=[_build_audio(dtype=np.float32, sampling_rate=16000, format="wav")],
    )


def _build_audio_count_extra_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build values where actual has one extra audio."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[_build_audio(dtype=np.float32, sampling_rate=16000, format="wav")],
        expected_audios=[],
    )


def _build_audio_sampling_rate_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build audio values with different sampling rates."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[_build_audio(dtype=np.float32, sampling_rate=8000, format="wav")],
        expected_audios=[_build_audio(dtype=np.float32, sampling_rate=16000, format="wav")],
    )


def _build_audio_format_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build audio values with different file formats."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[_build_audio(dtype=np.float32, sampling_rate=16000, format="wav")],
        expected_audios=[_build_audio(dtype=np.float32, sampling_rate=16000, format="flac")],
    )


def _build_audio_shape_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build audio values with different array shapes."""
    actual_audio = _build_audio(dtype=np.float32, sampling_rate=16000, format="wav")
    expected_audio = Audio(
        audio_array=np.array(object=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        sampling_rate=16000,
        format="wav",
    )
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[actual_audio],
        expected_audios=[expected_audio],
    )


def _build_audio_dtype_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build audio values with different array dtypes."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[_build_audio(dtype=np.float32, sampling_rate=16000, format="wav")],
        expected_audios=[_build_audio(dtype=np.float64, sampling_rate=16000, format="wav")],
    )


def _build_audio_value_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build audio values differing beyond the default tolerance."""
    actual_audio = Audio(
        audio_array=np.zeros(shape=(8,), dtype=np.float32),
        sampling_rate=16000,
        format="wav",
    )
    expected_audio = Audio(
        audio_array=np.full(shape=(8,), fill_value=1.1e-4, dtype=np.float32),
        sampling_rate=16000,
        format="wav",
    )
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[actual_audio],
        expected_audios=[expected_audio],
    )


def _build_audio_second_value_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build two audio lists whose second arrays differ beyond tolerance."""
    actual_second = Audio(
        audio_array=np.zeros(shape=(8,), dtype=np.float32),
        sampling_rate=16000,
        format="wav",
    )
    expected_second = Audio(
        audio_array=np.full(shape=(8,), fill_value=1.1e-4, dtype=np.float32),
        sampling_rate=16000,
        format="wav",
    )
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[
            _build_audio(dtype=np.float32, sampling_rate=16000, format="wav"),
            actual_second,
        ],
        expected_audios=[
            _build_audio(dtype=np.float32, sampling_rate=16000, format="wav"),
            expected_second,
        ],
    )


def _build_audio_attribute_sets_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build equal audios where only actual has an extra attribute."""
    actual_audio = _build_audio(dtype=np.float32, sampling_rate=16000, format="wav")
    setattr(actual_audio, "extra", 1)
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[actual_audio],
        expected_audios=[_build_audio(dtype=np.float32, sampling_rate=16000, format="wav")],
    )


def _build_audio_attribute_sets_expected_extra_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build equal audios where only expected has an extra attribute."""
    expected_audio = _build_audio(dtype=np.float32, sampling_rate=16000, format="wav")
    setattr(expected_audio, "extra", 1)
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[_build_audio(dtype=np.float32, sampling_rate=16000, format="wav")],
        expected_audios=[expected_audio],
    )


def _build_image_dtype_before_values_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build images whose dtype and values both differ."""
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[np.zeros(shape=(2, 2, 3), dtype=np.float32)],
        expected_images=[np.ones(shape=(2, 2, 3), dtype=np.float64)],
        actual_audios=[],
        expected_audios=[],
    )


def _build_audio_sampling_rate_before_values_pair() -> tuple[Tokenized, Tokenized]:
    r"""Build audios whose sampling rate and samples both differ."""
    actual_audio = Audio(audio_array=np.zeros(shape=(8,), dtype=np.float32), sampling_rate=8000, format="wav")
    expected_audio = Audio(audio_array=np.ones(shape=(8,), dtype=np.float32), sampling_rate=16000, format="wav")
    return _build_tokenized_pair(
        actual_tokens=[1],
        expected_tokens=[1],
        actual_prefix_ids=None,
        expected_prefix_ids=None,
        actual_images=[],
        expected_images=[],
        actual_audios=[actual_audio],
        expected_audios=[expected_audio],
    )


@pytest.mark.parametrize(
    argnames="pair_builder",
    argvalues=[
        pytest.param(_build_identical_pair, id="identical"),
        pytest.param(_build_different_private_text_pair, id="different-private-text"),
        pytest.param(_build_image_within_tolerance_pair, id="image-within-tolerance"),
        pytest.param(_build_audio_within_tolerance_pair, id="audio-within-tolerance"),
    ],
)
def test_assert_tokenized_equal_accepts(pair_builder: Callable[[], tuple[Tokenized, Tokenized]]) -> None:
    r"""Accept equivalent public tokenized fields and media within tolerance."""
    actual, expected = pair_builder()

    assert_tokenized_equal(actual=actual, expected=expected)


@pytest.mark.parametrize(
    argnames="pair_builder, message",
    argvalues=[
        pytest.param(_build_token_value_pair, _TOKEN_VALUE_MESSAGE, id="token-value"),
        pytest.param(_build_token_length_pair, _TOKEN_LENGTH_MESSAGE, id="token-length"),
        pytest.param(_build_token_length_extra_pair, _TOKEN_LENGTH_EXTRA_MESSAGE, id="token-length-extra"),
        pytest.param(_build_prefix_ids_pair, _PREFIX_IDS_MESSAGE, id="prefix-ids"),
        pytest.param(_build_prefix_ids_empty_vs_none_pair, _PREFIX_IDS_MESSAGE, id="prefix-ids-empty-vs-none"),
        pytest.param(_build_prefix_ids_value_pair, _PREFIX_IDS_VALUE_MESSAGE, id="prefix-ids-value"),
        pytest.param(_build_prefix_ids_length_pair, _PREFIX_IDS_LENGTH_MESSAGE, id="prefix-ids-length"),
        pytest.param(
            _build_prefix_ids_length_extra_pair, _PREFIX_IDS_LENGTH_EXTRA_MESSAGE, id="prefix-ids-length-extra"
        ),
        pytest.param(_build_image_count_pair, _IMAGE_COUNT_MESSAGE, id="image-count"),
        pytest.param(_build_image_count_extra_pair, _IMAGE_COUNT_MESSAGE, id="image-count-extra"),
        pytest.param(_build_image_shape_pair, _IMAGE_SHAPE_MESSAGE, id="image-shape"),
        pytest.param(_build_image_dtype_pair, _IMAGE_DTYPE_MESSAGE, id="image-dtype"),
        pytest.param(_build_image_value_pair, _IMAGE_VALUE_MESSAGE, id="image-value"),
        pytest.param(_build_image_value_relative_pair, _IMAGE_VALUE_MESSAGE, id="image-value-relative"),
        pytest.param(_build_image_second_value_pair, _IMAGE_SECOND_VALUE_MESSAGE, id="image-second-value"),
        pytest.param(_build_audio_count_pair, _AUDIO_COUNT_MESSAGE, id="audio-count"),
        pytest.param(_build_audio_count_extra_pair, _AUDIO_COUNT_MESSAGE, id="audio-count-extra"),
        pytest.param(_build_audio_sampling_rate_pair, _AUDIO_SAMPLING_RATE_MESSAGE, id="audio-sampling-rate"),
        pytest.param(_build_audio_format_pair, _AUDIO_FORMAT_MESSAGE, id="audio-format"),
        pytest.param(_build_audio_shape_pair, _AUDIO_SHAPE_MESSAGE, id="audio-shape"),
        pytest.param(_build_audio_dtype_pair, _AUDIO_DTYPE_MESSAGE, id="audio-dtype"),
        pytest.param(_build_audio_value_pair, _AUDIO_VALUE_MESSAGE, id="audio-value"),
        pytest.param(_build_audio_second_value_pair, _AUDIO_SECOND_VALUE_MESSAGE, id="audio-second-value"),
        pytest.param(_build_audio_attribute_sets_pair, _AUDIO_ATTRIBUTE_SETS_MESSAGE, id="audio-attribute-sets-differ"),
        pytest.param(
            _build_audio_attribute_sets_expected_extra_pair,
            _AUDIO_ATTRIBUTE_SETS_MESSAGE,
            id="audio-attribute-sets-differ-expected-extra",
        ),
        pytest.param(_build_image_dtype_before_values_pair, _IMAGE_DTYPE_MESSAGE, id="image-dtype-before-values"),
        pytest.param(
            _build_audio_sampling_rate_before_values_pair,
            _AUDIO_SAMPLING_RATE_MESSAGE,
            id="audio-sampling-rate-before-values",
        ),
    ],
)
def test_assert_tokenized_equal_rejects(pair_builder: Callable[[], tuple[Tokenized, Tokenized]], message: str) -> None:
    r"""Reject the first mismatch in each compared tokenized field."""
    actual, expected = pair_builder()

    with pytest.raises(expected_exception=AssertionError, match=message):
        assert_tokenized_equal(actual=actual, expected=expected)


def test_tokenized_fields_are_all_compared() -> None:
    r"""Fail when `Tokenized` gains or loses a field.

    A new field must be given a comparison in `assert_tokenized_equal` before the
    constant is updated.
    """
    assert set(Tokenized.model_fields) == _TOKENIZED_FIELDS

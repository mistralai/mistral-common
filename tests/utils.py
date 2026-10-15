import numpy as np
from numpy.testing import assert_allclose

from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import InstructTokenizer, SpecialTokenPolicy, Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

TOKENIZED_ATOL: float = 1e-4
TOKENIZED_RTOL: float = 0.0

# Fields `assert_tokenized_equal` compares; a guard test fails when `Tokenized` gains or loses one.
TOKENIZED_FIELDS: frozenset[str] = frozenset({"tokens", "prefix_ids", "images", "audios"})


def decode_keep(tokenizer: InstructTokenizer | MistralTokenizer, tokenized: Tokenized) -> str:
    r"""Decode `tokenized.tokens` back to text, keeping special tokens.

    Args:
        tokenizer: The tokenizer that produced `tokenized`.
        tokenized: The tokenized result to decode.

    Returns:
        The decoded string, including special tokens.
    """
    return tokenizer.decode(tokens=tokenized.tokens, special_token_policy=SpecialTokenPolicy.KEEP)


def _assert_ids_equal(*, field: str, actual: list[int] | None, expected: list[int] | None) -> None:
    r"""Assert two optional ID lists are equal, naming the first differing index.

    Args:
        field: The field name used in the failure message.
        actual: The value produced by the tokenizer.
        expected: The reviewed value to compare against.

    Raises:
        AssertionError: If one side is `None` and the other is not (reported at index 0), or at the first
            differing index, with `<end>` marking the side that is shorter.
    """
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{field} differ at index 0: actual {actual!r}, expected {expected!r}")
        return

    for index in range(max(len(actual), len(expected))):
        actual_id = actual[index] if index < len(actual) else "<end>"
        expected_id = expected[index] if index < len(expected) else "<end>"
        if actual_id != expected_id:
            raise AssertionError(f"{field} differ at index {index}: actual {actual_id!r}, expected {expected_id!r}")


def _assert_array_close(
    *, field: str, index: int, actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float
) -> None:
    r"""Assert two arrays have the same shape and dtype and close values.

    Args:
        field: The field name used in the failure message.
        index: The position of the compared item within its field.
        actual: The array produced by the tokenizer.
        expected: The reviewed array to compare against.
        atol: Absolute tolerance for the values.
        rtol: Relative tolerance for the values.

    Raises:
        AssertionError: If the shape, the dtype or a value differs.
    """
    if actual.shape != expected.shape:
        raise AssertionError(f"{field} differ at index {index}: shape {actual.shape} != {expected.shape}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{field} differ at index {index}: dtype {actual.dtype} != {expected.dtype}")
    assert_allclose(
        actual=actual,
        desired=expected,
        atol=atol,
        rtol=rtol,
        err_msg=f"{field} differ at index {index}: values",
    )


def _assert_audio_equal(*, index: int, actual: Audio, expected: Audio, atol: float, rtol: float) -> None:
    r"""Assert two audios have equal metadata and close samples.

    Args:
        index: The position of the audio within `Tokenized.audios`.
        actual: The audio produced by the tokenizer.
        expected: The reviewed audio to compare against.
        atol: Absolute tolerance for the samples.
        rtol: Relative tolerance for the samples.

    Raises:
        AssertionError: If the attributes other than the samples differ, or the samples differ.
    """
    actual_attributes = {name: value for name, value in vars(actual).items() if name != "audio_array"}
    expected_attributes = {name: value for name, value in vars(expected).items() if name != "audio_array"}
    if actual_attributes != expected_attributes:
        raise AssertionError(f"audios differ at index {index}: attributes {actual_attributes} != {expected_attributes}")
    _assert_array_close(
        field="audios",
        index=index,
        actual=actual.audio_array,
        expected=expected.audio_array,
        atol=atol,
        rtol=rtol,
    )


def assert_tokenized_equal(
    actual: Tokenized,
    expected: Tokenized,
    *,
    atol: float = TOKENIZED_ATOL,
    rtol: float = TOKENIZED_RTOL,
) -> None:
    r"""Compare `tokens`, `prefix_ids`, `images` and `audios`, ignoring the deprecated text.

    Args:
        actual: The value produced by the tokenizer.
        expected: The reviewed value to compare against.
        atol: Absolute tolerance for image and audio samples.
        rtol: Relative tolerance for image and audio samples.

    Raises:
        AssertionError: If a field differs, with the first failing field and index.
    """
    _assert_ids_equal(field="tokens", actual=actual.tokens, expected=expected.tokens)
    _assert_ids_equal(field="prefix_ids", actual=actual.prefix_ids, expected=expected.prefix_ids)

    if len(actual.images) != len(expected.images):
        index = min(len(actual.images), len(expected.images))
        raise AssertionError(f"images differ at index {index}: count {len(actual.images)} != {len(expected.images)}")
    for index, (actual_image, expected_image) in enumerate(zip(actual.images, expected.images)):
        _assert_array_close(
            field="images", index=index, actual=actual_image, expected=expected_image, atol=atol, rtol=rtol
        )

    if len(actual.audios) != len(expected.audios):
        index = min(len(actual.audios), len(expected.audios))
        raise AssertionError(f"audios differ at index {index}: count {len(actual.audios)} != {len(expected.audios)}")
    for index, (actual_audio, expected_audio) in enumerate(zip(actual.audios, expected.audios)):
        _assert_audio_equal(index=index, actual=actual_audio, expected=expected_audio, atol=atol, rtol=rtol)

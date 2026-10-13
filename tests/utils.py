import numpy as np
from numpy.testing import assert_allclose

from mistral_common.tokens.tokenizers.base import InstructTokenizer, SpecialTokenPolicy, Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

TOKENIZED_ATOL: float = 1e-4
TOKENIZED_RTOL: float = 0.0


def decode_keep(tokenizer: InstructTokenizer | MistralTokenizer, tokenized: Tokenized) -> str:
    r"""Decode `tokenized.tokens` back to text, keeping special tokens.

    Args:
        tokenizer: The tokenizer that produced `tokenized`.
        tokenized: The tokenized result to decode.

    Returns:
        The decoded string, including special tokens.
    """
    return tokenizer.decode(tokens=tokenized.tokens, special_token_policy=SpecialTokenPolicy.KEEP)


def assert_tokenized_equal(
    actual: Tokenized,
    expected: Tokenized,
    *,
    atol: float = TOKENIZED_ATOL,
    rtol: float = TOKENIZED_RTOL,
) -> None:
    r"""Compare token IDs, prefixes, and media while ignoring deprecated text.

    Args:
        actual: The value produced by the tokenizer.
        expected: The reviewed value to compare against.
        atol: Absolute tolerance for image and audio samples.
        rtol: Relative tolerance for image and audio samples.

    Raises:
        AssertionError: If a field differs, with the first failing field and index.
    """
    if actual.tokens != expected.tokens:
        for index in range(max(len(actual.tokens), len(expected.tokens))):
            actual_id = actual.tokens[index] if index < len(actual.tokens) else "<end>"
            expected_id = expected.tokens[index] if index < len(expected.tokens) else "<end>"
            if actual_id != expected_id:
                raise AssertionError(f"tokens differ at index {index}: actual {actual_id!r}, expected {expected_id!r}")

    if actual.prefix_ids is None or expected.prefix_ids is None:
        if actual.prefix_ids is not expected.prefix_ids:
            raise AssertionError(
                f"prefix_ids differ at index 0: actual {actual.prefix_ids!r}, expected {expected.prefix_ids!r}"
            )
    elif actual.prefix_ids != expected.prefix_ids:
        for index in range(max(len(actual.prefix_ids), len(expected.prefix_ids))):
            actual_prefix_id = actual.prefix_ids[index] if index < len(actual.prefix_ids) else "<end>"
            expected_prefix_id = expected.prefix_ids[index] if index < len(expected.prefix_ids) else "<end>"
            if actual_prefix_id != expected_prefix_id:
                raise AssertionError(
                    f"prefix_ids differ at index {index}: actual {actual_prefix_id!r}, expected {expected_prefix_id!r}"
                )

    if len(actual.images) != len(expected.images):
        index = min(len(actual.images), len(expected.images))
        raise AssertionError(f"images differ at index {index}: count {len(actual.images)} != {len(expected.images)}")

    for index, (actual_image, expected_image) in enumerate(zip(actual.images, expected.images)):
        if actual_image.shape != expected_image.shape:
            raise AssertionError(
                f"images differ at index {index}: shape {actual_image.shape} != {expected_image.shape}"
            )
        if actual_image.dtype != expected_image.dtype:
            raise AssertionError(
                f"images differ at index {index}: dtype {actual_image.dtype} != {expected_image.dtype}"
            )
        assert_allclose(
            actual=actual_image.astype(dtype=np.float64, copy=True),
            desired=expected_image.astype(dtype=np.float64, copy=True),
            atol=atol,
            rtol=rtol,
            err_msg=f"images differ at index {index}: values",
        )

    if len(actual.audios) != len(expected.audios):
        index = min(len(actual.audios), len(expected.audios))
        raise AssertionError(f"audios differ at index {index}: count {len(actual.audios)} != {len(expected.audios)}")

    for index, (actual_audio, expected_audio) in enumerate(zip(actual.audios, expected.audios)):
        if actual_audio.sampling_rate != expected_audio.sampling_rate:
            raise AssertionError(
                f"audios differ at index {index}: sampling_rate {actual_audio.sampling_rate} "
                f"!= {expected_audio.sampling_rate}"
            )
        if actual_audio.format != expected_audio.format:
            raise AssertionError(
                f"audios differ at index {index}: format {actual_audio.format!r} != {expected_audio.format!r}"
            )
        if actual_audio.audio_array.shape != expected_audio.audio_array.shape:
            raise AssertionError(
                f"audios differ at index {index}: shape {actual_audio.audio_array.shape} "
                f"!= {expected_audio.audio_array.shape}"
            )
        if actual_audio.audio_array.dtype != expected_audio.audio_array.dtype:
            raise AssertionError(
                f"audios differ at index {index}: dtype {actual_audio.audio_array.dtype} "
                f"!= {expected_audio.audio_array.dtype}"
            )
        assert_allclose(
            actual=actual_audio.audio_array.astype(dtype=np.float64, copy=True),
            desired=expected_audio.audio_array.astype(dtype=np.float64, copy=True),
            atol=atol,
            rtol=rtol,
            err_msg=f"audios differ at index {index}: values",
        )

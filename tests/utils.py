import base64
import json
from typing import Sequence

import numpy as np
from numpy.testing import assert_allclose

from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import (
    InstructTokenizer,
    SpecialTokenPolicy,
    SpecialTokens,
    Tokenized,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer, TokenInfo

TOKENIZED_ATOL: float = 1e-4
TOKENIZED_RTOL: float = 0.0

# Fields `assert_tokenized_equal` compares; a guard test fails when `Tokenized` gains or loses one.
TOKENIZED_FIELDS: frozenset[str] = frozenset({"tokens", "prefix_ids", "images", "audios"})


def is_unloadable_json(payload: str) -> bool:
    r"""Whether `json.loads` rejects `payload` with `ValueError` or `RecursionError`.

    On Python 3.14+ C recursion is guarded by remaining stack space instead of the recursion limit, so deeply
    nested JSON may parse instead of raising `RecursionError`, depending on the thread stack size.
    """
    try:
        json.loads(payload)
    except (ValueError, RecursionError):
        return True
    return False


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


def quick_vocab(extra_toks: Sequence[bytes] = ()) -> list[TokenInfo]:
    vocab = [TokenInfo(rank=i, token_bytes=base64.b64encode(bytes([i])).decode(), token_str=chr(i)) for i in range(256)]
    for i, tok in enumerate(extra_toks):
        vocab.append(
            TokenInfo(
                rank=256 + i,
                token_bytes=base64.b64encode(tok).decode(),
                token_str=tok.decode(),
            )
        )
    return vocab


def get_special_tokens(
    tokenizer_version: TokenizerVersion, add_audio: bool = False, add_think: bool = False
) -> list[SpecialTokenInfo]:
    special_tokens = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
    if tokenizer_version < TokenizerVersion.v7 and add_audio:
        raise ValueError("Audio tokens are only supported in v7 and above")

    if tokenizer_version <= TokenizerVersion.v7 and not add_audio:
        return special_tokens

    # fill special tokens until 24
    special_tokens += [
        SpecialTokenInfo(rank=i, token_str=f"<SPECIAL_{i}>", is_control=True) for i in range(len(special_tokens), 24)
    ]

    if add_audio:
        # add audio tokes
        special_tokens += [
            SpecialTokenInfo(rank=24, token_str=SpecialTokens.audio, is_control=True),
            SpecialTokenInfo(rank=25, token_str=SpecialTokens.begin_audio, is_control=True),
        ]

    # fill special tokens until 32
    special_tokens += [
        SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True) for i in range(len(special_tokens), 32)
    ]

    if tokenizer_version > TokenizerVersion.v7:
        special_tokens += [
            SpecialTokenInfo(rank=32, token_str=SpecialTokens.args, is_control=True),
            SpecialTokenInfo(rank=33, token_str=SpecialTokens.call_id, is_control=True),
        ]

    # fill special tokens until 34
    special_tokens += [
        SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True) for i in range(len(special_tokens), 34)
    ]

    if add_audio:
        assert not add_think, f"Audio and think tokens are mutually exclusive, got {add_audio} and {add_think}"
        special_tokens += [
            SpecialTokenInfo(rank=34, token_str=SpecialTokens.transcribe, is_control=True),
            SpecialTokenInfo(rank=35, token_str=SpecialTokens.text_to_audio, is_control=True),
            SpecialTokenInfo(rank=36, token_str=SpecialTokens.audio_to_text, is_control=True),
        ]

    if tokenizer_version < TokenizerVersion.v13:
        return special_tokens

    if not add_audio and add_think:
        special_tokens += [SpecialTokenInfo(rank=34, token_str=f"<SPCECIAL_{34}>", is_control=True)]

    if add_think:
        special_tokens += [
            SpecialTokenInfo(rank=35, token_str="[THINK]", is_control=True),
            SpecialTokenInfo(rank=36, token_str="[/THINK]", is_control=True),
        ]

    if tokenizer_version >= TokenizerVersion.v15:
        # fill until rank 37
        special_tokens += [
            SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True)
            for i in range(len(special_tokens), 37)
        ]
        special_tokens += [
            SpecialTokenInfo(rank=37, token_str=SpecialTokens.begin_model_settings, is_control=True),
            SpecialTokenInfo(rank=38, token_str=SpecialTokens.end_model_settings, is_control=True),
        ]

    return special_tokens

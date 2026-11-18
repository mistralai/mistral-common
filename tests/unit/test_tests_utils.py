r"""Test the helpers shared by tests (`tests/utils.py`).

Most cases build two copies of one reference `Tokenized` with every field set, apply a named
change to each copy, and expect `assert_tokenized_equal` to accept or reject the pair, with the
default tolerances or caller-supplied ones. One guard test checks the list of compared fields.
The remaining tests pin the suite-wide tolerance pair and the synthetic Tekken builders
`quick_vocab` and `get_special_tokens`.
"""

import base64
from collections.abc import Callable

import numpy as np
import pytest

from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import SpecialTokens, Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.utils import (
    TOKENIZED_ATOL,
    TOKENIZED_FIELDS,
    TOKENIZED_RTOL,
    assert_tokenized_equal,
    get_special_tokens,
    quick_vocab,
)

_Change = Callable[[Tokenized], None]


def _build_reference() -> Tokenized:
    r"""Build a fresh value with every field set: two images and two audios."""
    return Tokenized(
        tokens=[1, 2, 3],
        prefix_ids=[2, 3],
        images=[np.zeros(shape=(2, 2, 3), dtype=np.float32) for _ in range(2)],
        audios=[
            Audio(
                audio_array=np.array(object=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
                sampling_rate=16000,
                format="wav",
            )
            for _ in range(2)
        ],
    )


def _unchanged(tokenized: Tokenized) -> None:
    r"""Leave the reference as it is."""


def _change_second_token(tokenized: Tokenized) -> None:
    r"""Replace the second token ID with 9."""
    tokenized.tokens[1] = 9


def _drop_last_token(tokenized: Tokenized) -> None:
    r"""Remove the last token ID."""
    tokenized.tokens.pop()


def _append_token(tokenized: Tokenized) -> None:
    r"""Append the token ID 4."""
    tokenized.tokens.append(4)


def _set_prefix_ids_to_none(tokenized: Tokenized) -> None:
    r"""Remove the prefix IDs."""
    tokenized.prefix_ids = None


def _set_prefix_ids_to_empty(tokenized: Tokenized) -> None:
    r"""Set the prefix IDs to an empty list."""
    tokenized.prefix_ids = []


def _change_second_prefix_id(tokenized: Tokenized) -> None:
    r"""Replace the second prefix ID with 9."""
    assert tokenized.prefix_ids is not None
    tokenized.prefix_ids[1] = 9


def _drop_last_prefix_id(tokenized: Tokenized) -> None:
    r"""Remove the last prefix ID."""
    assert tokenized.prefix_ids is not None
    tokenized.prefix_ids.pop()


def _append_prefix_id(tokenized: Tokenized) -> None:
    r"""Append the prefix ID 4."""
    assert tokenized.prefix_ids is not None
    tokenized.prefix_ids.append(4)


def _drop_last_image(tokenized: Tokenized) -> None:
    r"""Remove the last image."""
    tokenized.images.pop()


def _replace_first_image_with_other_shape(tokenized: Tokenized) -> None:
    r"""Replace the first image with zeros of shape (2, 3, 2)."""
    tokenized.images[0] = np.zeros(shape=(2, 3, 2), dtype=np.float32)


def _replace_first_image_with_float64_zeros(tokenized: Tokenized) -> None:
    r"""Replace the first image with float64 zeros of the same shape."""
    tokenized.images[0] = np.zeros(shape=(2, 2, 3), dtype=np.float64)


def _replace_first_image_with_float64_ones(tokenized: Tokenized) -> None:
    r"""Replace the first image with float64 ones, changing dtype and values together."""
    tokenized.images[0] = np.ones(shape=(2, 2, 3), dtype=np.float64)


def _shift_first_image_within_tolerance(tokenized: Tokenized) -> None:
    r"""Fill the first image with 9e-5, below the default tolerance of 1e-4."""
    tokenized.images[0].fill(9e-5)


def _shift_first_image_beyond_tolerance(tokenized: Tokenized) -> None:
    r"""Fill the first image with 1.1e-4, above the default tolerance of 1e-4."""
    tokenized.images[0].fill(1.1e-4)


def _shift_second_image_beyond_tolerance(tokenized: Tokenized) -> None:
    r"""Fill the second image with 1.1e-4, above the default tolerance of 1e-4."""
    tokenized.images[1].fill(1.1e-4)


def _set_first_image_to_1000(tokenized: Tokenized) -> None:
    r"""Fill the first image with 1000.0."""
    tokenized.images[0].fill(1000.0)


def _set_first_image_to_1000_005(tokenized: Tokenized) -> None:
    r"""Fill the first image with 1000.005, which only a relative tolerance would accept."""
    tokenized.images[0].fill(1000.005)


def _drop_last_audio(tokenized: Tokenized) -> None:
    r"""Remove the last audio."""
    tokenized.audios.pop()


def _set_first_audio_sampling_rate_to_8000(tokenized: Tokenized) -> None:
    r"""Set the first audio's sampling rate to 8000."""
    tokenized.audios[0].sampling_rate = 8000


def _set_first_audio_sampling_rate_to_8000_and_shift_samples(tokenized: Tokenized) -> None:
    r"""Change the first audio's sampling rate and its samples together."""
    tokenized.audios[0].sampling_rate = 8000
    tokenized.audios[0].audio_array = tokenized.audios[0].audio_array + 1.0


def _set_first_audio_format_to_flac(tokenized: Tokenized) -> None:
    r"""Set the first audio's format to flac."""
    tokenized.audios[0].format = "flac"


def _drop_last_sample_of_first_audio(tokenized: Tokenized) -> None:
    r"""Remove the last sample of the first audio."""
    tokenized.audios[0].audio_array = tokenized.audios[0].audio_array[:-1]


def _cast_first_audio_to_float64(tokenized: Tokenized) -> None:
    r"""Cast the first audio's samples to float64."""
    tokenized.audios[0].audio_array = tokenized.audios[0].audio_array.astype(np.float64)


def _shift_first_audio_within_tolerance(tokenized: Tokenized) -> None:
    r"""Set the first sample of the first audio to 9e-5, below the default tolerance of 1e-4."""
    tokenized.audios[0].audio_array[0] = 9e-5


def _shift_first_audio_beyond_tolerance(tokenized: Tokenized) -> None:
    r"""Set the first sample of the first audio to 1.1e-4, above the default tolerance of 1e-4."""
    tokenized.audios[0].audio_array[0] = 1.1e-4


def _shift_second_audio_beyond_tolerance(tokenized: Tokenized) -> None:
    r"""Set the first sample of the second audio to 1.1e-4, above the default tolerance of 1e-4."""
    tokenized.audios[1].audio_array[0] = 1.1e-4


def _set_first_audio_sample_to_1000(tokenized: Tokenized) -> None:
    r"""Set the first sample of the first audio to 1000.0."""
    tokenized.audios[0].audio_array[0] = 1000.0


def _set_first_audio_sample_to_1000_005(tokenized: Tokenized) -> None:
    r"""Set the first sample of the first audio to 1000.005, which only a relative tolerance would accept."""
    tokenized.audios[0].audio_array[0] = 1000.005


def _add_extra_attribute_to_first_audio(tokenized: Tokenized) -> None:
    r"""Set an attribute named `extra` on the first audio."""
    # `Audio` declares no `extra` attribute, so a direct assignment fails mypy.
    setattr(tokenized.audios[0], "extra", 1)


@pytest.mark.parametrize(
    argnames="actual_change, expected_change",
    argvalues=[
        pytest.param(_unchanged, _unchanged, id="identical"),
        pytest.param(_set_prefix_ids_to_none, _set_prefix_ids_to_none, id="prefix-ids-both-none"),
        pytest.param(_shift_first_image_within_tolerance, _unchanged, id="images-value-within-tolerance"),
        pytest.param(_shift_first_audio_within_tolerance, _unchanged, id="audios-value-within-tolerance"),
    ],
)
def test_assert_tokenized_equal_accepts(actual_change: _Change, expected_change: _Change) -> None:
    r"""Accept a pair that differs only where the helper ignores or tolerates it."""
    actual = _build_reference()
    expected = _build_reference()
    actual_change(actual)
    expected_change(expected)

    assert_tokenized_equal(actual=actual, expected=expected)


@pytest.mark.parametrize(
    argnames="actual_change, expected_change, message",
    argvalues=[
        pytest.param(_change_second_token, _unchanged, r"tokens.*index 1.*9.*2", id="tokens-value-differs"),
        pytest.param(_drop_last_token, _unchanged, r"tokens.*index 2.*<end>.*3", id="tokens-actual-shorter"),
        pytest.param(_append_token, _unchanged, r"tokens.*index 3.*4.*<end>", id="tokens-actual-longer"),
        pytest.param(_set_prefix_ids_to_none, _unchanged, r"prefix_ids.*index 0", id="prefix-ids-actual-none"),
        pytest.param(
            _set_prefix_ids_to_empty, _set_prefix_ids_to_none, r"prefix_ids.*index 0", id="prefix-ids-empty-vs-none"
        ),
        pytest.param(_change_second_prefix_id, _unchanged, r"prefix_ids.*index 1.*9.*3", id="prefix-ids-value-differs"),
        pytest.param(
            _drop_last_prefix_id, _unchanged, r"prefix_ids.*index 1.*<end>.*3", id="prefix-ids-actual-shorter"
        ),
        pytest.param(_append_prefix_id, _unchanged, r"prefix_ids.*index 2.*4.*<end>", id="prefix-ids-actual-longer"),
        pytest.param(_drop_last_image, _unchanged, r"images.*index 1.*count", id="images-actual-fewer"),
        pytest.param(_unchanged, _drop_last_image, r"images.*index 1.*count", id="images-actual-more"),
        pytest.param(
            _replace_first_image_with_other_shape, _unchanged, r"images.*index 0.*shape", id="images-shape-differs"
        ),
        pytest.param(
            _replace_first_image_with_float64_zeros, _unchanged, r"images.*index 0.*dtype", id="images-dtype-differs"
        ),
        pytest.param(
            _shift_first_image_beyond_tolerance, _unchanged, r"images.*index 0.*values", id="images-value-differs"
        ),
        pytest.param(
            _set_first_image_to_1000,
            _set_first_image_to_1000_005,
            r"images.*index 0.*values",
            id="images-value-differs-relative",
        ),
        pytest.param(
            _shift_second_image_beyond_tolerance,
            _unchanged,
            r"images.*index 1.*values",
            id="images-second-value-differs",
        ),
        pytest.param(
            _replace_first_image_with_float64_ones,
            _unchanged,
            r"images.*index 0.*dtype",
            id="images-dtype-and-value-differ",
        ),
        pytest.param(_drop_last_audio, _unchanged, r"audios.*index 1.*count", id="audios-actual-fewer"),
        pytest.param(_unchanged, _drop_last_audio, r"audios.*index 1.*count", id="audios-actual-more"),
        pytest.param(
            _set_first_audio_sampling_rate_to_8000,
            _unchanged,
            r"audios.*index 0.*sampling_rate",
            id="audios-sampling-rate-differs",
        ),
        pytest.param(
            _set_first_audio_sampling_rate_to_8000_and_shift_samples,
            _unchanged,
            r"audios.*index 0.*sampling_rate",
            id="audios-sampling-rate-and-value-differ",
        ),
        pytest.param(
            _set_first_audio_format_to_flac, _unchanged, r"audios.*index 0.*format", id="audios-format-differs"
        ),
        pytest.param(
            _drop_last_sample_of_first_audio, _unchanged, r"audios.*index 0.*shape", id="audios-shape-differs"
        ),
        pytest.param(_cast_first_audio_to_float64, _unchanged, r"audios.*index 0.*dtype", id="audios-dtype-differs"),
        pytest.param(
            _shift_first_audio_beyond_tolerance, _unchanged, r"audios.*index 0.*values", id="audios-value-differs"
        ),
        pytest.param(
            _shift_second_audio_beyond_tolerance,
            _unchanged,
            r"audios.*index 1.*values",
            id="audios-second-value-differs",
        ),
        pytest.param(
            _add_extra_attribute_to_first_audio,
            _unchanged,
            r"audios.*index 0.*extra",
            id="audios-attribute-only-in-actual",
        ),
        pytest.param(
            _unchanged,
            _add_extra_attribute_to_first_audio,
            r"audios.*index 0.*extra",
            id="audios-attribute-only-in-expected",
        ),
    ],
)
def test_assert_tokenized_equal_rejects(actual_change: _Change, expected_change: _Change, message: str) -> None:
    r"""Reject a pair that differs in one field, naming that field and the first differing index."""
    actual = _build_reference()
    expected = _build_reference()
    actual_change(actual)
    expected_change(expected)

    with pytest.raises(expected_exception=AssertionError, match=message):
        assert_tokenized_equal(actual=actual, expected=expected)


@pytest.mark.parametrize(
    argnames="actual_change, expected_change",
    argvalues=[
        pytest.param(_set_first_image_to_1000, _set_first_image_to_1000_005, id="images-looser-rtol"),
        pytest.param(_set_first_audio_sample_to_1000, _set_first_audio_sample_to_1000_005, id="audios-looser-rtol"),
    ],
)
def test_assert_tokenized_equal_accepts_within_caller_tolerance(
    actual_change: _Change, expected_change: _Change
) -> None:
    r"""Accept a relative difference that the default `rtol` of 0 rejects but the caller's `rtol` allows."""
    actual = _build_reference()
    expected = _build_reference()
    actual_change(actual)
    expected_change(expected)

    assert_tokenized_equal(actual=actual, expected=expected, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize(
    argnames="actual_change, message",
    argvalues=[
        pytest.param(_shift_first_image_within_tolerance, r"images.*index 0.*values", id="images-tighter-atol"),
        pytest.param(_shift_first_audio_within_tolerance, r"audios.*index 0.*values", id="audios-tighter-atol"),
    ],
)
def test_assert_tokenized_equal_rejects_beyond_caller_tolerance(actual_change: _Change, message: str) -> None:
    r"""Reject a difference that the default tolerance accepts but the caller's tighter `atol` does not."""
    actual = _build_reference()
    expected = _build_reference()
    actual_change(actual)

    with pytest.raises(expected_exception=AssertionError, match=message):
        assert_tokenized_equal(actual=actual, expected=expected, atol=5e-5, rtol=0.0)


def test_tokenized_fields_are_all_compared() -> None:
    r"""Fail when `Tokenized` gains or loses a field.

    A new field must be given a comparison in `assert_tokenized_equal` before the
    constant is updated.
    """
    assert set(Tokenized.model_fields) == TOKENIZED_FIELDS


def test_tokenized_tolerances_are_the_suite_wide_pair() -> None:
    r"""Pin the one suite-wide numeric tolerance pair used as `assert_tokenized_equal` defaults."""
    assert TOKENIZED_ATOL == 1e-4
    assert TOKENIZED_RTOL == 0.0


def test_quick_vocab_builds_the_256_byte_tokens() -> None:
    r"""Build one token per byte value, ranked and base64-encoded in order."""
    vocab = quick_vocab()
    assert len(vocab) == 256
    for rank, token in enumerate(vocab):
        assert token["rank"] == rank
        assert base64.b64decode(token["token_bytes"]) == bytes([rank])
        assert token["token_str"] == chr(rank)


def test_quick_vocab_appends_extra_tokens_after_the_byte_tokens() -> None:
    r"""Append extra tokens at ranks 256+, base64-encoded and decoded as their string."""
    vocab = quick_vocab(extra_toks=[b"synthetic", b"xy"])
    assert len(vocab) == 258
    assert vocab[256] == {
        "rank": 256,
        "token_bytes": base64.b64encode(b"synthetic").decode(),
        "token_str": "synthetic",
    }
    assert vocab[257] == {"rank": 257, "token_bytes": base64.b64encode(b"xy").decode(), "token_str": "xy"}


@pytest.mark.parametrize("version", [TokenizerVersion.v1, TokenizerVersion.v2, TokenizerVersion.v3])
def test_get_special_tokens_rejects_audio_before_v7(version: TokenizerVersion) -> None:
    r"""Reject `add_audio` for tokenizer versions below v7."""
    with pytest.raises(ValueError, match="Audio tokens are only supported in v7 and above"):
        get_special_tokens(tokenizer_version=version, add_audio=True)


@pytest.mark.parametrize("version", [TokenizerVersion.v1, TokenizerVersion.v7])
def test_get_special_tokens_returns_deprecated_tokens_for_v7_and_below(version: TokenizerVersion) -> None:
    r"""Return exactly the deprecated special tokens for v7 and below, ignoring `add_think`."""
    assert get_special_tokens(tokenizer_version=version) == list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
    assert get_special_tokens(tokenizer_version=version, add_think=True) == list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)


def test_get_special_tokens_v7_audio_layout() -> None:
    r"""Fill to rank 24, add the audio pair, fill to 34, then the three audio-mode tokens."""
    tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v7, add_audio=True)
    assert len(tokens) == 37
    assert [token["rank"] for token in tokens] == list(range(37))
    assert tokens[24]["token_str"] == SpecialTokens.audio
    assert tokens[25]["token_str"] == SpecialTokens.begin_audio
    assert tokens[34]["token_str"] == SpecialTokens.transcribe
    assert tokens[35]["token_str"] == SpecialTokens.text_to_audio
    assert tokens[36]["token_str"] == SpecialTokens.audio_to_text
    assert all(token["is_control"] for token in tokens)


def test_get_special_tokens_v11_adds_tool_tokens() -> None:
    r"""Fill to rank 32, then add `[ARGS]` and `[CALL_ID]` for versions above v7."""
    tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v11)
    assert len(tokens) == 34
    assert [token["rank"] for token in tokens] == list(range(34))
    assert tokens[20]["token_str"] == "<SPECIAL_20>"
    assert tokens[24]["token_str"] == "<SPCECIAL_24>"
    assert tokens[32]["token_str"] == SpecialTokens.args
    assert tokens[33]["token_str"] == SpecialTokens.call_id


def test_get_special_tokens_v13_think_layout() -> None:
    r"""Add the think pair at ranks 35-36 for v13 and above when `add_think` is set."""
    tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v13, add_think=True)
    assert len(tokens) == 37
    assert tokens[34]["token_str"] == "<SPCECIAL_34>"
    assert tokens[35]["token_str"] == "[THINK]"
    assert tokens[36]["token_str"] == "[/THINK]"


def test_get_special_tokens_v15_adds_model_settings() -> None:
    r"""Fill to rank 37, then add the model-settings pair for v15 and above."""
    tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v15)
    assert len(tokens) == 39
    assert [token["rank"] for token in tokens] == list(range(39))
    assert tokens[37]["token_str"] == SpecialTokens.begin_model_settings
    assert tokens[38]["token_str"] == SpecialTokens.end_model_settings


@pytest.mark.parametrize(
    "version", [TokenizerVersion.v7, TokenizerVersion.v11, TokenizerVersion.v13, TokenizerVersion.v15]
)
def test_get_special_tokens_audio_and_think_are_mutually_exclusive(version: TokenizerVersion) -> None:
    r"""Reject `add_audio` together with `add_think` for v7 and above."""
    with pytest.raises(AssertionError, match="Audio and think tokens are mutually exclusive"):
        get_special_tokens(tokenizer_version=version, add_audio=True, add_think=True)


@pytest.mark.parametrize("version", list(TokenizerVersion))
@pytest.mark.parametrize(("add_audio", "add_think"), [(False, False), (False, True), (True, False)])
def test_get_special_tokens_ranks_are_contiguous(version: TokenizerVersion, add_audio: bool, add_think: bool) -> None:
    r"""Every accepted combination yields tokens ranked 0..n-1 without gaps or duplicates."""
    if version < TokenizerVersion.v7 and add_audio:
        with pytest.raises(ValueError, match="only supported in v7 and above"):
            get_special_tokens(tokenizer_version=version, add_audio=add_audio, add_think=add_think)
        return
    if add_audio and add_think:
        with pytest.raises(AssertionError, match="mutually exclusive"):
            get_special_tokens(tokenizer_version=version, add_audio=add_audio, add_think=add_think)
        return
    tokens = get_special_tokens(tokenizer_version=version, add_audio=add_audio, add_think=add_think)
    assert [token["rank"] for token in tokens] == list(range(len(tokens)))

r"""Test the synthetic Tekken builders (`tests/utils/special_tokens.py`)."""

import base64

import pytest

from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.utils.special_tokens import get_special_tokens, quick_vocab


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

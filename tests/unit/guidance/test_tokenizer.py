r"""Unit tests for `mistral_common.guidance.tokenizer`.

Migrated from the legacy `tests/guidance/test_tokenizer.py`. The synthetic
Tekken/Mistral/llguidance resources are immutable, so they are built once at
module level; every former loop input is an independently named parametrize
row.
"""

import re
from unittest.mock import MagicMock

import llguidance as llg
import pytest

from mistral_common.guidance.tokenizer import MistralLLGTokenizer, from_mistral_tokenizer
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens, Tokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV7
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer
from tests.utils.special_tokens import get_special_tokens, quick_vocab

_NUM_SPECIAL_TOKENS = 100
_EXTRA_TOKENS = [b"a", b"b", b"c", b"f", b"de", b"he", b"llo"]

TEKKENIZER = Tekkenizer(
    quick_vocab(_EXTRA_TOKENS),
    special_tokens=get_special_tokens(TokenizerVersion.v7),
    pattern=r".+",
    vocab_size=256 + _NUM_SPECIAL_TOKENS,
    num_special_tokens=_NUM_SPECIAL_TOKENS,
    version=TokenizerVersion.v7,
)
LLG_TOKENIZER = MistralLLGTokenizer(TEKKENIZER)
MISTRAL_TOKENIZER = MistralTokenizer(
    InstructTokenizerV7(TEKKENIZER),
    validator=get_validator(TEKKENIZER.version, mode=ValidationMode.test),
    request_normalizer=get_normalizer(TEKKENIZER.version, TEKKENIZER.model_settings_builder),
)
LL_TOKENIZER = from_mistral_tokenizer(MISTRAL_TOKENIZER)


def _expected_special_str(token_value: str) -> str:
    r"""The llguidance spelling of a special token: bracketed tokens become angle-bracketed."""
    if re.fullmatch(r"\[.*\]", token_value):
        return token_value.replace("[", "<").replace("]", ">")
    return token_value


# One named row per registered special token present in the synthetic tokenizer, plus one
# per filler special token; together they cover every special id exactly once.
_SPECIAL_TOKEN_ROWS: list[tuple[int, str]] = []
for _token in SpecialTokens:
    try:
        _rank = TEKKENIZER.get_special_token(_token.value)
    except ValueError:
        continue
    _SPECIAL_TOKEN_ROWS.append((_rank, _expected_special_str(_token.value)))
for _i in range(TEKKENIZER.num_special_tokens):
    _piece = TEKKENIZER.id_to_piece(_i)
    if re.fullmatch(r"<SPECIAL_\d+>", _piece):
        _SPECIAL_TOKEN_ROWS.append((_i, _piece))

_NON_SPECIAL_IDS = list(range(TEKKENIZER.num_special_tokens, TEKKENIZER.n_words))

# Registered special tokens (without filler specials) also get decode round-trip rows.
_DECODE_SPECIAL_ROWS = [
    (rank, expected) for rank, expected in _SPECIAL_TOKEN_ROWS if not re.fullmatch(r"<SPECIAL_\d+>", expected)
]


class TestMistralLLGTokenizer:
    def test_init_rejects_non_tekkenizer(self) -> None:
        mock_tokenizer = MagicMock(spec=Tokenizer)
        with pytest.raises(TypeError, match="Guidance only supports Tekken tokenizers"):
            MistralLLGTokenizer(mock_tokenizer)

    def test_eos_and_bos_ids(self) -> None:
        assert LLG_TOKENIZER.eos_token_id == TEKKENIZER.eos_id
        assert LLG_TOKENIZER.bos_token_id == TEKKENIZER.bos_id

    def test_tokens_length(self) -> None:
        assert len(LLG_TOKENIZER.tokens) == TEKKENIZER.n_words

    def test_special_token_ids_count(self) -> None:
        assert len(LLG_TOKENIZER.special_token_ids) == TEKKENIZER.num_special_tokens
        assert len(_SPECIAL_TOKEN_ROWS) == TEKKENIZER.num_special_tokens

    @pytest.mark.parametrize("token_id", list(LLG_TOKENIZER.special_token_ids), ids=lambda i: f"special_{i}")
    def test_all_special_tokens_are_angle_bracketed(self, token_id: int) -> None:
        token_bytes = LLG_TOKENIZER.tokens[token_id]
        token_str = token_bytes.decode("utf-8")
        assert re.fullmatch(r"<.*>", token_str), f"Special token at id={token_id} is not angle-bracketed: {token_str!r}"

    @pytest.mark.parametrize(
        ("rank", "expected"),
        _SPECIAL_TOKEN_ROWS,
        ids=[f"rank_{rank}_{expected}" for rank, expected in _SPECIAL_TOKEN_ROWS],
    )
    def test_special_token_conversion(self, rank: int, expected: str) -> None:
        assert LLG_TOKENIZER.tokens[rank] == expected.encode("utf-8"), (
            f"Token at rank={rank}: expected {expected!r}, got {LLG_TOKENIZER.tokens[rank]!r}"
        )

    @pytest.mark.parametrize("token_id", _NON_SPECIAL_IDS, ids=lambda i: f"token_{i}")
    def test_non_special_tokens_match_byte_pieces(self, token_id: int) -> None:
        expected = TEKKENIZER.id_to_byte_piece(token_id, SpecialTokenPolicy.RAISE)
        assert LLG_TOKENIZER.tokens[token_id] == expected, (
            f"Token at id={token_id}: expected {expected!r}, got {LLG_TOKENIZER.tokens[token_id]!r}"
        )

    def test_call_encodes_string(self) -> None:
        test_string = "abc"
        assert LLG_TOKENIZER(test_string) == TEKKENIZER.encode(test_string, bos=False, eos=False)

    def test_init_rejects_invalid_special_token_format(self) -> None:
        base = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
        next_rank = len(base)
        special_tokens: list[SpecialTokenInfo] = [
            *base,
            SpecialTokenInfo(rank=next_rank, token_str="INVALID_NO_BRACKETS", is_control=True),
        ]
        vocab = quick_vocab()
        num_special = len(special_tokens)
        tekkenizer = Tekkenizer(
            vocab,
            special_tokens=special_tokens,
            pattern=r".+",
            vocab_size=len(vocab) + num_special,
            num_special_tokens=num_special,
            version=TokenizerVersion.v7,
        )
        with pytest.raises(ValueError, match="Invalid special token"):
            MistralLLGTokenizer(tekkenizer)

    def test_init_rejects_duplicate_special_tokens(self) -> None:
        base = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
        next_rank = len(base)
        # Both [CUSTOM] and <CUSTOM> map to <CUSTOM> after bracket conversion
        special_tokens: list[SpecialTokenInfo] = [
            *base,
            SpecialTokenInfo(rank=next_rank, token_str="<CUSTOM>", is_control=True),
            SpecialTokenInfo(rank=next_rank + 1, token_str="[CUSTOM]", is_control=True),
        ]
        vocab = quick_vocab()
        num_special = len(special_tokens)
        tekkenizer = Tekkenizer(
            vocab,
            special_tokens=special_tokens,
            pattern=r".+",
            vocab_size=len(vocab) + num_special,
            num_special_tokens=num_special,
            version=TokenizerVersion.v7,
        )
        with pytest.raises(ValueError, match="Duplicate special token"):
            MistralLLGTokenizer(tekkenizer)


class TestFromMistralTokenizer:
    def test_properties(self) -> None:
        assert isinstance(LL_TOKENIZER, llg.LLTokenizer)
        assert LL_TOKENIZER.vocab_size == TEKKENIZER.n_words
        assert LL_TOKENIZER.eos_token == TEKKENIZER.eos_id
        for i in range(TEKKENIZER.num_special_tokens):
            assert LL_TOKENIZER.is_special_token(i), f"Token id={i} should be special"

    @pytest.mark.parametrize("test_string", ["abc", "hello", "de", "abcdefhello"])
    def test_tokenize_str_matches_tekkenizer(self, test_string: str) -> None:
        expected = TEKKENIZER.encode(test_string, bos=False, eos=False)
        result = LL_TOKENIZER.tokenize_str(test_string)
        assert result == expected, f"Mismatch for {test_string!r}: expected {expected}, got {result}"

    @pytest.mark.parametrize("test_string", ["abc", "hello", "de"])
    def test_decode_str_roundtrip(self, test_string: str) -> None:
        tokens = TEKKENIZER.encode(test_string, bos=False, eos=False)
        decoded = LL_TOKENIZER.decode_str(tokens)
        assert decoded == test_string, f"Roundtrip failed for {test_string!r}: got {decoded!r}"

    @pytest.mark.parametrize("test_string", ["abc", "hello"])
    def test_decode_bytes_roundtrip(self, test_string: str) -> None:
        tokens = TEKKENIZER.encode(test_string, bos=False, eos=False)
        decoded = LL_TOKENIZER.decode_bytes(tokens)
        assert decoded == test_string.encode("utf-8"), f"Roundtrip failed for {test_string!r}: got {decoded!r}"

    @pytest.mark.parametrize(
        ("rank", "expected"),
        _DECODE_SPECIAL_ROWS,
        ids=[f"rank_{rank}_{expected}" for rank, expected in _DECODE_SPECIAL_ROWS],
    )
    def test_decode_special_tokens(self, rank: int, expected: str) -> None:
        decoded_str = LL_TOKENIZER.decode_str([rank])
        assert decoded_str == expected, f"decode_str at id={rank}: expected {expected!r}, got {decoded_str!r}"
        decoded_bytes = LL_TOKENIZER.decode_bytes([rank])
        assert decoded_bytes == expected.encode("utf-8"), (
            f"decode_bytes at id={rank}: expected {expected.encode('utf-8')!r}, got {decoded_bytes!r}"
        )

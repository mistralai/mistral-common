from __future__ import annotations

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
from tests.test_tekken import get_special_tokens, quick_vocab

_NUM_SPECIAL_TOKENS = 100
_EXTRA_TOKENS = [b"a", b"b", b"c", b"f", b"de", b"he", b"llo"]


@pytest.fixture(scope="module")
def tekkenizer() -> Tekkenizer:
    special_tokens = get_special_tokens(TokenizerVersion.v7)
    return Tekkenizer(
        quick_vocab(_EXTRA_TOKENS),
        special_tokens=special_tokens,
        pattern=r".+",
        vocab_size=256 + _NUM_SPECIAL_TOKENS,
        num_special_tokens=_NUM_SPECIAL_TOKENS,
        version=TokenizerVersion.v7,
    )


@pytest.fixture(scope="module")
def llg_tokenizer(tekkenizer: Tekkenizer) -> MistralLLGTokenizer:
    return MistralLLGTokenizer(tekkenizer)


@pytest.fixture(scope="module")
def mistral_tokenizer(tekkenizer: Tekkenizer) -> MistralTokenizer:
    instruct_tokenizer = InstructTokenizerV7(tekkenizer)
    normalizer = get_normalizer(tekkenizer.version, tekkenizer.model_settings_builder)
    validator = get_validator(tekkenizer.version, mode=ValidationMode.test)
    return MistralTokenizer(instruct_tokenizer, validator=validator, request_normalizer=normalizer)


@pytest.fixture(scope="module")
def ll_tokenizer(mistral_tokenizer: MistralTokenizer) -> llg.LLTokenizer:
    return from_mistral_tokenizer(mistral_tokenizer)


class TestMistralLLGTokenizer:
    def test_init_rejects_non_tekkenizer(self) -> None:
        mock_tokenizer = MagicMock(spec=Tokenizer)
        with pytest.raises(TypeError, match="Guidance only supports Tekken tokenizers"):
            MistralLLGTokenizer(mock_tokenizer)

    def test_eos_and_bos_ids(self, tekkenizer: Tekkenizer, llg_tokenizer: MistralLLGTokenizer) -> None:
        assert llg_tokenizer.eos_token_id == tekkenizer.eos_id
        assert llg_tokenizer.bos_token_id == tekkenizer.bos_id

    def test_tokens_length(self, tekkenizer: Tekkenizer, llg_tokenizer: MistralLLGTokenizer) -> None:
        assert len(llg_tokenizer.tokens) == tekkenizer.n_words

    def test_special_token_ids_count(self, tekkenizer: Tekkenizer, llg_tokenizer: MistralLLGTokenizer) -> None:
        assert len(llg_tokenizer.special_token_ids) == tekkenizer.num_special_tokens

    def test_all_special_tokens_are_angle_bracketed(self, llg_tokenizer: MistralLLGTokenizer) -> None:
        for i in llg_tokenizer.special_token_ids:
            token_bytes = llg_tokenizer.tokens[i]
            token_str = token_bytes.decode("utf-8")
            assert re.fullmatch(r"<.*>", token_str), f"Special token at id={i} is not angle-bracketed: {token_str!r}"

    def test_special_token_conversion(self, tekkenizer: Tekkenizer, llg_tokenizer: MistralLLGTokenizer) -> None:
        checked = 0
        for token in SpecialTokens:
            try:
                rank = tekkenizer.get_special_token(token.value)
            except ValueError:
                continue

            llg_bytes = llg_tokenizer.tokens[rank]
            if re.fullmatch(r"\[.*\]", token.value):
                expected = token.value.replace("[", "<").replace("]", ">")
            else:
                expected = token.value

            assert llg_bytes == expected.encode("utf-8"), (
                f"Token {token.name} at rank={rank}: expected {expected!r}, got {llg_bytes!r}"
            )
            checked += 1

        # Filler tokens (<SPECIAL_XX>) should be preserved as-is
        for i in range(tekkenizer.num_special_tokens):
            piece = tekkenizer.id_to_piece(i)
            if re.fullmatch(r"<SPECIAL_\d+>", piece):
                assert llg_tokenizer.tokens[i] == piece.encode("utf-8"), (
                    f"Filler token at id={i}: expected {piece!r}, got {llg_tokenizer.tokens[i]!r}"
                )
                checked += 1

        assert checked == tekkenizer.num_special_tokens

    def test_non_special_tokens_match_byte_pieces(
        self, tekkenizer: Tekkenizer, llg_tokenizer: MistralLLGTokenizer
    ) -> None:
        for i in range(tekkenizer.num_special_tokens, tekkenizer.n_words):
            expected = tekkenizer.id_to_byte_piece(i, SpecialTokenPolicy.RAISE)
            assert llg_tokenizer.tokens[i] == expected, (
                f"Token at id={i}: expected {expected!r}, got {llg_tokenizer.tokens[i]!r}"
            )

    def test_call_encodes_string(self, tekkenizer: Tekkenizer, llg_tokenizer: MistralLLGTokenizer) -> None:
        test_string = "abc"
        assert llg_tokenizer(test_string) == tekkenizer.encode(test_string, bos=False, eos=False)

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
    def test_properties(self, tekkenizer: Tekkenizer, ll_tokenizer: llg.LLTokenizer) -> None:
        assert isinstance(ll_tokenizer, llg.LLTokenizer)
        assert ll_tokenizer.vocab_size == tekkenizer.n_words
        assert ll_tokenizer.eos_token == tekkenizer.eos_id
        for i in range(tekkenizer.num_special_tokens):
            assert ll_tokenizer.is_special_token(i), f"Token id={i} should be special"

    def test_tokenize_str_matches_tekkenizer(self, tekkenizer: Tekkenizer, ll_tokenizer: llg.LLTokenizer) -> None:
        test_strings = ["abc", "hello", "de", "abcdefhello"]
        for s in test_strings:
            expected = tekkenizer.encode(s, bos=False, eos=False)
            result = ll_tokenizer.tokenize_str(s)
            assert result == expected, f"Mismatch for {s!r}: expected {expected}, got {result}"

    def test_decode_str_roundtrip(self, tekkenizer: Tekkenizer, ll_tokenizer: llg.LLTokenizer) -> None:
        test_strings = ["abc", "hello", "de"]
        for s in test_strings:
            tokens = tekkenizer.encode(s, bos=False, eos=False)
            decoded = ll_tokenizer.decode_str(tokens)
            assert decoded == s, f"Roundtrip failed for {s!r}: got {decoded!r}"

    def test_decode_bytes_roundtrip(self, tekkenizer: Tekkenizer, ll_tokenizer: llg.LLTokenizer) -> None:
        test_strings = ["abc", "hello"]
        for s in test_strings:
            tokens = tekkenizer.encode(s, bos=False, eos=False)
            decoded = ll_tokenizer.decode_bytes(tokens)
            assert decoded == s.encode("utf-8"), f"Roundtrip failed for {s!r}: got {decoded!r}"

    def test_decode_special_tokens(self, tekkenizer: Tekkenizer, ll_tokenizer: llg.LLTokenizer) -> None:
        for token in SpecialTokens:
            try:
                rank = tekkenizer.get_special_token(token.value)
            except ValueError:
                continue

            if re.fullmatch(r"\[.*\]", token.value):
                expected = token.value.replace("[", "<").replace("]", ">")
            else:
                expected = token.value

            decoded_str = ll_tokenizer.decode_str([rank])
            assert decoded_str == expected, (
                f"decode_str for {token.name} (id={rank}): expected {expected!r}, got {decoded_str!r}"
            )
            decoded_bytes = ll_tokenizer.decode_bytes([rank])
            assert decoded_bytes == expected.encode("utf-8"), (
                f"decode_bytes for {token.name} (id={rank}): "
                f"expected {expected.encode('utf-8')!r}, got {decoded_bytes!r}"
            )

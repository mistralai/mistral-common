import re
from typing import Any

from mistral_common.imports import assert_llguidance_installed, is_llguidance_installed
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import is_tekkenizer

if is_llguidance_installed():
    import llguidance as llg


class MistralLLGTokenizer:
    r"""Wraps a Tekken tokenizer for use with llguidance."""

    @property
    def bos_token_id(self) -> int:
        r"""The beginning of string token id."""
        return self._tokenizer.bos_id

    @property
    def eos_token_id(self) -> int:
        r"""The end of string token id."""
        return self._tokenizer.eos_id

    @property
    def tokens(self) -> list[bytes]:
        r"""The list of token byte representations."""
        return self._tokens

    @property
    def special_token_ids(self) -> list[int]:
        r"""The list of special token ids."""
        return self._special_token_ids

    def __init__(self, tokenizer: Tokenizer) -> None:
        r"""Initialize the wrapper.

        Args:
            tokenizer: The Tekken tokenizer to wrap for llguidance compatibility.

        Raises:
            TypeError: If the tokenizer is not a Tekkenizer.
            ValueError: If a special token has an invalid format.
        """
        assert_llguidance_installed()

        if not is_tekkenizer(tokenizer):
            raise TypeError(f"Guidance only supports Tekken tokenizers, got {type(tokenizer)}")

        self._tokenizer = tokenizer
        self._tokens: list[bytes] = []
        self._special_token_ids: list[int] = []

        seen_special_tokens: set[str] = set()
        for i in range(self._tokenizer.n_words):
            # Convert square brackets to angle brackets for special tokens,
            # since llg only recognizes the latter.
            if i < self._tokenizer.num_special_tokens:
                token_rep = self._tokenizer.id_to_piece(i)
                if match := re.fullmatch(r"\[(.*)\]", token_rep):
                    token_rep_llg = f"<{match.group(1)}>"
                else:
                    token_rep_llg = token_rep

                if not re.fullmatch(r"<.*>", token_rep_llg):
                    raise ValueError(f"Invalid special token: {token_rep_llg} ({token_rep})")
                if token_rep_llg in seen_special_tokens:
                    raise ValueError(f"Duplicate special token: {token_rep_llg} (already seen: {seen_special_tokens})")
                seen_special_tokens.add(token_rep_llg)
                self._special_token_ids.append(i)
                self._tokens.append(token_rep_llg.encode("utf-8"))
            else:
                token_bytes = self._tokenizer.id_to_byte_piece(i, SpecialTokenPolicy.RAISE)
                self._tokens.append(token_bytes)

        if len(self._special_token_ids) != self._tokenizer.num_special_tokens:
            raise ValueError(
                f"Expected {self._tokenizer.num_special_tokens} special tokens, but found "
                f"{len(self._special_token_ids)}"
            )

    def __call__(self, s: str, *args: Any, **kwargs: Any) -> list[int]:
        r"""Tokenizes a string into token ids.

        Args:
            s: The string to tokenize.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            The list of token ids.
        """
        return self._tokenizer.encode(s, bos=False, eos=False)


def from_mistral_tokenizer(tokenizer: MistralTokenizer) -> "llg.LLTokenizer":
    r"""Creates an llguidance tokenizer from a Mistral tokenizer.

    Args:
        tokenizer: The Mistral tokenizer to convert. Must wrap a Tekkenizer.

    Returns:
        The llguidance tokenizer.

    Raises:
        TypeError: If the underlying tokenizer is not a Tekkenizer.
    """
    assert_llguidance_installed()
    inner_tokenizer = tokenizer.instruct_tokenizer.tokenizer
    tokenizer_data = MistralLLGTokenizer(inner_tokenizer)
    return llg.LLTokenizer(llg.TokenizerWrapper(tokenizer_data))

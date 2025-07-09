import logging
import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Set, Union

from sentencepiece import SentencePieceProcessor

from mistral_common.exceptions import TokenizerException
from mistral_common.tokens.tokenizers.base import (
    SpecialTokenPolicy,
    Tokenizer,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.image import ImageConfig, MultiModalVersion


def is_sentencepiece(path: Union[str, Path]) -> bool:
    r"""Check if the given path is a SentencePiece model."""
    if isinstance(path, str):
        path = Path(path)

    instruct_versions = list(TokenizerVersion.__members__)
    mm_versions = list(MultiModalVersion.__members__) + [""]  # allow no mm version
    suffixes = [f".model.{v}{m}" for v in instruct_versions for m in mm_versions] + [".model"]

    return path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes)


def get_spm_version(tokenizer_filename: Union[str, Path], raise_deprecated: bool = False) -> TokenizerVersion:
    r"""Get the version of the tokenizer from the filename."""
    tokenizer_filename = str(tokenizer_filename)

    _version_str = tokenizer_filename.split(".")[-1]
    if _version_str != "model":  # filter tokenizer_filename == "/path/to/tokenizer.model" case
        _version_str = _version_str.split("m")[0]

    if _version_str == "model":
        if raise_deprecated:
            raise TokenizerException(f"Make sure to rename your tokenizer file to end with {tokenizer_filename}.v1.")

        # tokenizer.model => tokenizer.model.v1
        return TokenizerVersion("v1")

    if _version_str not in TokenizerVersion.__members__:
        raise TokenizerException(f"Unrecognized tokenizer filename: {tokenizer_filename}")

    return TokenizerVersion(_version_str)


def get_image_config(tokenizer_filename: Union[str, Path]) -> Optional[ImageConfig]:
    r"""Get the image config from the tokenizer filename."""
    tokenizer_filename = str(tokenizer_filename)

    _version_str = tokenizer_filename.split(".")[-1]
    if _version_str == "model" or "m" not in _version_str:
        return None

    _mm_version_str = "m" + _version_str.split("m")[-1]

    if _mm_version_str not in MultiModalVersion.__members__:
        raise TokenizerException(f"Unrecognized tokenizer filename: {tokenizer_filename}")

    return MultiModalVersion(_mm_version_str).config


class SentencePieceTokenizer(Tokenizer):
    r"""[SentencePiece](https://github.com/google/sentencepiece) tokenizer."""

    def __init__(self, model_path: Union[str, Path], tokenizer_version: Optional[TokenizerVersion] = None) -> None:
        r"""Initialize the `SentencePieceTokenizer`.

        Args:
            model_path: The path to the `SentencePiece` model.
            tokenizer_version: The version of the tokenizer. If not provided, it will be inferred from the model path.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self._model = SentencePieceProcessor(
            model_file=model_path if isinstance(model_path, str) else model_path.as_posix()
        )

        assert self._model.vocab_size() == self._model.get_piece_size()
        self._vocab = [self._model.id_to_piece(i) for i in range(self.n_words)]

        self._version: TokenizerVersion = tokenizer_version or get_spm_version(model_path, raise_deprecated=False)

        self._file_path = Path(model_path)
        super().__init__()

    @property
    def file_path(self) -> Path:
        r"""The path to the tokenizer model."""
        return self._file_path

    @property
    def version(self) -> TokenizerVersion:
        r"""The version of the tokenizer."""
        return self._version

    def get_control_token(self, s: str) -> int:
        r"""Get the control token for the given string."""
        return self._model.piece_to_id(s)  # type: ignore

    @property
    def n_words(self) -> int:
        r"""Vocabulary size of the tokenizer."""
        return self._model.vocab_size()  # type: ignore

    def vocab(self) -> List[str]:
        r"""All tokens in the vocabulary as strings."""
        return self._vocab

    @cached_property
    def bos_id(self) -> int:
        r"""The beginning of sentence token id."""
        return self._model.bos_id()  # type: ignore

    @cached_property
    def eos_id(self) -> int:
        r"""The end of sentence token id."""
        return self._model.eos_id()  # type: ignore

    @cached_property
    def _control_tokens(self) -> Set[int]:
        return {tok for tok in range(self.n_words) if self._model.IsControl(tok)}

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        r"""Encode the given string into a list of token ids.

        Args:
            s: The string to encode.
            bos: Whether to add the beginning of sentence token.
            eos: Whether to add the end of sentence token.

        Returns:
            The list of token ids.
        """
        assert isinstance(s, str)
        t: List[int] = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        if eos:
            t = [*t, self.eos_id]
        return t

    def decode(self, tokens: List[int], special_token_policy: Optional[SpecialTokenPolicy] = None) -> str:
        r"""Decode the given list of token ids into a string.

        Note:
            Using `special_token_policy=SpecialTokenPolicy.KEEP` will keep the special tokens and the normal tokens as
            SentencePiece pieces.

        Args:
            tokens: The list of token ids.
            special_token_policy: The policy to use for special tokens. If `None`, the default policy
                is `SpecialTokenPolicy.IGNORE`.  Passing `None` is deprecated and will be changed
                to `SpecialTokenPolicy.IGNORE` in `mistral_common=1.10.0`.

        Returns:
            The decoded string.
        """
        if special_token_policy is not None and not isinstance(special_token_policy, SpecialTokenPolicy):
            raise ValueError(
                f"Expected `special_token_policy` to be None or SpecialTokenPolicy, got {type(special_token_policy)}."
            )

        if special_token_policy is None:
            warnings.warn(
                (
                    "Using the tokenizer's special token policy `None` is deprecated. "
                    "It will be removed in 1.10.0. "
                    "Please pass a special token policy explicitly. "
                    "Future default will be SpecialTokenPolicy.IGNORE."
                ),
                FutureWarning,
            )
            special_token_policy = SpecialTokenPolicy.IGNORE

        if special_token_policy in [SpecialTokenPolicy.KEEP, SpecialTokenPolicy.RAISE]:
            return self._decode_with_special_tokens(tokens, special_token_policy)

        return self._model.decode(tokens)  # type: ignore

    def id_to_piece(self, token_id: int) -> str:
        r"""Convert the given token id to a token piece."""
        return self._model.id_to_piece(token_id)  # type: ignore

    def _decode_with_special_tokens(self, tokens: List[int], special_token_policy: SpecialTokenPolicy) -> str:
        text = ""
        curr_tokens: List[int] = []
        for tok in tokens:
            if tok in self._control_tokens:
                if special_token_policy == SpecialTokenPolicy.RAISE:
                    raise ValueError("Decoding `tokens` that contain special tokens with special_token_policy=RAISE.")
                if curr_tokens:
                    text += "".join([self.id_to_piece(tok) for tok in curr_tokens])
                    curr_tokens = []

                text += self.id_to_piece(tok)

            else:
                curr_tokens.append(tok)

        if curr_tokens:
            text += "".join([self.id_to_piece(tok) for tok in curr_tokens])

        return text

    def to_string(self, tokens: List[int]) -> str:
        r"""[DEPRECATED] Converts a list of token ids into a string, keeping special tokens.

        Use `decode` with `special_token_policy=SpecialTokenPolicy.KEEP` instead.

        This is a convenient method for debugging.
        """
        warnings.warn(
            (
                "`to_string` is deprecated and will be removed in 1.10.0. "
                "Use `decode` with `special_token_policy=SpecialTokenPolicy.KEEP` instead."
            ),
            FutureWarning,
        )
        return self._to_string(tokens)

    def _to_string(self, tokens: List[int]) -> str:
        return self.decode(tokens, special_token_policy=SpecialTokenPolicy.KEEP)

    @property
    def pad_id(self) -> int:
        r"""The padding token id."""
        return self._model.pad_id()  # type: ignore

    @property
    def unk_id(self) -> int:
        r"""The unknown token id."""
        return self._model.unk_id()  # type: ignore

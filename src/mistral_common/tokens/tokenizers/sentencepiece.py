import logging
import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import Protocol, TypeGuard, cast

import numpy as np

from mistral_common.exceptions import TokenizerException
from mistral_common.imports import assert_sentencepiece_installed, is_sentencepiece_installed
from mistral_common.tokens.tokenizers.base import (
    SpecialTokenPolicy,
    Tokenizer,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.image import ImageConfig, MultiModalVersion
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder

warnings.filterwarnings(
    action="once",
    category=FutureWarning,
    message=r".*`get_control_token` is deprecated.*",
)
warnings.filterwarnings(
    action="once",
    category=FutureWarning,
    message=r".*`_control_tokens` is deprecated.*",
)


if is_sentencepiece_installed():
    from sentencepiece import SentencePieceProcessor


class _SentencePieceModel(Protocol):
    r"""Typed surface of the optional SentencePiece model used by this module."""

    def piece_to_id(self, piece: str) -> int:
        r"""Get the token ID of a piece string."""
        ...

    def vocab_size(self) -> int:
        r"""Get the vocabulary size."""
        ...

    def get_piece_size(self) -> int:
        r"""Get the total number of pieces."""
        ...

    def id_to_piece(self, piece_id: int) -> str:
        r"""Get the piece string of a token ID."""
        ...

    def IsControl(self, token: int) -> bool:
        r"""Check whether a token ID is a control token."""
        ...

    def encode(self, input: str) -> list[int]:
        r"""Encode a string into token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        r"""Decode token IDs into a string."""
        ...

    def bos_id(self) -> int:
        r"""Get the beginning-of-sentence token ID."""
        ...

    def eos_id(self) -> int:
        r"""Get the end-of-sentence token ID."""
        ...

    def pad_id(self) -> int:
        r"""Get the padding token ID."""
        ...

    def unk_id(self) -> int:
        r"""Get the unknown token ID."""
        ...


def is_sentencepiece(path: str | Path) -> bool:
    r"""Check if the given path is a SentencePiece model.

    Recognizes files ending in .model or .model.<version>[<mm-version>],
    e.g. tokenizer.model.v3 or tokenizer.model.v3m1.

    Args:
        path: The path to check.

    Returns:
        `True` if the path is an existing file with a known sentencepiece
        suffix, `False` otherwise.
    """
    if isinstance(path, str):
        path = Path(path)

    instruct_versions = list(TokenizerVersion.__members__)
    mm_versions = list(MultiModalVersion.__members__) + [""]  # allow no mm version
    suffixes = [f".model.{v}{m}" for v in instruct_versions for m in mm_versions] + [".model"]

    return path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes)


def get_spm_version(tokenizer_filename: str | Path, raise_deprecated: bool = False) -> TokenizerVersion:
    r"""Get the version of the tokenizer from the filename.

    Expects filenames like "tokenizer.model.v3" or "tokenizer.model.v7m1".
    A bare `tokenizer.model` is treated as v1.

    Args:
        tokenizer_filename: The tokenizer model path to parse.
        raise_deprecated: If `True`, raise for the deprecated unversioned
            `tokenizer.model` filename instead of silently returning v1.

    Returns:
        The TokenizerVersion parsed from the filename.

    Raises:
        TokenizerException: If `raise_deprecated` is `True` and the filename has
            no version suffix, or the version is unrecognized.
    """
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


def get_image_config(tokenizer_filename: str | Path) -> ImageConfig | None:
    r"""Get the image config from the tokenizer filename.

    The multimodal version suffix (e.g., "m1" in "tokenizer.model.v7m1")
    selects the image configuration.

    Args:
        tokenizer_filename: The tokenizer model path to parse.

    Returns:
        The ImageConfig for the multimodal version in the filename, or `None`
        if the filename carries no multimodal version.

    Raises:
        TokenizerException: If the multimodal version in the filename is
            unrecognized.
    """
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

    def __init__(self, model_path: str | Path, tokenizer_version: TokenizerVersion | None = None) -> None:
        r"""Initialize the SentencePieceTokenizer.

        Args:
            model_path: The path to the SentencePiece model file.
            tokenizer_version: The version of the tokenizer. If `None`, inferred
                from the model path filename.

        Raises:
            AssertionError: If the model file does not exist or its internal
                vocab size is inconsistent.
            TokenizerException: If the version inferred from the filename is
                unrecognized.
        """
        assert_sentencepiece_installed()

        self._logger = logging.getLogger(self.__class__.__name__)
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        # SentencePiece is optional and conditionally imported, so narrow its dynamic result to this local protocol.
        self._model: _SentencePieceModel = cast(
            _SentencePieceModel,
            SentencePieceProcessor(model_file=model_path if isinstance(model_path, str) else model_path.as_posix()),
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

    @property
    def model_settings_builder(self) -> ModelSettingsBuilder | None:
        r"""Always returns `None` as SentencePiece does not support `model_settings_builder`."""
        if self.version.supports_model_settings:
            raise ValueError(f"SentencePieceTokenizer does not support model settings for version {self.version}")
        return None

    def get_special_token(self, s: str) -> int:
        r"""Get the token ID for a special token string.

        Args:
            s: The special token string (e.g., "<s>").

        Returns:
            The token ID for the special token. Unknown strings map to the
            unknown token ID (sentencepiece behavior).
        """
        return self._model.piece_to_id(s)

    def get_control_token(self, s: str) -> int:
        r"""Get the token ID of a control token. Deprecated: use `get_special_token()` instead.

        Args:
            s: The special token string.

        Returns:
            The token ID for the special token.
        """
        warnings.warn("`get_control_token` is deprecated. Use `get_special_token` instead.", FutureWarning)
        return self.get_special_token(s)

    @property
    def n_words(self) -> int:
        r"""Vocabulary size of the tokenizer."""
        return self._model.vocab_size()

    @property
    def num_special_tokens(self) -> int:
        r"""The number of special tokens of the tokenizer."""
        return len(self.special_ids)

    def vocab(self) -> list[str]:
        r"""Get all tokens in the vocabulary as strings."""
        return self._vocab

    @cached_property
    def bos_id(self) -> int:
        r"""The beginning of sentence token id."""
        return self._model.bos_id()

    @cached_property
    def eos_id(self) -> int:
        r"""The end of sentence token id."""
        return self._model.eos_id()

    def is_special(self, token: int | np.integer | str) -> bool:
        r"""Check if a token is a special (control) token.

        Args:
            token: Token ID (int or numpy integer) or token string to check.

        Returns:
            `True` if the token is a sentencepiece control token, `False` otherwise.

        Raises:
            TypeError: If token is not an int, numpy integer, or str.
        """
        if isinstance(token, (int, np.integer)):
            return self._model.IsControl(int(token))
        elif isinstance(token, str):
            token_int = self._model.piece_to_id(token)
            return self._model.IsControl(token_int)
        else:
            raise TypeError(f"Expected int or str, got {type(token).__name__}")

    @cached_property
    def _control_tokens(self) -> set[int]:
        warnings.warn(
            "`_control_tokens` is deprecated. Make use of `is_special` or `special_ids` instead.", FutureWarning
        )
        return self.special_ids

    @cached_property
    def special_ids(self) -> set[int]:
        r"""Ids of the special tokens."""
        return {tok for tok in range(self.n_words) if self._model.IsControl(tok)}

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        r"""Encode the given string into a list of token ids.

        Args:
            s: The string to encode.
            bos: Whether to add the beginning of sentence token.
            eos: Whether to add the end of sentence token.

        Returns:
            The list of token ids.
        """
        assert isinstance(s, str)
        t: list[int] = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        if eos:
            t = [*t, self.eos_id]
        return t

    def decode(self, tokens: list[int], special_token_policy: SpecialTokenPolicy = SpecialTokenPolicy.IGNORE) -> str:
        r"""Decode the given list of token ids into a string.

        Note:
            Using `special_token_policy=SpecialTokenPolicy.KEEP` will keep the special tokens and the normal tokens as
            SentencePiece pieces.

        Args:
            tokens: The list of token ids.
            special_token_policy: The policy to use for special tokens.

        Returns:
            The decoded string.
        """
        try:
            special_token_policy = SpecialTokenPolicy(special_token_policy)
        except ValueError as e:
            valid = ", ".join(repr(p.value) for p in SpecialTokenPolicy)
            raise ValueError(
                f"Invalid `special_token_policy` {special_token_policy!r}. Expected one of: {valid}."
            ) from e

        if special_token_policy == SpecialTokenPolicy.IGNORE:
            decoded = self._model.decode(tokens)
            assert isinstance(decoded, str), f"Sentencepiece model decoded a {type(decoded)}, not a string."
            return decoded

        return self._decode_with_special_tokens(tokens, special_token_policy)

    def id_to_piece(self, token_id: int) -> str:
        r"""Convert the given token id to a token piece."""
        return self._model.id_to_piece(token_id)

    def _decode_with_special_tokens(self, tokens: list[int], special_token_policy: SpecialTokenPolicy) -> str:
        text_list = []
        curr_tokens: list[int] = []
        for tok in tokens:
            if self.is_special(tok):
                if special_token_policy == SpecialTokenPolicy.RAISE:
                    raise ValueError("Decoding `tokens` that contain special tokens with special_token_policy=RAISE.")
                if curr_tokens:
                    text_list.extend([self.id_to_piece(t) for t in curr_tokens])
                    curr_tokens = []

                text_list.append(self.id_to_piece(tok))

            else:
                curr_tokens.append(tok)

        if curr_tokens:
            if special_token_policy == SpecialTokenPolicy.RAISE:
                text_list.append(self._model.decode(curr_tokens))
            else:
                text_list.extend([self.id_to_piece(t) for t in curr_tokens])

        return "".join(text_list)

    def _to_string(self, tokens: list[int]) -> str:
        return self.decode(tokens, special_token_policy=SpecialTokenPolicy.KEEP)

    @property
    def pad_id(self) -> int:
        r"""The padding token id."""
        return self._model.pad_id()

    @property
    def unk_id(self) -> int:
        r"""The unknown token id."""
        return self._model.unk_id()


def is_sentencepiece_tokenizer(tokenizer: Tokenizer) -> TypeGuard[SentencePieceTokenizer]:
    r"""Return whether the tokenizer is a SentencePieceTokenizer."""
    return isinstance(tokenizer, SentencePieceTokenizer)

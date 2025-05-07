import logging
import os
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Set, Union

from sentencepiece import SentencePieceProcessor

from mistral_common.exceptions import TokenizerException
from mistral_common.tokens.tokenizers.base import (
    Tokenizer,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.multimodal import MultimodalConfig, MultiModalVersion


def is_sentencepiece(path: Union[str, Path]) -> bool:
    if isinstance(path, str):
        path = Path(path)

    instruct_versions = list(TokenizerVersion.__members__)
    mm_versions = list(MultiModalVersion.__members__) + [""]  # allow no mm version
    suffixes = [f".model.{v}{m}" for v in instruct_versions for m in mm_versions] + [".model"]

    return path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes)


def get_spm_version(tokenizer_filename: str, raise_deprecated: bool = False) -> TokenizerVersion:
    _version_str = tokenizer_filename.split(".")[-1].split("m")[0]
    if _version_str == "model":
        if raise_deprecated:
            raise TokenizerException(f"Make sure to rename your tokenizer file to end with {tokenizer_filename}.v1.")

        # tokenizer.model => tokenizer.model.v1
        return TokenizerVersion("v1")

    if _version_str not in TokenizerVersion.__members__:
        raise TokenizerException(f"Unrecognized tokenizer filename: {tokenizer_filename}")

    return TokenizerVersion(_version_str)


def get_mm_config(tokenizer_filename: str) -> Optional[MultimodalConfig]:
    _version_str = tokenizer_filename.split(".")[-1]
    if "m" not in _version_str:
        return None

    _mm_version_str = "m" + _version_str.split("m")[-1]

    if _mm_version_str not in MultiModalVersion.__members__:
        raise TokenizerException(f"Unrecognized tokenizer filename: {tokenizer_filename}")

    return MultiModalVersion(_mm_version_str).config


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str, tokenizer_version: Optional[TokenizerVersion] = None) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self._model = SentencePieceProcessor(model_file=model_path)

        assert self._model.vocab_size() == self._model.get_piece_size()
        self._vocab = [self._model.id_to_piece(i) for i in range(self.n_words)]

        self._version: TokenizerVersion = tokenizer_version or get_spm_version(model_path, raise_deprecated=False)

        super().__init__()

    @property
    def version(self) -> TokenizerVersion:
        return self._version

    def get_control_token(self, s: str) -> int:
        return self._model.piece_to_id(s)  # type: ignore

    @property
    def n_words(self) -> int:
        return self._model.vocab_size()  # type: ignore

    def vocab(self) -> List[str]:
        return self._vocab

    @property
    def bos_id(self) -> int:
        return self._model.bos_id()  # type: ignore

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()  # type: ignore

    @cached_property
    def _control_tokens(self) -> Set[int]:
        return {tok for tok in range(self.n_words) if self._model.IsControl(tok)}

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert isinstance(s, str)
        t: List[int] = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        if eos:
            t = [*t, self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)  # type: ignore

    def id_to_piece(self, token_id: int) -> str:
        return self._model.id_to_piece(token_id)  # type: ignore

    def to_string(self, tokens: List[int]) -> str:
        """
        Converts tokens into a string for debugging purposes
        """
        text = ""
        curr_tokens: List[int] = []
        for tok in tokens:
            if tok in self._control_tokens:
                if curr_tokens:
                    text += "".join([self.id_to_piece(tok) for tok in curr_tokens])
                    curr_tokens = []

                text += self.id_to_piece(tok)

            else:
                curr_tokens.append(tok)

        if curr_tokens:
            text += "".join([self.id_to_piece(tok) for tok in curr_tokens])

        return text

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()  # type: ignore

    @property
    def unk_id(self) -> int:
        return self._model.unk_id()  # type: ignore

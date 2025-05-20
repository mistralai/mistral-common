import base64
import json
import logging
import warnings
from enum import Enum
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Type, TypedDict, Union

import tiktoken

from mistral_common.tokens.tokenizers.base import (
    SpecialTokens,
    Tokenizer,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.multimodal import MultimodalConfig

logger = logging.getLogger(__name__)


def is_tekken(path: Union[str, Path]) -> bool:
    if isinstance(path, str):
        path = Path(path)
    return path.is_file() and "tekken" in path.name and path.suffix == ".json"


# Formatting specification of the JSON file
class TokenInfo(TypedDict):
    rank: int
    token_bytes: str  # base64 encoded
    token_str: Optional[str]


class SpecialTokenInfo(TypedDict):
    rank: int
    token_str: str
    is_control: bool


class TekkenConfig(TypedDict):
    pattern: str
    num_vocab_tokens: int
    default_vocab_size: int
    default_num_special_tokens: int
    version: str


class ModelData(TypedDict):
    vocab: List[TokenInfo]
    config: TekkenConfig
    version: int
    type: str
    multimodal: MultimodalConfig


class SpecialTokenPolicy(Enum):
    """What to do with special tokens when encoding/decoding."""

    IGNORE = 0
    KEEP = 1
    RAISE = 2


class Tekkenizer(Tokenizer):
    DEPRECATED_SPECIAL_TOKENS = (
        SpecialTokenInfo(rank=0, token_str=SpecialTokens.unk, is_control=True),
        SpecialTokenInfo(rank=1, token_str=SpecialTokens.bos, is_control=True),
        SpecialTokenInfo(rank=2, token_str=SpecialTokens.eos, is_control=True),
        SpecialTokenInfo(rank=3, token_str=SpecialTokens.begin_inst, is_control=True),
        SpecialTokenInfo(rank=4, token_str=SpecialTokens.end_inst, is_control=True),
        SpecialTokenInfo(rank=5, token_str=SpecialTokens.begin_tools, is_control=True),
        SpecialTokenInfo(rank=6, token_str=SpecialTokens.end_tools, is_control=True),
        SpecialTokenInfo(rank=7, token_str=SpecialTokens.begin_tool_results, is_control=True),
        SpecialTokenInfo(rank=8, token_str=SpecialTokens.end_tool_results, is_control=True),
        SpecialTokenInfo(rank=9, token_str=SpecialTokens.tool_calls, is_control=True),
        SpecialTokenInfo(rank=10, token_str=SpecialTokens.img, is_control=True),
        SpecialTokenInfo(rank=11, token_str=SpecialTokens.pad, is_control=True),
        SpecialTokenInfo(rank=12, token_str=SpecialTokens.img_break, is_control=True),
        SpecialTokenInfo(rank=13, token_str=SpecialTokens.img_end, is_control=True),
        SpecialTokenInfo(rank=14, token_str=SpecialTokens.prefix, is_control=True),
        SpecialTokenInfo(rank=15, token_str=SpecialTokens.middle, is_control=True),
        SpecialTokenInfo(rank=16, token_str=SpecialTokens.suffix, is_control=True),
        SpecialTokenInfo(rank=17, token_str=SpecialTokens.begin_system, is_control=True),
        SpecialTokenInfo(rank=18, token_str=SpecialTokens.end_system, is_control=True),
        SpecialTokenInfo(rank=19, token_str=SpecialTokens.begin_tool_content, is_control=True),
    )

    SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"

    # # note that params has a vocab_size field, but it's not used

    def __init__(
        self,
        vocab: List[TokenInfo],
        special_tokens: List[SpecialTokenInfo],
        pattern: str,
        vocab_size: int,
        num_special_tokens: int,
        version: TokenizerVersion,
        *,
        name: str = "tekkenizer",
        _path: Optional[str] = None,
        mm_config: Optional[MultimodalConfig] = None,
    ):
        assert vocab_size <= len(vocab) + num_special_tokens, (
            vocab_size,
            len(vocab),
            num_special_tokens,
        )
        self._vocab_size = vocab_size

        # The number of special tokens defined in the tokenizer json
        num_defined_special_tokens = len(set([t["token_str"] for t in special_tokens]))

        assert len(special_tokens) == num_defined_special_tokens, f"Special tokens must be unique: {special_tokens}"
        assert len(special_tokens) <= num_special_tokens

        special_filler = [
            SpecialTokenInfo(rank=i, token_str=self.SPECIAL_TOKEN_TEMPLATE.format(id=i), is_control=True)
            for i in range(len(special_tokens), num_special_tokens)
        ]
        if special_filler:
            logger.info(
                f"Adding special tokens {special_filler[0]['token_str']}, ..., {special_filler[-1]['token_str']}"
            )
        special_tokens = special_tokens + special_filler

        assert (
            len(set([t["token_str"] for t in special_tokens])) == len(special_tokens) == num_special_tokens
        ), special_tokens
        inner_vocab_size = vocab_size - num_special_tokens

        # reload vocab
        self._tekken_token2id_nospecial = _reload_mergeable_ranks(vocab, max_vocab=inner_vocab_size)
        assert set(range(inner_vocab_size)) == set(self._tekken_token2id_nospecial.values()), (
            inner_vocab_size,
            self._tekken_token2id_nospecial,
        )
        self._model = tiktoken.Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=self._tekken_token2id_nospecial,
            special_tokens={},  # special tokens are handled manually
        )

        self._version = version
        self._mm_config = mm_config
        self._all_special_tokens = special_tokens
        self._special_tokens_reverse_vocab = {t["token_str"]: t["rank"] for t in special_tokens}
        self._vocab = [self.id_to_piece(i) for i in range(vocab_size)]
        self._special_token_policy = SpecialTokenPolicy.IGNORE

    @classmethod
    def from_file(cls: Type["Tekkenizer"], path: Union[str, Path]) -> "Tekkenizer":
        if isinstance(path, str):
            path = Path(path)
        assert path.exists(), path
        with open(path, "r") as f:
            untyped = json.load(f)

        _version_str = untyped["config"].get("version")
        if _version_str not in TokenizerVersion.__members__:
            raise ValueError(
                f"Unknown version: {_version_str} in {path}. "
                f"Make sure to use a valid version string: {list(TokenizerVersion.__members__)}"
            )

        assert _version_str is not None
        version = TokenizerVersion(_version_str)

        special_tokens_dicts: Optional[List[SpecialTokenInfo]] = untyped.get("special_tokens", None)
        if special_tokens_dicts is None:
            err_msg = (
                f"Special tokens not found in {path} and default to {Tekkenizer.DEPRECATED_SPECIAL_TOKENS}. "
                "This behavior will be deprecated going forward. "
                "Please update your tokenizer file and include all special tokens you need."
            )
            # Tokenizer > v7 should find special tokens in the tokenizer file
            if version > TokenizerVersion("v7"):
                raise ValueError(err_msg)
            else:
                warnings.warn(
                    err_msg,
                    FutureWarning,
                )
                special_tokens = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
        else:
            special_tokens = [token for token in special_tokens_dicts]

        untyped["special_tokens"] = special_tokens

        if mm := untyped.get("multimodal", None):
            untyped["multimodal"] = MultimodalConfig(**mm)

        model_data: ModelData = untyped

        return cls(
            vocab=model_data["vocab"],
            special_tokens=special_tokens,
            pattern=model_data["config"]["pattern"],
            vocab_size=model_data["config"]["default_vocab_size"],
            num_special_tokens=model_data["config"]["default_num_special_tokens"],
            version=version,
            name=path.name.replace(".json", ""),
            mm_config=model_data.get("multimodal"),
        )

    @property
    def multimodal(self) -> Optional[MultimodalConfig]:
        return self._mm_config

    @multimodal.setter
    def multimodal(self, value: MultimodalConfig) -> None:
        raise ValueError("Can only set Multimodal config at init")

    @property
    def num_special_tokens(self) -> int:
        return len(self._all_special_tokens)

    @property
    def n_words(self) -> int:
        return self._vocab_size

    @property
    def version(self) -> TokenizerVersion:
        return self._version

    @property
    def special_token_policy(self) -> SpecialTokenPolicy:
        return self._special_token_policy

    @special_token_policy.setter
    def special_token_policy(self, policy: SpecialTokenPolicy) -> None:
        self._special_token_policy = policy

    @cached_property
    def bos_id(self) -> int:
        return self.get_control_token("<s>")

    @cached_property
    def eos_id(self) -> int:
        return self.get_control_token("</s>")

    @cached_property
    def pad_id(self) -> int:
        return self.get_control_token("<pad>")

    @cached_property
    def unk_id(self) -> int:
        return self.get_control_token("<unk>")

    def vocab(self) -> List[str]:
        # when returning self._vocab this will collapse
        # all tokens for which we have a decoding error into
        # the <?> string. This is bad and results in things
        # like len(set(vocab)) != len(vocab))
        # be careful when using self._vocab
        return self._vocab

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        tokens: List[int] = self._model.encode(s)
        tokens = [t + self.num_special_tokens for t in tokens]
        if bos:
            tokens = [self.bos_id, *tokens]
        if eos:
            tokens = [*tokens, self.eos_id]
        return tokens

    def _decode_all(self, tokens: List[int], special_token_policy: SpecialTokenPolicy) -> List[str]:
        # Lump special and non-special tokens together to minimize calls to decode
        decoded: List[str] = []
        for is_special, group in groupby(tokens, lambda t: t < self.num_special_tokens):
            if is_special:
                if special_token_policy == SpecialTokenPolicy.RAISE:
                    raise ValueError(
                        f"Decoding `tokens` that contain special tokens ({list(group)}) is not allowed. \n"
                        "Either make sure `tokens` do not include any special tokens or, "
                        "if you want to decode `tokens` that includes special tokens, "
                        "change the tokenizer's special token policy to IGNORE or KEEP: \n"
                        "```\nfrom mistral_common.tokens.tokenizers.mistral import MistralTokenizer"
                        "\nfrom mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy"
                        "\n\ntokenizer = MistralTokenizer.v3(is_tekken=True)"
                        "\ntekken = tokenizer.instruct_tokenizer.tokenizer"
                        "\ntekken.special_token_policy = SpecialTokenPolicy.IGNORE  # or SpecialTokenPolicy.KEEP"
                        "\n```"
                    )
                elif special_token_policy == SpecialTokenPolicy.KEEP:
                    decoded.extend(self._all_special_tokens[t]["token_str"] for t in group)
                elif special_token_policy == SpecialTokenPolicy.IGNORE:
                    continue
                # TODO: Could use "tokens_str" from vocab.json
                # but need to handle null cases.
            else:
                decoded.append(self._model.decode([t - self.num_special_tokens for t in group]))
        return decoded

    def is_byte(self, token_id: int) -> bool:
        return 0 <= token_id - self.num_special_tokens < 256

    def get_control_token(self, s: str) -> int:
        if s in self._special_tokens_reverse_vocab:
            return self._special_tokens_reverse_vocab[s]
        else:
            raise ValueError(f"Unknown control token {s}")

    def decode(self, tokens: List[int]) -> str:
        return "".join(self._decode_all(tokens, special_token_policy=self._special_token_policy))

    def to_string(self, tokens: List[int]) -> str:
        return "".join(self._decode_all(tokens, special_token_policy=SpecialTokenPolicy.KEEP))

    def id_to_piece(self, token_id: int) -> str:
        """convert a token id to its string representation."""
        return self._decode_all([token_id], special_token_policy=SpecialTokenPolicy.KEEP)[0]

    def id_to_byte_piece(self, token_id: int) -> bytes:
        """convert a token id to its byte representation."""
        if token_id < self.num_special_tokens:
            if self._special_token_policy == SpecialTokenPolicy.KEEP:
                return self._all_special_tokens[token_id]["token_str"].encode("utf-8")
            elif self._special_token_policy == SpecialTokenPolicy.RAISE:
                raise ValueError(f"{token_id} is a special token")

        return self._model.decode_single_token_bytes(token_id - self.num_special_tokens)


def _reload_mergeable_ranks(
    vocab: List[TokenInfo],
    max_vocab: Union[int, None] = None,
) -> Dict[bytes, int]:
    """
    Reload our tokenizer JSON file and convert it to Tiktoken format.
    """
    logger.info(f"Vocab size: {len(vocab)}")
    if max_vocab is not None:
        assert len(vocab) >= max_vocab, (len(vocab), max_vocab)
        vocab = vocab[:max_vocab]
        logger.info(f"Cutting vocab to first {len(vocab)} tokens.")

    # build ranks
    ranks: Dict[bytes, int] = {}
    for i, x in enumerate(vocab):
        assert x.keys() == {"rank", "token_bytes", "token_str"}
        assert x["rank"] == i
        merge = base64.b64decode(x["token_bytes"])
        assert i >= 256 or merge == bytes([i]), (i, merge)
        ranks[merge] = x["rank"]

    # sanity check
    assert len(ranks) == len(vocab)
    assert set(ranks.values()) == set(range(len(ranks)))

    return ranks

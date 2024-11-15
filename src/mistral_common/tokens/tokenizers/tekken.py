import base64
import json
import logging
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
    SPECIAL_TOKENS = (
        "<unk>",
        SpecialTokens.bos,
        SpecialTokens.eos,
        SpecialTokens.begin_inst,
        SpecialTokens.end_inst,
        SpecialTokens.begin_tools,
        SpecialTokens.end_tools,
        SpecialTokens.begin_tool_results,
        SpecialTokens.end_tool_results,
        SpecialTokens.tool_calls,
        SpecialTokens.img,
        "<pad>",
        SpecialTokens.img_break,
        SpecialTokens.img_end,
        SpecialTokens.prefix,
        SpecialTokens.middle,
        SpecialTokens.suffix,
        SpecialTokens.begin_system,
        SpecialTokens.end_system,
        SpecialTokens.begin_tool_content,
    )
    SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"

    # # note that params has a vocab_size field, but it's not used

    def __init__(
        self,
        vocab: List[TokenInfo],
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
        self._path = _path

        special_tokens = list(self.SPECIAL_TOKENS)
        assert len(special_tokens) == len(set(special_tokens)), f"Special tokens must be unique: {special_tokens}"
        assert len(special_tokens) < num_special_tokens

        special_filler = [
            self.SPECIAL_TOKEN_TEMPLATE.format(id=i) for i in range(len(special_tokens), num_special_tokens)
        ]
        if special_filler:
            logger.info(f"Adding special tokens {special_filler[0]}, ..., {special_filler[-1]}")
        special_tokens = special_tokens + special_filler
        assert len(set(special_tokens)) == len(special_tokens) == num_special_tokens, special_tokens
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
        self._all_special_tokens = special_tokens
        self._vocab = [self.id_to_piece(i) for i in range(vocab_size)]
        self._version = version
        self._special_token_policy = SpecialTokenPolicy.RAISE
        self._mm_config = mm_config

    @classmethod
    def from_file(cls: Type["Tekkenizer"], path: Union[str, Path]) -> "Tekkenizer":
        if isinstance(path, str):
            path = Path(path)
        assert path.exists()
        with open(path, "r") as f:
            untyped = json.load(f)
            if mm := untyped.get("multimodal", None):
                untyped["multimodal"] = MultimodalConfig(**mm)
            model_data: ModelData = untyped

        _version_str = model_data["config"].get("version")
        if _version_str not in TokenizerVersion.__members__:
            raise ValueError(
                f"Unknown version: {_version_str} in {path}. "
                f"Make sure to use a valid version string: {list(TokenizerVersion.__members__)}"
            )

        return cls(
            vocab=model_data["vocab"],
            pattern=model_data["config"]["pattern"],
            vocab_size=model_data["config"]["default_vocab_size"],
            num_special_tokens=model_data["config"]["default_num_special_tokens"],
            version=TokenizerVersion(_version_str),
            name=path.name.replace(".json", ""),
            _path=str(path),
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
        return self.SPECIAL_TOKENS.index("<s>")

    @cached_property
    def eos_id(self) -> int:
        return self.SPECIAL_TOKENS.index("</s>")

    @cached_property
    def pad_id(self) -> int:
        return self.SPECIAL_TOKENS.index("<pad>")

    @cached_property
    def unk_id(self) -> int:
        return self.SPECIAL_TOKENS.index("<unk>")

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
                    decoded.extend(self._all_special_tokens[t] for t in group)
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
        try:
            return self._all_special_tokens.index(s)
        except ValueError:
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
                return self._all_special_tokens[token_id].encode("utf-8")
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

import base64
import json
import logging
import warnings
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Type, TypedDict, Union

import tiktoken

from mistral_common.tokens.tokenizers.base import (
    SpecialTokenPolicy,
    SpecialTokens,
    Tokenizer,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.image import ImageConfig

logger = logging.getLogger(__name__)


def is_tekken(path: Union[str, Path]) -> bool:
    r"""Check if the given path is a tekken tokenizer file."""
    if isinstance(path, str):
        path = Path(path)
    return path.is_file() and "tekken" in path.name and path.suffix == ".json"


# Formatting specification of the JSON file
class TokenInfo(TypedDict):
    r"""Token information in the JSON file.

    Attributes:
        rank: The rank of the token.
        token_bytes: The token in bytes, base64 encoded.
        token_str: The token in string format.
    """

    rank: int
    token_bytes: str  # base64 encoded
    token_str: Optional[str]


class SpecialTokenInfo(TypedDict):
    r"""Special token information in the JSON file.

    Attributes:
        rank: The rank of the token.
        token_str: The token in string format.
        is_control: Whether the token is a control token.
    """

    rank: int
    token_str: str
    is_control: bool


class TekkenConfig(TypedDict):
    r"""Tekken configuration in the JSON file.

    Attributes:
        pattern: The pattern of the tokenizer.
        num_vocab_tokens: The number of vocabulary tokens.
        default_vocab_size: The default vocabulary size.
        default_num_special_tokens: The default number of special tokens.
        version: The version of the tokenizer.
    """

    pattern: str
    num_vocab_tokens: int
    default_vocab_size: int
    default_num_special_tokens: int
    version: str


class ModelData(TypedDict):
    r"""The data of the tekken tokenizer model.

    Attributes:
        vocab: The vocabulary of the tokenizer.
        config: The configuration of the tokenizer.
        version: The version of the tokenizer.
        type: The type of the tokenizer.
        image: The image configuration of the tokenizer.
    """

    vocab: List[TokenInfo]
    special_tokens: Optional[List[SpecialTokenInfo]]
    config: TekkenConfig
    version: int
    type: str
    image: ImageConfig


class Tekkenizer(Tokenizer):
    r"""Tekken tokenizer.

    This tokenizer is based on the [tiktoken](https://github.com/openai/tiktoken) library. It fastens the tokenization
    for multiple languages.
    """

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
        _path: Optional[Union[str, Path]] = None,
        image_config: Optional[ImageConfig] = None,
    ):
        r"""Initialize the tekken tokenizer.

        Args:
            vocab: The vocabulary of the tokenizer.
            special_tokens: The special tokens of the tokenizer.
            pattern: The pattern of the tokenizer.
            vocab_size: The vocabulary size of the tokenizer.
            num_special_tokens: The number of special tokens of the tokenizer.
            version: The version of the tokenizer.
            name: The name of the tokenizer.
            image_config: The image configuration of the tokenizer.
        """
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

        assert len(set([t["token_str"] for t in special_tokens])) == len(special_tokens) == num_special_tokens, (
            special_tokens
        )
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
        self._image_config = image_config
        self._all_special_tokens = special_tokens
        self._special_tokens_reverse_vocab = {t["token_str"]: t["rank"] for t in special_tokens}
        self._vocab = [self.id_to_piece(i) for i in range(vocab_size)]
        self._special_token_policy = SpecialTokenPolicy.IGNORE
        self._file_path = Path(_path) if _path is not None else None

    @property
    def file_path(self) -> Path:
        r"""The path to the tokenizer file."""
        if self._file_path is None:
            raise ValueError("The tokenizer was not loaded from a file.")
        return self._file_path

    @classmethod
    def from_file(cls: Type["Tekkenizer"], path: Union[str, Path]) -> "Tekkenizer":
        r"""Load the tekken tokenizer from a file.

        Args:
            path: The path to the tokenizer file.

        Returns:
            The tekken tokenizer.
        """
        if isinstance(path, str):
            path = Path(path)
        assert path.exists(), path
        with open(path, "r", encoding="utf-8") as f:
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
            # Tokenizer > v7 should find special tokens in the tokenizer file
            if version > TokenizerVersion("v7"):
                raise ValueError(
                    f"Special tokens not found in {path}. "
                    "Please update your tokenizer file and include all special tokens you need."
                )
            else:
                special_tokens = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
        else:
            special_tokens = [token for token in special_tokens_dicts]

        untyped["special_tokens"] = special_tokens

        if mm := untyped.get("multimodal", None):
            # deprecated - only allowed for tokenizers <= v11
            if version > TokenizerVersion("v11"):
                raise ValueError(
                    f"The image config has to be called 'image' in {path} for tokenizers of version {version.value}."
                )

            untyped["image"] = ImageConfig(**mm)
        elif image := untyped.get("image", None):
            untyped["image"] = ImageConfig(**image)

        model_data: ModelData = untyped

        return cls(
            vocab=model_data["vocab"],
            special_tokens=special_tokens,
            pattern=model_data["config"]["pattern"],
            vocab_size=model_data["config"]["default_vocab_size"],
            num_special_tokens=model_data["config"]["default_num_special_tokens"],
            version=version,
            name=path.name.replace(".json", ""),
            image_config=model_data.get("image"),
            _path=path,
        )

    @property
    def image(self) -> Optional[ImageConfig]:
        r"""The image configuration of the tokenizer."""
        return self._image_config

    @image.setter
    def image(self, value: ImageConfig) -> None:
        raise ValueError("Can only set Image config at init")

    @property
    def num_special_tokens(self) -> int:
        r"""The number of special tokens of the tokenizer."""
        return len(self._all_special_tokens)

    @property
    def n_words(self) -> int:
        r"""Vocabulary size of the tokenizer."""
        return self._vocab_size

    @property
    def version(self) -> TokenizerVersion:
        r"""The version of the tokenizer."""
        return self._version

    @property
    def special_token_policy(self) -> SpecialTokenPolicy:
        r"""The policy for handling special tokens."""
        return self._special_token_policy

    @special_token_policy.setter
    def special_token_policy(self, policy: SpecialTokenPolicy) -> None:
        r"""Set the policy for handling special tokens."""
        if not isinstance(policy, SpecialTokenPolicy):
            raise ValueError(f"Expected SpecialTokenPolicy, got {type(policy)}.")

        warnings.warn(
            (
                "The attributed `special_token_policy` is deprecated and will be removed in 1.7.0. "
                "Please pass a special token policy explicitly to the relevant methods."
            ),
            FutureWarning,
        )

        self._special_token_policy = policy

    @cached_property
    def bos_id(self) -> int:
        r"""The beginning of sentence token id."""
        return self.get_control_token("<s>")

    @cached_property
    def eos_id(self) -> int:
        r"""The end of sentence token id."""
        return self.get_control_token("</s>")

    @cached_property
    def pad_id(self) -> int:
        r"""The padding token id."""
        return self.get_control_token("<pad>")

    @cached_property
    def unk_id(self) -> int:
        r"""The unknown token id."""
        return self.get_control_token("<unk>")

    def vocab(self) -> List[str]:
        r"""All tokens in the vocabulary as strings.

        Note:
           This will collapse all tokens for which we have a decoding error into
           the <?> string. This is bad and results in things like len(set(vocab)) != len(vocab)).

        Returns:
            The vocabulary of the tokenizer.
        """
        # when returning self._vocab this will collapse
        # all tokens for which we have a decoding error into
        # the <?> string. This is bad and results in things
        # like len(set(vocab)) != len(vocab))
        # be careful when using self._vocab
        return self._vocab

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        r"""Encode a string into a list of token ids.

        Args:
            s: The string to encode.
            bos: Whether to add the beginning of sentence token.
            eos: Whether to add the end of sentence token.

        Returns:
            The list of token ids.
        """
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
        r"""Check if a token id is a byte token."""
        return 0 <= token_id - self.num_special_tokens < 256

    def get_control_token(self, s: str) -> int:
        r"""Get the token id of a control token."""
        if s in self._special_tokens_reverse_vocab:
            return self._special_tokens_reverse_vocab[s]
        else:
            raise ValueError(f"Unknown control token {s}")

    def decode(self, tokens: List[int], special_token_policy: Optional[SpecialTokenPolicy] = None) -> str:
        r"""Decode a list of token ids into a string.

        Args:
            tokens: The list of token ids to decode.
            special_token_policy: The policy for handling special tokens.
                Use the tokenizer's [attribute][mistral_common.tokens.tokenizers.tekken.Tekkenizer.special_token_policy]
                if `None`. Passing `None` is deprecated and will be changed
                to `SpecialTokenPolicy.IGNORE` in `mistral_common=1.7.0`.

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
                    f"Using the tokenizer's special token policy ({self._special_token_policy}) is deprecated. "
                    "It will be removed in 1.7.0. "
                    "Please pass a special token policy explicitly. "
                    "Future default will be SpecialTokenPolicy.IGNORE."
                ),
                FutureWarning,
            )
            special_token_policy = self._special_token_policy

        return "".join(self._decode_all(tokens, special_token_policy=special_token_policy))

    def to_string(self, tokens: List[int]) -> str:
        r"""[DEPRECATED] Converts a list of token ids into a string, keeping special tokens.

        Use `decode` with `special_token_policy=SpecialTokenPolicy.KEEP` instead.

        This is a convenient method for debugging.
        """
        warnings.warn(
            (
                "`to_string` is deprecated and will be removed in 1.7.0. "
                "Use `decode` with `special_token_policy=SpecialTokenPolicy.KEEP` instead."
            ),
            FutureWarning,
        )
        return self._to_string(tokens)

    def _to_string(self, tokens: List[int]) -> str:
        return self.decode(tokens, special_token_policy=SpecialTokenPolicy.KEEP)

    def id_to_piece(self, token_id: int) -> str:
        r"""Convert a token id to its string representation."""
        return self.decode([token_id], special_token_policy=SpecialTokenPolicy.KEEP)

    def id_to_byte_piece(self, token_id: int, special_token_policy: Optional[SpecialTokenPolicy] = None) -> bytes:
        r"""Convert a token id to its byte representation.

        Args:
            token_id: The token id to convert.
            special_token_policy: The policy for handling special tokens.
                Use the tokenizer's [attribute][mistral_common.tokens.tokenizers.tekken.Tekkenizer.special_token_policy]
                if `None`. Passing `None` is deprecated and will be changed
                to `SpecialTokenPolicy.IGNORE` in `mistral_common=1.7.0`.

        Returns:
            The byte representation of the token.
        """
        if special_token_policy is None:
            warnings.warn(
                (
                    f"Using the tokenizer's special token policy ({self._special_token_policy}) is deprecated. "
                    "It will be removed in 1.7.0. "
                    "Please pass a special token policy explicitly. "
                    "Future default will be SpecialTokenPolicy.IGNORE."
                ),
                FutureWarning,
            )
            special_token_policy = self._special_token_policy

        if token_id < self.num_special_tokens:
            if special_token_policy == SpecialTokenPolicy.KEEP:
                return self._all_special_tokens[token_id]["token_str"].encode("utf-8")
            elif special_token_policy == SpecialTokenPolicy.RAISE:
                raise ValueError(f"{token_id} is a special token")
            elif special_token_policy == SpecialTokenPolicy.IGNORE:
                return b""
            else:
                raise ValueError(f"Unknown special token policy {special_token_policy}")

        return self._model.decode_single_token_bytes(token_id - self.num_special_tokens)


def _reload_mergeable_ranks(
    vocab: List[TokenInfo],
    max_vocab: Union[int, None] = None,
) -> Dict[bytes, int]:
    r"""Reload our tokenizer JSON file and convert it to Tiktoken format."""
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

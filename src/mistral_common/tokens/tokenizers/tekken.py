import base64
import json
import logging
import warnings
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import TypedDict, TypeGuard

import numpy as np
import tiktoken

from mistral_common.tokens.tokenizers.audio import AudioConfig, AudioSpectrogramConfig
from mistral_common.tokens.tokenizers.base import (
    SpecialTokenPolicy,
    SpecialTokens,
    Tokenizer,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.image import ImageConfig
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder

warnings.filterwarnings(
    action="once",
    category=FutureWarning,
    message=r".*`get_control_token` is deprecated.*",
)


logger = logging.getLogger(__name__)


def is_tekken(path: str | Path) -> bool:
    r"""Check if the given path is a tekken tokenizer file."""
    if isinstance(path, str):
        path = Path(path)
    return path.is_file() and "tekken" in path.name and path.suffix == ".json"


# Formatting specification of the JSON file
class TokenInfo(TypedDict):
    r"""Token information in the JSON file.

    Attributes:
        rank: The integer rank/index of this token in the vocabulary.
        token_bytes: The token's byte representation, base64 encoded.
        token_str: The token's string representation, or `None` if not applicable.
    """

    rank: int
    token_bytes: str  # base64 encoded
    token_str: str | None


class SpecialTokenInfo(TypedDict):
    r"""Special token information in the JSON file.

    Attributes:
        rank: The integer rank/index of this special token.
        token_str: The string representation of the special token.
        is_control: `True` if this is a control token (non-printable), `False` otherwise.
    """

    rank: int
    token_str: str
    is_control: bool


class TekkenConfig(TypedDict):
    r"""Tekken tokenizer configuration in the JSON file.

    Attributes:
        pattern: Regex pattern string used for tokenization (tiktoken `pat_str`).
        num_vocab_tokens: Number of regular (non-special) tokens in the vocabulary.
        default_vocab_size: Default total vocabulary size (vocab + special tokens).
        default_num_special_tokens: Default number of special tokens.
        version: Version string of the tokenizer (e.g., "v1", "v2").
    """

    pattern: str
    num_vocab_tokens: int
    default_vocab_size: int
    default_num_special_tokens: int
    version: str


class ModelData(TypedDict):
    r"""Complete tekken tokenizer model data from the JSON file.

    Attributes:
        vocab: List of TokenInfo entries for regular vocabulary tokens.
        special_tokens: List of SpecialTokenInfo entries, or `None` to use defaults.
        config: TekkenConfig with tokenizer settings.
        version: Integer version of the tokenizer file format.
        type: String type identifier for the tokenizer.
        image: ImageConfig for image processing, or missing if not supported.
        audio: AudioConfig for audio processing, or missing if not supported.
    """

    vocab: list[TokenInfo]
    special_tokens: list[SpecialTokenInfo] | None
    config: TekkenConfig
    version: int
    type: str
    image: ImageConfig
    audio: AudioConfig


class Tekkenizer(Tokenizer):
    r"""Tekken tokenizer based on the tiktoken library.

    High-performance tokenizer for multiple languages, supporting text, image, and
    audio modalities. Uses byte-level BPE with customizable patterns and special
    tokens.

    The tokenizer works by:
    1. Using tiktoken's fast BPE encoder for regular tokens
    2. Managing special tokens separately (prefixed to the vocabulary)
    3. Supporting multimodal configurations via `image_config` and `audio_config`
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
        vocab: list[TokenInfo],
        special_tokens: list[SpecialTokenInfo],
        pattern: str,
        vocab_size: int,
        num_special_tokens: int,
        version: TokenizerVersion,
        *,
        name: str = "tekkenizer",
        _path: str | Path | None = None,
        image_config: ImageConfig | None = None,
        audio_config: AudioConfig | None = None,
        model_settings_builder: ModelSettingsBuilder | None = None,
    ):
        r"""Initialize the tekken tokenizer.

        Args:
            vocab: List of token information defining the vocabulary. Each entry contains
                the token's rank, base64-encoded bytes, and string representation.
            special_tokens: List of special token definitions. If fewer than
                `num_special_tokens` are provided, filler tokens are generated automatically.
            pattern: Regex pattern used for tokenization (tiktoken `pat_str`).
            vocab_size: Total vocabulary size (vocab tokens + special tokens).
                Must be <= len(vocab) + `num_special_tokens`.
            num_special_tokens: Total number of special tokens. If `special_tokens` list
                is shorter, filler tokens are added to reach this count.
            version: Tokenizer version. Determines supported features and validation rules.
            name: Identifier for this tokenizer instance. Defaults to "tekkenizer".
            _path: Source file path. Internal use only; not for direct instantiation.
            image_config: Configuration for image processing, or `None` if not supported.
            audio_config: Configuration for audio processing, or `None` if not supported.
            model_settings_builder: Builder for model-specific settings. Must be `None` if
                version does not support model settings (pre-v15).

        Raises:
            ValueError: If `model_settings_builder` is provided but version does not support it.
            AssertionError: If `vocab_size` constraint is violated or special tokens are invalid.
        """
        if not version.supports_model_settings and model_settings_builder is not None:
            raise ValueError(
                f"model_settings_builder is not supported for {version=} but got {model_settings_builder=}"
            )

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
        logger.info(f"Non special vocabulary size is {inner_vocab_size} with {num_special_tokens} special tokens.")
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
        self._audio_config = audio_config

        self._all_special_tokens = special_tokens
        self._special_token_ids = {t["rank"] for t in special_tokens}
        self._special_tokens_reverse_vocab = {t["token_str"]: t["rank"] for t in special_tokens}
        self._vocab = [self.id_to_piece(i) for i in range(vocab_size)]
        self._special_token_policy = SpecialTokenPolicy.IGNORE
        self._file_path = Path(_path) if _path is not None else None
        self._model_settings_builder = model_settings_builder

    @property
    def file_path(self) -> Path:
        r"""The path to the tokenizer file.

        Returns:
            Path to the source JSON file.

        Raises:
            ValueError: If the tokenizer was not loaded from a file (e.g., constructed
                directly via __init__ without _path).
        """
        if self._file_path is None:
            raise ValueError("The tokenizer was not loaded from a file.")
        return self._file_path

    @property
    def model_settings_builder(self) -> ModelSettingsBuilder | None:
        r"""The model settings builder for this tokenizer.

        Returns:
            ModelSettingsBuilder instance if the tokenizer version supports model
            settings, otherwise `None`.
        """
        return self._model_settings_builder

    @classmethod
    def from_file(cls: type["Tekkenizer"], path: str | Path) -> "Tekkenizer":
        r"""Load the tekken tokenizer from a JSON file.

        The file must contain vocab, config, and optionally `special_tokens`, image,
        audio, and `model_settings_builder` sections.

        Args:
            path: Path to the tokenizer JSON file. Must exist and be readable.

        Returns:
            A Tekkenizer instance configured from the file.

        Raises:
            ValueError: If the file has an unknown version, is missing required fields,
                or contains incompatible configuration (e.g., `model_settings_builder`
                with a version that does not support it).
            AssertionError: If the file does not exist.
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

        special_tokens_dicts: list[SpecialTokenInfo] | None = untyped.get("special_tokens", None)
        if special_tokens_dicts is None:
            # Tokenizer > v7 should find special tokens in the tokenizer file
            if version > TokenizerVersion.v7:
                raise ValueError(
                    f"Special tokens not found in {path}. "
                    "Please update your tokenizer file and include all special tokens you need."
                )
            else:
                special_tokens = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
        else:
            special_tokens = [token for token in special_tokens_dicts]

        untyped["special_tokens"] = special_tokens

        if mm := untyped.get("multimodal"):
            # deprecated - only allowed for tokenizers <= v11
            if version > TokenizerVersion.v11:
                raise ValueError(
                    f"The image config has to be called 'image' in {path} for tokenizers of version {version.value}."
                )

            untyped["image"] = ImageConfig(**mm)
        elif image := untyped.get("image"):
            untyped["image"] = ImageConfig(**image)

        if audio := untyped.get("audio"):
            encoding_config = audio.pop("audio_encoding_config")
            encoding_config = AudioSpectrogramConfig(**encoding_config)
            untyped["audio"] = AudioConfig(encoding_config=encoding_config, **audio)

        if (
            model_settings_builder := untyped.get("model_settings_builder")
        ) is not None and not version.supports_model_settings:
            raise ValueError(
                f"model_settings_builder is not supported for {version=} but got {model_settings_builder=}"
            )
        elif model_settings_builder is not None:
            model_settings_builder = ModelSettingsBuilder.model_validate(model_settings_builder)

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
            audio_config=model_data.get("audio"),
            model_settings_builder=model_settings_builder,
            _path=path,
        )

    @property
    def image(self) -> ImageConfig | None:
        r"""The image configuration for this tokenizer.

        Returns:
            ImageConfig instance if image support is configured, otherwise `None`.
        """
        return self._image_config

    @image.setter
    def image(self, value: ImageConfig) -> None:
        r"""Setting the image config is not allowed.

        Raises:
            ValueError: Always; the image config can only be set at init.
        """
        raise ValueError("Can only set Image config at init")

    @property
    def audio(self) -> AudioConfig | None:
        r"""The audio configuration for this tokenizer.

        Returns:
            AudioConfig instance if audio support is configured, otherwise `None`.
        """
        return self._audio_config

    @audio.setter
    def audio(self, value: AudioConfig) -> None:
        r"""Setting the audio config is not allowed.

        Raises:
            ValueError: Always; the audio config can only be set at init.
        """
        raise ValueError("Can only set Audio config at init")

    @property
    def num_special_tokens(self) -> int:
        r"""The total number of special tokens in this tokenizer.

        Returns:
            Count of all special tokens (original + filler tokens).
        """
        return len(self._all_special_tokens)

    @property
    def n_words(self) -> int:
        r"""Total vocabulary size of the tokenizer.

        Returns:
            Sum of regular vocabulary tokens and special tokens.
        """
        return self._vocab_size

    @cached_property
    def special_ids(self) -> set[int]:
        r"""Set of all special token IDs.

        Returns:
            Set of integer IDs for all special tokens in this tokenizer.
        """
        return {token["rank"] for token in self._all_special_tokens}

    @property
    def version(self) -> TokenizerVersion:
        r"""The version of this tokenizer.

        Returns:
            TokenizerVersion enum value indicating the tokenizer's version.
        """
        return self._version

    @cached_property
    def bos_id(self) -> int:
        r"""The beginning-of-sentence token ID."""
        return self.get_special_token("<s>")

    @cached_property
    def eos_id(self) -> int:
        r"""The end-of-sentence token ID."""
        return self.get_special_token("</s>")

    @cached_property
    def pad_id(self) -> int:
        r"""The padding token ID."""
        return self.get_special_token("<pad>")

    @cached_property
    def unk_id(self) -> int:
        r"""The unknown token ID."""
        return self.get_special_token("<unk>")

    def vocab(self) -> list[str]:
        r"""Get all tokens in the vocabulary as strings.

        Note:
            Tokens with decoding errors are collapsed into the "<?>" string. This may
            result in len(set(vocab)) != len(vocab). Use with caution.

        Returns:
            List of all tokens in the vocabulary as strings, in token ID order.
        """
        # when returning self._vocab this will collapse
        # all tokens for which we have a decoding error into
        # the <?> string. This is bad and results in things
        # like len(set(vocab)) != len(vocab))
        # be careful when using self._vocab
        return self._vocab

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        r"""Encode a string into a list of token ids.

        Args:
            s: The string to encode.
            bos: If `True`, prepends the beginning-of-sentence token ID to the result.
            eos: If `True`, appends the end-of-sentence token ID to the result.

        Returns:
            List of token IDs. Regular tokens are offset by `num_special_tokens`.
        """
        tokens: list[int] = self._model.encode(s)
        tokens = [t + self.num_special_tokens for t in tokens]
        if bos:
            tokens = [self.bos_id, *tokens]
        if eos:
            tokens = [*tokens, self.eos_id]
        return tokens

    def _decode_all(self, tokens: list[int], special_token_policy: SpecialTokenPolicy) -> list[str]:
        # Lump special and non-special tokens together to minimize calls to decode
        decoded: list[str] = []
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
        r"""Check if a token ID represents a single byte.

        Args:
            token_id: The token ID to check.

        Returns:
            `True` if the token (after subtracting special token offset) is in the
            range [0, 255], meaning it represents a single byte.
        """
        return 0 <= token_id - self.num_special_tokens < 256

    def get_special_token(self, s: str) -> int:
        r"""Get the token ID of a special token by its string representation.

        Args:
            s: The string representation of the special token (e.g., "<s>", "<eos>").

        Returns:
            The integer token ID for the special token.

        Raises:
            ValueError: If the special token string is not recognized.
        """
        if s in self._special_tokens_reverse_vocab:
            return self._special_tokens_reverse_vocab[s]
        else:
            raise ValueError(f"Unknown control token {s}")

    def is_special(self, token: int | np.integer | str) -> bool:
        r"""Check if a token is a special token.

        Args:
            token: Token ID (int or numpy integer) or token string to check.

        Returns:
            `True` if the token is a special token, `False` otherwise.

        Raises:
            TypeError: If token is not an int, numpy integer, or str.
        """
        if isinstance(token, (int, np.integer)):
            return token in self._special_token_ids
        elif isinstance(token, str):
            return token in self._special_tokens_reverse_vocab
        else:
            raise TypeError(f"Expected int or str, got {type(token).__name__}")

    def get_control_token(self, s: str) -> int:
        r"""Get the token ID of a control token. Deprecated: use `get_special_token()` instead.

        Deprecated: Use `get_special_token()` instead.

        Args:
            s: The string representation of the control token.

        Returns:
            The integer token ID for the control token.
        """
        warnings.warn("`get_control_token` is deprecated. Use `get_special_token` instead.", FutureWarning)
        return self.get_special_token(s)

    def decode(self, tokens: list[int], special_token_policy: SpecialTokenPolicy = SpecialTokenPolicy.IGNORE) -> str:
        r"""Decode a list of token IDs into a string.

        Args:
            tokens: List of token IDs to decode.
            special_token_policy: Policy for handling special tokens:
                - IGNORE: Skip special tokens (default)
                - KEEP: Include special token strings in output
                - RAISE: Raise ValueError if special tokens are present

        Returns:
            The decoded UTF-8 string.

        Raises:
            ValueError: If `special_token_policy` is invalid or RAISE is set and
                special tokens are encountered.
        """
        try:
            special_token_policy = SpecialTokenPolicy(special_token_policy)
        except ValueError as e:
            valid = ", ".join(repr(p.value) for p in SpecialTokenPolicy)
            raise ValueError(
                f"Invalid `special_token_policy` {special_token_policy!r}. Expected one of: {valid}."
            ) from e

        return "".join(self._decode_all(tokens, special_token_policy=special_token_policy))

    def _to_string(self, tokens: list[int]) -> str:
        return self.decode(tokens, special_token_policy=SpecialTokenPolicy.KEEP)

    def id_to_piece(self, token_id: int) -> str:
        r"""Convert a token ID to its string representation.

        Args:
            token_id: The token ID to convert.

        Returns:
            The string representation of the token. Special tokens are decoded as their
            token string; regular tokens are decoded using the underlying tiktoken model.
        """
        return self.decode([token_id], special_token_policy=SpecialTokenPolicy.KEEP)

    def id_to_byte_piece(
        self, token_id: int, special_token_policy: SpecialTokenPolicy = SpecialTokenPolicy.IGNORE
    ) -> bytes:
        r"""Convert a token ID to its byte representation.

        Args:
            token_id: The token ID to convert.
            special_token_policy: Policy for handling special tokens:
                - IGNORE: Return empty bytes for special tokens
                - KEEP: Return UTF-8 encoded special token string
                - RAISE: Raise ValueError if token is special

        Returns:
            The byte representation of the token. For regular tokens, returns the
            original byte sequence from the vocabulary.

        Raises:
            ValueError: If `special_token_policy` is RAISE and `token_id` is a special token,
                or if `special_token_policy` is invalid.
        """
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
    vocab: list[TokenInfo],
    max_vocab: int | None = None,
) -> dict[bytes, int]:
    r"""Convert vocab TokenInfo list to tiktoken mergeable ranks format.

    Args:
        vocab: List of TokenInfo entries from the tokenizer JSON.
        max_vocab: Maximum number of vocabulary entries to include. If `None`,
            includes all. If provided and less than len(vocab), truncates to
            first `max_vocab` entries.

    Returns:
        Dictionary mapping byte sequences to their integer ranks.
    """
    if max_vocab is not None:
        assert len(vocab) >= max_vocab, (len(vocab), max_vocab)
        if len(vocab) > max_vocab:
            vocab = vocab[:max_vocab]
            logger.info(f"Cutting non special vocabulary to first {len(vocab)} tokens.")

    # build ranks
    ranks: dict[bytes, int] = {}
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


def is_tekkenizer(tokenizer: Tokenizer) -> TypeGuard[Tekkenizer]:
    r"""Check if a tokenizer is a Tekkenizer.

    Args:
        tokenizer: The tokenizer to check.

    Returns:
        `True` if the tokenizer is an instance of Tekkenizer, `False` otherwise.
    """
    return isinstance(tokenizer, Tekkenizer)

import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
from pydantic import ConfigDict

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.messages import (
    AssistantMessageType,
    ContentChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import Tool
from mistral_common.tokens.instruct.request import FIMRequest, InstructRequest
from mistral_common.tokens.tokenizers.image import ImageEncoder


class UserMessagePosition(str, Enum):
    """Where to encode available tools"""

    first = "first"
    last = "last"


class SpecialTokens(str, Enum):
    r"""[DEPRECATED] Enum of special tokens used in the tokenizer.

    Attributes:
        unk: The unknown token.
        bos: The beginning of string token.
        eos: The end of string token.
        begin_inst: The beginning of instruction token.
        end_inst: The end of instruction token.
        begin_tools: The beginning of tools token.
        end_tools: The end of tools token.
        begin_tool_results: The beginning of tool results token.
        end_tool_results: The end of tool results token.
        tool_calls: The tool calls token.
        img: The image token.
        pad: The pad token.
        img_break: The image break token.
        img_end: The image end token.
        prefix: The prefix token for FIM.
        middle: The middle token for FIM.
        suffix: The suffix token for FIM.
        begin_system: The beginning of system prompt token.
        end_system: The end of system prompt token.
        begin_tool_content: The beginning of tool content token.

    Examples:
        >>> unk = SpecialTokens.unk
    """

    unk = "<unk>"
    bos = "<s>"
    eos = "</s>"
    begin_inst = "[INST]"
    end_inst = "[/INST]"
    begin_tools = "[AVAILABLE_TOOLS]"
    end_tools = "[/AVAILABLE_TOOLS]"
    begin_tool_results = "[TOOL_RESULTS]"
    end_tool_results = "[/TOOL_RESULTS]"
    tool_calls = "[TOOL_CALLS]"
    img = "[IMG]"
    pad = "<pad>"
    img_break = "[IMG_BREAK]"
    img_end = "[IMG_END]"
    prefix = "[PREFIX]"
    middle = "[MIDDLE]"
    suffix = "[SUFFIX]"
    begin_system = "[SYSTEM_PROMPT]"
    end_system = "[/SYSTEM_PROMPT]"
    begin_tool_content = "[TOOL_CONTENT]"
    args = "[ARGS]"
    call_id = "[CALL_ID]"


class SpecialTokenPolicy(int, Enum):
    r"""What to do with special tokens when encoding/decoding.

    Attributes:
        IGNORE: Ignore special tokens.
        KEEP: Keep special tokens.
        RAISE: Raise an error if special tokens are found.
    """

    IGNORE = 0
    KEEP = 1
    RAISE = 2


class TokenizerVersion(str, Enum):
    r"""Enum of tokenizer versions.

    Allow to distinguish between different versions of the tokenizer and maintain backward compatibility.

    Attributes:
        v1: The first version of the tokenizer.
        v2: The second version of the tokenizer that includes special control tokens [INST], [\INST].
        v3: The third version of the tokenizer that includes improved function calling.
        v7: The seventh version of the tokenizer that includes improved system prompt and function calling.
        v11: The eleventh version of the tokenizer that includes improved function calling.
        v13: The thirteenth version of the tokenizer that includes no call id tokenization and better prompt caching.

    Examples:
        >>> version = TokenizerVersion.v1
    """

    def __new__(cls, value: str) -> "TokenizerVersion":
        if not re.match(r"^v\d+$", value):
            raise ValueError(f"Invalid version format: {value}. Must be 'v' followed by a number.")
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    @property
    def _version_num(self) -> int:
        return int(self.value[1:])

    def __lt__(self, other: "str | TokenizerVersion") -> bool:
        if isinstance(other, str):
            other = TokenizerVersion(other)
        return self._version_num < other._version_num

    def __le__(self, other: "str | TokenizerVersion") -> bool:
        if isinstance(other, str):
            other = TokenizerVersion(other)
            return self._version_num <= other._version_num

    def __gt__(self, other: "str | TokenizerVersion") -> bool:
        if isinstance(other, str):
            other = TokenizerVersion(other)
            return self._version_num > other._version_num

    def __ge__(self, other: "str | TokenizerVersion") -> bool:
        if isinstance(other, str):
            other = TokenizerVersion(other)
            return self._version_num >= other._version_num

    v1 = "v1"  # vocab_size = 32000
    v2 = "v2"  # vocab_size = 32768 with special control tokens [INST], [\INST]
    v3 = "v3"  # vocab_size = 32768 (spm) OR 131072 (tekken) with improved function calling
    v7 = "v7"  # vocab_size = 32768 (spm) or 131072 (tekken) with improved system prompt and function calling
    v11 = "v11"  # 131072 (tekken) with improved function calling
    v13 = "v13"  # 131072 (tekken) with no call id and better prompt caching


class Tokenized(MistralBase):
    r"""A tokenized [`InstructRequest`][mistral_common.tokens.instruct.request].

    Attributes:
        tokens: The token ids.
        text: The text representation of the tokens.
        prefix_ids: The prefix ids for FIM.
        images: The loaded images associated with the tokens.

    Examples:
        >>> tokenized = Tokenized(tokens=[1, 2, 3], text="Hello world", prefix_ids=[1], images=[])
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tokens: List[int]
    text: Optional[str] = None
    prefix_ids: Optional[List[int]] = None
    images: List[np.ndarray] = []


class Tokenizer(ABC):
    @property
    @abstractmethod
    def n_words(self) -> int:
        r"""Vocabulary size of the tokenizer."""

    @abstractmethod
    def vocab(self) -> List[str]:
        r"""All tokens in the vocabulary as strings."""

    @abstractmethod
    def id_to_piece(self, token_id: int) -> str:
        r"""Convert a token id to the token str."""

    @property
    @abstractmethod
    def bos_id(self) -> int:
        r"""id of the Beginning of String token."""

    @property
    @abstractmethod
    def eos_id(self) -> int:
        r"""id of the End of String token."""

    @property
    @abstractmethod
    def pad_id(self) -> int:
        r"""id of the Pad token."""

    @property
    @abstractmethod
    def unk_id(self) -> int:
        r"""id of the Unk token."""

    @abstractmethod
    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """Convert a string to a list of token ids."""

    @abstractmethod
    def decode(self, tokens: List[int], special_token_policy: Optional[SpecialTokenPolicy] = None) -> str:
        r"""Decode the token ids to a string.

        Args:
            tokens: The token ids to decode.
            special_token_policy: The policy to use for special tokens.
                Passing `None` will default to `self._special_token_policy` for
                [Tekkenizer][mistral_common.tokens.tokenizers.tekken.Tekkenizer] and `SpecialTokenPolicy.IGNORE`
                for [SentencePieceTokenizer][mistral_common.tokens.tokenizers.sentencepiece.SentencePieceTokenizer].
                Note that passing `None` will be deprecated and `special_token_policy` will default to
                `SpecialTokenPolicy.IGNORE` in `mistral_common=1.7.0`.

        Returns:
            The decoded string.
        """

    @abstractmethod
    def get_control_token(self, s: str) -> int:
        r"""Get the id of a control token."""

    @property
    @abstractmethod
    def version(self) -> TokenizerVersion:
        r"""Get the version of the tokenizer."""

    @abstractmethod
    def to_string(self, tokens: List[int]) -> str:
        r"""[DEPRECATED] Converts a list of token ids into a string, keeping special tokens.

        Use `decode` with `special_token_policy=SpecialTokenPolicy.KEEP` instead.

        This is a convenient method for debugging.
        """
        ...

    @abstractmethod
    def _to_string(self, tokens: List[int]) -> str: ...

    @property
    @abstractmethod
    def file_path(self) -> Path:
        r"""The file path of the tokenizer."""
        ...


InstructRequestType = TypeVar("InstructRequestType", bound=InstructRequest)
FIMRequestType = TypeVar("FIMRequestType", bound=FIMRequest)
TokenizedType = TypeVar("TokenizedType", bound=Tokenized)


class InstructTokenizer(Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]):
    r"""Base class for instruct tokenizers.

    Attributes:
        tokenizer: The tokenizer to use.
        image_encoder: The image encoder to use if any.
    """

    tokenizer: Tokenizer
    image_encoder: Optional[ImageEncoder]

    def __init__(self, tokenizer: Tokenizer, image_encoder: Optional[ImageEncoder]) -> None:
        r"""Initialize the instruct tokenizer.

        Args:
            tokenizer: The tokenizer to use.
            image_encoder: The image encoder to use if any.
        """

    @abstractmethod
    def encode_instruct(self, request: InstructRequestType) -> TokenizedType:
        r"""Instruct request to Tokenized object

        Args:
            request: The instruct request to encode.

        Returns:
            The tokenized instruct request.
        """

    @abstractmethod
    def decode(self, tokens: List[int], special_token_policy: Optional[SpecialTokenPolicy] = None) -> str:
        r"""Convert token ids to string

        Args:
            tokens: The token ids to decode.
            special_token_policy: The policy to use for special tokens.
                Passing `None` will default to `self._special_token_policy` for
                [Tekkenizer][mistral_common.tokens.tokenizers.tekken.Tekkenizer] and `SpecialTokenPolicy.IGNORE`
                for [SentencePieceTokenizer][mistral_common.tokens.tokenizers.sentencepiece.SentencePieceTokenizer].
                Note that passing `None` will be deprecated and `special_token_policy` will default to
                `SpecialTokenPolicy.IGNORE` in `mistral_common=1.7.0`.

        Returns:
            The decoded string.
        """

    @abstractmethod
    def encode_fim(self, request: FIMRequestType) -> TokenizedType:
        r"""FIM request to Tokenized object

        Args:
            request: The FIM request to encode.

        Returns:
            The tokenized FIM request.
        """

    @abstractmethod
    def encode_user_message(
        self,
        message: UserMessage,
        available_tools: Optional[List[Tool]],
        is_last: bool,
        is_first: bool,
        system_prompt: Optional[str] = None,
        force_img_first: bool = False,
    ) -> Tuple[List[int], List[np.ndarray]]:
        r"""Encode a user message.

        Args:
            message: The user message to encode.
            available_tools: The available tools.
            is_last: Whether the message is the last one.
            is_first: Whether the message is the first one.
            system_prompt: The system prompt.
            force_img_first: Whether to force the image to be first.

        Returns:
            The encoded tokens and images.
        """
        ...

    @abstractmethod
    def encode_user_content(
        self,
        content: Union[str, List[ContentChunk]],
        is_last: bool,
        system_prompt: Optional[str] = None,
        force_img_first: bool = False,
    ) -> Tuple[List[int], List[np.ndarray]]:
        r"""Encode a user content.

        Args:
            content: The user content to encode.
            is_last: Whether the content is the last one.
            system_prompt: The system prompt.
            force_img_first: Whether to force the image to be first.

        Returns:
            The encoded tokens and images.
        """
        ...

    @abstractmethod
    def _to_string(self, tokens: List[int]) -> str: ...

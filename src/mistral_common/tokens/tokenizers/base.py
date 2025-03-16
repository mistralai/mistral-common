from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generic, List, Optional, Protocol, Tuple, TypeVar, Union

import numpy as np
from pydantic import ConfigDict

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.messages import (
    AssistantMessageType,
    ContentChunk,
    ImageChunk,
    ImageURLChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import Tool
from mistral_common.tokens.instruct.request import FIMRequest, InstructRequest


class SpecialTokens(str, Enum):
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
    img_break = "[IMG_BREAK]"
    img_end = "[IMG_END]"
    prefix = "[PREFIX]"
    middle = "[MIDDLE]"
    suffix = "[SUFFIX]"
    begin_system = "[SYSTEM_PROMPT]"
    end_system = "[/SYSTEM_PROMPT]"
    begin_tool_content = "[TOOL_CONTENT]"


class TokenizerVersion(str, Enum):
    v1 = "v1"  # vocab_size = 32000
    v2 = "v2"  # vocab_size = 32768 with special control tokens [INST], [\INST]
    v3 = "v3"  # vocab_size = 32768 (spm) OR 128000 (tekken) with improved function calling
    v7 = "v7"  # vocab_size = 32768 (spm) or 128000 (tekken) with improved system prompt and function calling


class Tokenized(MistralBase):
    """
    A tokenized InstructRequest
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
        """Vocabulary size"""

    @abstractmethod
    def vocab(self) -> List[str]:
        """All tokens in the vocabulary as strings"""

    @abstractmethod
    def id_to_piece(self, token_id: int) -> str:
        """Convert a token id to the token str"""

    @property
    @abstractmethod
    def bos_id(self) -> int:
        """id of the Beginning of String token"""

    @property
    @abstractmethod
    def eos_id(self) -> int:
        """id of the End of String token"""

    @property
    @abstractmethod
    def pad_id(self) -> int:
        """id of the Pad token"""

    @property
    @abstractmethod
    def unk_id(self) -> int:
        """id of the Unk token"""

    @abstractmethod
    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """String to token ids"""

    @abstractmethod
    def decode(self, t: List[int]) -> str:
        """Token ids to string"""

    @abstractmethod
    def get_control_token(self, s: str) -> int:
        """Get the id of a control token"""

    @property
    @abstractmethod
    def version(self) -> TokenizerVersion:
        """Get the version of the tokenizer"""

    @abstractmethod
    def to_string(self, tokens: List[int]) -> str:
        """Convert token ids to string"""


InstructRequestType = TypeVar("InstructRequestType", bound=InstructRequest)
FIMRequestType = TypeVar("FIMRequestType", bound=FIMRequest)
TokenizedType = TypeVar("TokenizedType", bound=Tokenized)


@dataclass
class ImageEncoding:
    tokens: List[int]
    image: np.ndarray


@dataclass
class SpecialImageIDs:
    img: int
    img_break: int
    img_end: int

    @staticmethod
    def from_tokenizer(tokenizer: "Tokenizer") -> "SpecialImageIDs":
        return SpecialImageIDs(
            img=tokenizer.get_control_token(SpecialTokens.img.value),
            img_break=tokenizer.get_control_token(SpecialTokens.img_break.value),
            img_end=tokenizer.get_control_token(SpecialTokens.img_end.value),
        )


class MultiModalEncoder(Protocol):
    def __call__(self, content: Union[ImageChunk, ImageURLChunk]) -> ImageEncoding:
        """
        Encode the given content.

        Args:
            content (ChunkContent): The content to be encoded.

        Returns:
            ImageEncoding: The encoded image content.
        """
        ...

    @property
    def image_token(self) -> int:
        ...


class InstructTokenizer(Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]):
    tokenizer: Tokenizer
    mm_encoder: Optional[MultiModalEncoder]

    def __init__(self, tokenizer: Tokenizer, mm_encoder: Optional[MultiModalEncoder]) -> None:
        """Init from tokenizer"""

    @abstractmethod
    def encode_instruct(self, request: InstructRequestType) -> TokenizedType:
        """Instruct request to Tokenized object"""

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Convert token ids to string"""

    @abstractmethod
    def encode_fim(self, request: FIMRequestType) -> TokenizedType:
        """FIM request to Tokenized object"""

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
        ...

    @abstractmethod
    def encode_user_content(
        self,
        content: Union[str, List[ContentChunk]],
        is_last: bool,
        system_prompt: Optional[str] = None,
        force_img_first: bool = False,
    ) -> Tuple[List[int], List[np.ndarray]]:
        ...

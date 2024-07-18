from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, List, Optional, TypeVar

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.messages import AssistantMessageType
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
    prefix = "[PREFIX]"
    middle = "[MIDDLE]"
    suffix = "[SUFFIX]"


class TokenizerVersion(str, Enum):
    v1 = "v1"  # vocab_size = 32000
    v2 = "v2"  # vocab_size = 32768 with special control tokens [INST], [\INST]
    v3 = "v3"  # vocab_size = 32768 (spm) OR 128000 (tekken) with improved function calling


class Tokenized(MistralBase):
    """
    A tokenized InstructRequest
    """

    tokens: List[int]
    text: Optional[str] = None
    prefix_ids: Optional[List[int]] = None


class Tokenizer(ABC):
    @property
    @abstractmethod
    def n_words(self) -> int:
        """Vocabulary size"""

    @abstractmethod
    def vocab(self) -> List[str]:
        """All tokens in the vocabulary as strings"""

    @property
    @abstractmethod
    def bos_id(self) -> int:
        """id of the Beginning of String token"""

    @property
    @abstractmethod
    def eos_id(self) -> int:
        """id of the End of String token"""

    @abstractmethod
    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """String to token ids"""

    @abstractmethod
    def decode(self, t: List[int]) -> str:
        """Token ids to string"""

    @abstractmethod
    def get_control_token(self, s: str) -> int:
        """Get the id of a control token"""

    @abstractmethod
    def to_string(self, tokens: List[int]) -> str:
        """Convert token ids to string"""


InstructRequestType = TypeVar("InstructRequestType", bound=InstructRequest)
FIMRequestType = TypeVar("FIMRequestType", bound=FIMRequest)
TokenizedType = TypeVar("TokenizedType", bound=Tokenized)


class InstructTokenizer(Generic[InstructRequestType, FIMRequestType, TokenizedType, AssistantMessageType]):
    tokenizer: Tokenizer

    def __init__(self, tokenizer: Tokenizer) -> None:
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

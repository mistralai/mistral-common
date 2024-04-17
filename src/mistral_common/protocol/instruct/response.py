import time
from enum import Enum
from typing import List, Optional

from pydantic import Field

from mistral_common.base import MistralBase
from mistral_common.protocol.base import UsageInfo
from mistral_common.protocol.instruct.tool_calls import ToolCall
from mistral_common.protocol.utils import random_uuid


class FinishReason(str, Enum):
    stop: str = "stop"
    length: str = "length"
    model_length: str = "model_length"
    error: str = "error"
    tool_call: str = "tool_calls"


class ChatCompletionTokenLogprobs(MistralBase):
    token: str
    logprob: float
    bytes: List[int]


class ChatCompletionResponseChoiceLogprobs(MistralBase):
    content: List[ChatCompletionTokenLogprobs]


class DeltaMessage(MistralBase):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionResponseChoice(MistralBase):
    index: int
    message: DeltaMessage
    finish_reason: Optional[FinishReason] = None
    logprobs: Optional[ChatCompletionResponseChoiceLogprobs] = None


class ChatCompletionResponse(MistralBase):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionResponseStreamChoice(MistralBase):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[FinishReason] = None
    logprobs: Optional[ChatCompletionResponseChoiceLogprobs] = None


class ChatCompletionStreamResponse(MistralBase):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None

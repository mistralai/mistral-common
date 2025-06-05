import time
from enum import Enum
from typing import List, Optional

from pydantic import Field

from mistral_common.base import MistralBase
from mistral_common.protocol.base import UsageInfo
from mistral_common.protocol.instruct.tool_calls import ToolCall
from mistral_common.protocol.utils import random_uuid


class FinishReason(str, Enum):
    r"""Possible finish reasons.

    Attributes:
       stop: The model hit a natural stop point or a provided stop sequence.
       length: The maximum number of tokens specified in the request was reached.
       model_length: The model hit its context length limit.
       error: An error occurred during generation.
       tool_calls: The model called a tool.

    Examples:
        >>> reason = FinishReason.stop
    """

    stop = "stop"
    length = "length"
    model_length = "model_length"
    error = "error"
    tool_call = "tool_calls"


class ChatCompletionTokenLogprobs(MistralBase):
    r"""Log probabilities for a token.

    Attributes:
        token: The token.
        logprob: The log probability of the token.
        bytes: The bytes of the token.

    Examples:
        >>> token_logprobs = ChatCompletionTokenLogprobs(token="hello", logprob=-0.5, bytes=[104, 101, 108, 108, 111])
    """

    token: str
    logprob: float
    bytes: List[int]


class ChatCompletionResponseChoiceLogprobs(MistralBase):
    r"""Log probabilities for a choice.

    Attributes:
        content: The log probabilities for the content.

    Examples:
       >>> choice_logprobs = ChatCompletionResponseChoiceLogprobs(
       ...     content=[ChatCompletionTokenLogprobs(token="hello", logprob=-0.5, bytes=[104, 101, 108, 108, 111])]
       ... )
    """

    content: List[ChatCompletionTokenLogprobs]


class DeltaMessage(MistralBase):
    r"""A message in a chat completion.

    Attributes:
        role: The role of the message.
        content: The content of the message.
        tool_calls: The tool calls in the message.

    Examples:
        >>> message = DeltaMessage(role="user", content="Hello, world!")
    """

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionResponseChoice(MistralBase):
    r"""A choice in a chat completion.

    Attributes:
       index: The index of the choice.
       message: The message of the choice.
       finish_reason: The finish reason of the choice.
       logprobs: The log probabilities of the choice.

    Examples:
        >>> choice = ChatCompletionResponseChoice(index=0, message=DeltaMessage(role="user", content="Hello, world!"))
    """

    index: int
    message: DeltaMessage
    finish_reason: Optional[FinishReason] = None
    logprobs: Optional[ChatCompletionResponseChoiceLogprobs] = None


class ChatCompletionResponse(MistralBase):
    r"""A chat completion response.

    See [ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest] for the request.

    Attributes:
        id: The id of the response.
        object: The object of the response.
        created: The creation time of the response.
        model: The model of the response.
        choices: The choices of the response.
        usage: The usage of the response.

    Examples:
        >>> response = ChatCompletionResponse(
        ...     id="chatcmpl-123",
        ...     object="chat.completion",
        ...     created=1677652288,
        ...     model="mistral-tiny",
        ...     choices=[
        ...         ChatCompletionResponseChoice(index=0, message=DeltaMessage(role="user", content="Hello, world!"))
        ...     ],
        ...     usage=UsageInfo(prompt_tokens=10, total_tokens=20, completion_tokens=10),
        ... )
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionResponseStreamChoice(MistralBase):
    r"""A choice in a chat completion stream response.

    Attributes:
        index: The index of the choice.
        delta: The delta of the choice.
        finish_reason: The finish reason of the choice.
        logprobs: The log probabilities of the choice.

    Examples:
        >>> choice = ChatCompletionResponseStreamChoice(
        ...     index=0, delta=DeltaMessage(role="user", content="Hello, world!")
        ... )

    """

    index: int
    delta: DeltaMessage
    finish_reason: Optional[FinishReason] = None
    logprobs: Optional[ChatCompletionResponseChoiceLogprobs] = None


class ChatCompletionStreamResponse(MistralBase):
    r"""A chat completion stream response.

    See [ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest] for the request.

    Attributes:
        id: The id of the response.
        object: The object of the response.
        created: The creation time of the response.
        model: The model of the response.
        choices: The choices of the response.
        usage: The usage of the response.

    Examples:
       >>> response = ChatCompletionStreamResponse(
       ...     id="chatcmpl-123",
       ...     object="chat.completion.chunk",
       ...     created=1677652288,
       ...     model="mistral-tiny",
       ...     choices=[
       ...         ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(role="user", content="Hello, world!"))
       ...     ],
       ...     usage=UsageInfo(prompt_tokens=10, total_tokens=20, completion_tokens=10),
       ... )
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None

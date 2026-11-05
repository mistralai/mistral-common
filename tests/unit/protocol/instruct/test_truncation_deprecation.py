import warnings
from collections.abc import Callable, Generator
from typing import Any

import pytest

import mistral_common.deprecation
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

_REMOVAL = "is deprecated and will be removed in 1.15.0. Truncate the conversation before encoding instead."
_CHAT_FIELD_WARNING = f"`ChatCompletionRequest.truncate_for_context_length` {_REMOVAL}"
_INSTRUCT_FIELD_WARNING = f"`InstructRequest.truncate_at_max_tokens` {_REMOVAL}"
_ENCODE_ARG_WARNING = f"`MistralTokenizer.encode_chat_completion(max_model_input_len=...)` {_REMOVAL}"
_WARNED_KEYS = (
    _CHAT_FIELD_WARNING,
    "InstructRequest.truncate_at_max_tokens",
    "MistralTokenizer.encode_chat_completion.max_model_input_len",
)

_MESSAGES: list[ChatMessage] = [
    UserMessage(content="first question " * 10),
    AssistantMessage(content="first answer " * 10),
    UserMessage(content="second question"),
]
_OPENAI_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _reset_truncation_warnings() -> Generator[None, None, None]:
    for key in _WARNED_KEYS:
        mistral_common.deprecation._warned_keys.discard(key)
    yield
    for key in _WARNED_KEYS:
        mistral_common.deprecation._warned_keys.discard(key)


def _record_warnings(call: Callable[[], Any]) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    for warning in caught:
        assert warning.category is DeprecationWarning
        assert warning.filename == __file__
    return [str(warning.message) for warning in caught]


@pytest.mark.parametrize(
    "build",
    [
        lambda: ChatCompletionRequest(messages=_MESSAGES, truncate_for_context_length=True),
        lambda: ChatCompletionRequest(messages=_MESSAGES, truncate_for_context_length=False),
        lambda: ChatCompletionRequest.model_validate(
            {"messages": _OPENAI_MESSAGES, "truncate_for_context_length": True}
        ),
        lambda: ChatCompletionRequest.from_openai(_OPENAI_MESSAGES, truncate_for_context_length=True),
        lambda: ChatCompletionRequest.from_openai(_OPENAI_MESSAGES, truncate_for_context_length=False),
    ],
    ids=["init_true", "init_false", "model_validate", "from_openai_true", "from_openai_false"],
)
def test_truncate_for_context_length_warns(build: Callable[[], Any]) -> None:
    assert _record_warnings(build) == [_CHAT_FIELD_WARNING]


@pytest.mark.parametrize(
    "build",
    [
        lambda: InstructRequest(messages=_MESSAGES, truncate_at_max_tokens=10),
        lambda: InstructRequest(messages=_MESSAGES, truncate_at_max_tokens=None),
        lambda: InstructRequest.model_validate({"messages": _OPENAI_MESSAGES, "truncate_at_max_tokens": 10}),
    ],
    ids=["init_int", "init_none", "model_validate"],
)
def test_truncate_at_max_tokens_warns(build: Callable[[], Any]) -> None:
    assert _record_warnings(build) == [_INSTRUCT_FIELD_WARNING]


def test_encode_chat_completion_max_model_input_len_warns_once_per_deprecated_name() -> None:
    tokenizer = MistralTokenizer.v7()
    full = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=_MESSAGES))
    budget = len(full.tokens) - 10

    def encode() -> Any:
        request = ChatCompletionRequest(messages=_MESSAGES, truncate_for_context_length=True)
        return tokenizer.encode_chat_completion(request, max_model_input_len=budget)

    # One warning per deprecated name: building the InstructRequest internally does not warn.
    assert _record_warnings(encode) == [_CHAT_FIELD_WARNING, _ENCODE_ARG_WARNING]

    # Warnings are emitted once per process, and truncation still applies.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        truncated = encode()
    assert len(truncated.tokens) < len(full.tokens)
    assert truncated.tokens[-4:] == full.tokens[-4:]


def test_no_truncation_warning_without_deprecated_arguments() -> None:
    tokenizer = MistralTokenizer.v7()

    def use_defaults() -> None:
        request = ChatCompletionRequest(messages=_MESSAGES)
        round_trip = ChatCompletionRequest.from_openai(**request.to_openai())
        ChatCompletionRequest.from_openai(_OPENAI_MESSAGES)
        InstructRequest(messages=_MESSAGES)
        tokenizer.encode_chat_completion(round_trip)
        tokenizer.encode_chat_completion(request, max_model_input_len=None)

    assert _record_warnings(use_defaults) == []

import io
import warnings
from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel
from pydantic_extra_types.language_code import LanguageAlpha2

import mistral_common.deprecation
from mistral_common.protocol.base import _DEPRECATED_FIELDS_MESSAGE, BaseCompletionRequest
from mistral_common.protocol.fim.request import FIMRequest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest, InstructRequest, ReasoningEffort
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from tests.unit.protocol.audio_helpers import audio_bytes

_MESSAGE = (
    "The following mistral-common request fields are deprecated and will be removed in 1.15.0:\n"
    "- ChatCompletionRequest: temperature, top_p, max_tokens, random_seed, truncate_for_context_length\n"
    "- FIMRequest: temperature, top_p, max_tokens, random_seed\n"
    "- InstructRequest: truncate_at_max_tokens\n"
    "- SpeechRequest: id, model, temperature, top_p, max_tokens, random_seed\n"
    "- TranscriptionRequest: id, model, temperature, top_p, max_tokens, random_seed, strict_audio_validation\n"
    "mistral-common does not use id, model, temperature, top_p, max_tokens or random_seed: pass them to your "
    "inference engine instead. Instead of truncate_for_context_length and truncate_at_max_tokens, truncate the "
    "conversation before encoding. strict_audio_validation has no effect.\n"
    "If you do not set these fields through mistral-common, you can ignore this warning."
)
_SAMPLING: dict[str, Any] = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 16, "random_seed": 7}
_UNUSED: dict[str, Any] = {"id": "1", "model": "m", **_SAMPLING}
_OPENAI_SAMPLING: dict[str, Any] = {"model": "my-model", "temperature": 0.2, "top_p": 0.9, "max_tokens": 16, "seed": 7}

_REQUIRED: dict[type[BaseModel], dict[str, Any]] = {
    ChatCompletionRequest: {"messages": [UserMessage(content="Hi")]},
    FIMRequest: {"prompt": "def"},
    InstructRequest: {"messages": [UserMessage(content="Hi")]},
    SpeechRequest: {"input": "Hi", "voice": "v"},
    TranscriptionRequest: {"audio": ""},
}
# Every deprecated request field, with a value to set it to (defaults included: any explicit value warns).
_DEPRECATED: list[tuple[type[BaseModel], str, Any]] = [
    *[(ChatCompletionRequest, field, value) for field, value in _SAMPLING.items()],
    (ChatCompletionRequest, "truncate_for_context_length", False),
    *[(FIMRequest, field, value) for field, value in _SAMPLING.items()],
    (InstructRequest, "truncate_at_max_tokens", None),
    *[(SpeechRequest, field, value) for field, value in _UNUSED.items()],
    *[(TranscriptionRequest, field, value) for field, value in _UNUSED.items()],
    (TranscriptionRequest, "strict_audio_validation", True),
]


@pytest.fixture(autouse=True)
def _reset_warnings() -> Generator[None, None, None]:
    mistral_common.deprecation._warned_keys.discard(_DEPRECATED_FIELDS_MESSAGE)
    yield
    mistral_common.deprecation._warned_keys.discard(_DEPRECATED_FIELDS_MESSAGE)


def _record_warnings(call: Callable[[], Any]) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    for warning in caught:
        assert warning.category is DeprecationWarning
        assert warning.filename == __file__
    return [str(warning.message) for warning in caught]


@pytest.fixture
def wav(audio_samples: np.ndarray) -> bytes:
    return audio_bytes(samples=audio_samples, fmt="wav")


@pytest.mark.parametrize(
    ("cls", "field", "value"), _DEPRECATED, ids=[f"{cls.__name__}.{field}" for cls, field, _ in _DEPRECATED]
)
def test_deprecated_field_warns_once_for_all_fields(cls: type[BaseModel], field: str, value: Any) -> None:
    requests: list[Any] = []
    assert _record_warnings(lambda: requests.append(cls(**_REQUIRED[cls], **{field: value}))) == [_MESSAGE]
    assert getattr(requests[0], field) == value

    # One warning per process for all deprecated fields: setting any of them again is silent.
    def set_every_deprecated_field() -> None:
        for other_cls, other_field, other_value in _DEPRECATED:
            other_cls.model_validate({**_REQUIRED[other_cls], other_field: other_value})

    assert _record_warnings(set_every_deprecated_field) == []


def test_no_warning_on_defaults_or_used_fields(wav: bytes) -> None:
    def build() -> None:
        ChatCompletionRequest(
            messages=[UserMessage(content="Hi")], parallel_tool_calls=True, reasoning_effort=ReasoningEffort.high
        )
        FIMRequest(prompt="def", suffix="return")
        SpeechRequest(input="Hi", voice="v")
        TranscriptionRequest(audio=wav, language=LanguageAlpha2("en"), target_streaming_delay_ms=None)
        BaseCompletionRequest(**_SAMPLING)

    assert _record_warnings(build) == []


def test_chat_from_openai_is_quiet() -> None:
    def build() -> ChatCompletionRequest:
        return ChatCompletionRequest.from_openai(
            messages=[{"role": "user", "content": "Hi"}], stream=False, n=1, **_OPENAI_SAMPLING
        )

    assert _record_warnings(build) == []
    request = build()
    assert request.model_dump(include={"model", *_SAMPLING}) == {"model": "my-model", **_SAMPLING}
    assert _record_warnings(lambda: ChatCompletionRequest.from_openai(**request.to_openai())) == []


def test_speech_from_openai_is_quiet() -> None:
    def build() -> SpeechRequest:
        return SpeechRequest.from_openai({"input": "Hi", "voice": "v", **_OPENAI_SAMPLING})

    assert _record_warnings(build) == []
    request = build()
    assert request.model_dump(include={"model", *_SAMPLING}) == {"model": "my-model", **_SAMPLING}
    assert _record_warnings(lambda: SpeechRequest.from_openai(request.to_openai())) == []


def test_transcription_from_openai_is_quiet(wav: bytes) -> None:
    def build() -> TranscriptionRequest:
        return TranscriptionRequest.from_openai({"file": io.BytesIO(wav), "language": "en", **_OPENAI_SAMPLING})

    assert _record_warnings(build) == []
    request = build()
    assert request.model_dump(include={"model", *_SAMPLING}) == {"model": "my-model", **_SAMPLING}
    assert _record_warnings(lambda: TranscriptionRequest.from_openai(request.to_openai())) == []

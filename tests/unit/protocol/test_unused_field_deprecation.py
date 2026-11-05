import io
import warnings
from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import pytest
from pydantic_extra_types.language_code import LanguageAlpha2

import mistral_common.deprecation
from mistral_common.protocol.base import BaseCompletionRequest
from mistral_common.protocol.fim.request import FIMRequest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from tests.unit.protocol.audio_helpers import audio_bytes

_UNUSED = (
    "is deprecated and will be removed in 1.15.0. "
    "mistral-common does not use it; pass it to your inference engine instead."
)
_SAMPLING: dict[str, Any] = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 16, "random_seed": 7}
_OPENAI_SAMPLING: dict[str, Any] = {"model": "my-model", "temperature": 0.2, "top_p": 0.9, "max_tokens": 16, "seed": 7}
_CLASSES = (ChatCompletionRequest, FIMRequest, SpeechRequest, TranscriptionRequest)


def _sampling_warnings(owner: str) -> list[str]:
    return [f"`{owner}.{field}` {_UNUSED}" for field in _SAMPLING]


@pytest.fixture(autouse=True)
def _reset_warnings() -> Generator[None, None, None]:
    keys = {message for cls in _CLASSES for message in cls._deprecated_fields.values()}
    mistral_common.deprecation._warned_keys.difference_update(keys)
    yield
    mistral_common.deprecation._warned_keys.difference_update(keys)


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


@pytest.mark.parametrize("via", ["init", "model_validate"])
def test_chat_sampling_fields_warn_once_each(via: str) -> None:
    values: dict[str, Any] = {"messages": [UserMessage(content="Hi")], **_SAMPLING}

    def build() -> Any:
        return ChatCompletionRequest(**values) if via == "init" else ChatCompletionRequest.model_validate(values)

    assert _record_warnings(build) == _sampling_warnings("ChatCompletionRequest")
    # Once per process, and the values are still applied.
    assert _record_warnings(build) == []
    assert build().model_dump(include=set(_SAMPLING)) == _SAMPLING


@pytest.mark.parametrize("value", [None, 0.7, 1.0])
def test_chat_explicit_value_warns(value: float | None) -> None:
    field = "max_tokens" if value is None else "temperature"
    values: dict[str, Any] = {field: value}
    warnings_ = _record_warnings(lambda: ChatCompletionRequest(messages=[UserMessage(content="Hi")], **values))
    assert warnings_ == [f"`ChatCompletionRequest.{field}` {_UNUSED}"]


def test_fim_sampling_fields_warn() -> None:
    assert _record_warnings(lambda: FIMRequest(prompt="def", **_SAMPLING)) == _sampling_warnings("FIMRequest")


def test_speech_unused_fields_warn() -> None:
    warnings_ = _record_warnings(lambda: SpeechRequest(input="Hi", voice="v", id="1", model="tts", **_SAMPLING))
    assert warnings_ == [
        f"`SpeechRequest.id` {_UNUSED}",
        f"`SpeechRequest.model` {_UNUSED}",
        *_sampling_warnings("SpeechRequest"),
    ]


def test_transcription_unused_fields_warn() -> None:
    def build() -> TranscriptionRequest:
        return TranscriptionRequest.model_validate(
            {"audio": "", "id": "1", "model": "stt", "strict_audio_validation": False, **_SAMPLING}
        )

    assert _record_warnings(build) == [
        f"`TranscriptionRequest.id` {_UNUSED}",
        f"`TranscriptionRequest.model` {_UNUSED}",
        *_sampling_warnings("TranscriptionRequest"),
        "`TranscriptionRequest.strict_audio_validation` is deprecated and will be removed in 1.15.0. It has no effect.",
    ]


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

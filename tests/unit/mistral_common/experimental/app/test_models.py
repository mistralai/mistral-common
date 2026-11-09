import math

import pytest
from pydantic import ValidationError

from mistral_common.experimental.app.models import OpenAIChatCompletionRequest, Settings
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.types import TimeoutType

_TIMEOUT_ERROR = "timeout must be a positive finite number of seconds"


def test_openai_request_accepts_null_assistant_content_and_converts() -> None:
    request = OpenAIChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Paris"}',
                            },
                        }
                    ],
                },
            ]
        }
    )

    ChatCompletionRequest.from_openai(**request.model_dump())


@pytest.fixture
def no_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENGINE_URL", raising=False)
    monkeypatch.delenv("TIMEOUT", raising=False)


def _settings_with_timeout(timeout: TimeoutType, from_env: bool, monkeypatch: pytest.MonkeyPatch) -> Settings:
    if from_env:
        monkeypatch.setenv("TIMEOUT", str(timeout))
        return Settings()
    return Settings(timeout=timeout)  # type: ignore[arg-type]


@pytest.mark.usefixtures("no_app_env")
def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.engine_url == "http://127.0.0.1:8080"
    assert settings.timeout == 60.0


@pytest.mark.usefixtures("no_app_env")
@pytest.mark.parametrize("from_env", [False, True], ids=["init", "env"])
@pytest.mark.parametrize(("timeout", "expected"), [(1, 1.0), (1.5, 1.5), ("30", 30.0), (" 2.5 ", 2.5)])
def test_settings_converts_timeout_to_float(
    timeout: TimeoutType, expected: float, from_env: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_with_timeout(timeout, from_env, monkeypatch)
    assert settings.timeout == expected
    assert isinstance(settings.timeout, float)


@pytest.mark.usefixtures("no_app_env")
@pytest.mark.parametrize("from_env", [False, True], ids=["init", "env"])
@pytest.mark.parametrize("timeout", [0, -5, math.inf, math.nan, "abc", "inf", "-1"])
def test_settings_rejects_invalid_timeout(
    timeout: TimeoutType, from_env: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValidationError, match=_TIMEOUT_ERROR):
        _settings_with_timeout(timeout, from_env, monkeypatch)

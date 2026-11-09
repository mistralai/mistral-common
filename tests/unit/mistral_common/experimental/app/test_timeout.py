import math

import pytest
import uvicorn
from click.testing import CliRunner
from fastapi import FastAPI
from pydantic import ValidationError

from mistral_common.experimental.app.main import cli, create_app
from mistral_common.experimental.app.models import Settings, get_settings
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.types import TimeoutType

_TEKKEN = MistralTokenizer._data_path() / "tekken_240911.json"
_INVALID_TIMEOUTS: list[TimeoutType] = [0, -5, math.inf, math.nan, "abc", "inf", "-1"]
_ERROR = "timeout must be a positive finite number of seconds"


@pytest.fixture(autouse=True)
def _no_timeout_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TIMEOUT", raising=False)


@pytest.mark.parametrize("timeout", _INVALID_TIMEOUTS)
def test_settings_rejects_invalid_timeout(timeout: TimeoutType) -> None:
    with pytest.raises(ValidationError, match=_ERROR):
        Settings(timeout=timeout)  # type: ignore[arg-type]


@pytest.mark.parametrize(("timeout", "expected"), [(1, 1.0), (1.5, 1.5), ("30", 30.0), (" 2.5 ", 2.5)])
def test_settings_converts_timeout_to_float(timeout: TimeoutType, expected: float) -> None:
    settings = Settings(timeout=timeout)  # type: ignore[arg-type]
    assert settings.timeout == expected
    assert isinstance(settings.timeout, float)


def test_settings_default_timeout() -> None:
    assert Settings().timeout == 60.0


@pytest.mark.parametrize(("env", "expected"), [("1.5", 1.5), ("30", 30.0)])
def test_settings_reads_timeout_env(env: str, expected: float, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIMEOUT", env)
    assert Settings().timeout == expected


@pytest.mark.parametrize("env", ["0", "-5", "inf", "nan", "abc"])
def test_settings_rejects_invalid_timeout_env(env: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIMEOUT", env)
    with pytest.raises(ValidationError, match=_ERROR):
        Settings()


@pytest.mark.parametrize("timeout", _INVALID_TIMEOUTS)
def test_create_app_rejects_invalid_timeout(timeout: TimeoutType) -> None:
    with pytest.raises(ValueError, match=_ERROR):
        create_app(str(_TEKKEN), timeout=timeout)


@pytest.mark.parametrize(("timeout", "expected"), [(1.5, 1.5), ("30", 30.0)])
def test_create_app_accepts_timeout(timeout: TimeoutType, expected: float) -> None:
    app = create_app(str(_TEKKEN), timeout=timeout)
    assert app.dependency_overrides[get_settings]().timeout == expected


@pytest.mark.parametrize("timeout", ["0", "-5", "inf", "nan"])
def test_serve_rejects_invalid_timeout(timeout: str, monkeypatch: pytest.MonkeyPatch) -> None:
    apps: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))

    result = CliRunner().invoke(cli, ["serve", str(_TEKKEN), "--timeout", timeout])

    assert result.exit_code == 2
    assert f"Invalid value for '--timeout': {_ERROR}" in result.output
    assert apps == []


def test_serve_rejects_non_numeric_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    apps: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))

    result = CliRunner().invoke(cli, ["serve", str(_TEKKEN), "--timeout", "abc"])

    assert result.exit_code == 2
    assert "Invalid value for '--timeout': 'abc' is not a valid float." in result.output
    assert apps == []


@pytest.mark.parametrize(("timeout", "expected"), [("1", 1.0), ("1.5", 1.5)])
def test_serve_accepts_positive_timeout(timeout: str, expected: float, monkeypatch: pytest.MonkeyPatch) -> None:
    apps: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))

    result = CliRunner().invoke(cli, ["serve", str(_TEKKEN), "--timeout", timeout])

    assert result.exit_code == 0, result.output
    assert [app.dependency_overrides[get_settings]().timeout for app in apps] == [expected]


def test_serve_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    apps: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))

    result = CliRunner().invoke(cli, ["serve", str(_TEKKEN)])

    assert result.exit_code == 0, result.output
    assert [app.dependency_overrides[get_settings]().timeout for app in apps] == [60.0]

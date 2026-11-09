import math
from unittest.mock import patch

import pytest
import uvicorn
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mistral_common.experimental.app.main import cli, create_app
from mistral_common.experimental.app.models import get_settings
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.types import TimeoutType

_TEKKEN = MistralTokenizer._data_path() / "tekken_240911.json"
_TIMEOUT_ERROR = "timeout must be a positive finite number of seconds"


@pytest.fixture(autouse=True)
def _no_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENGINE_URL", raising=False)
    monkeypatch.delenv("TIMEOUT", raising=False)


@pytest.fixture
def served_apps(monkeypatch: pytest.MonkeyPatch) -> list[FastAPI]:
    apps: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))
    return apps


def test_create_app_default_engine_url_matches_serve(served_apps: list[FastAPI]) -> None:
    result = CliRunner().invoke(cli, ["serve", str(_TEKKEN)])

    assert result.exit_code == 0, result.output
    assert served_apps[0].dependency_overrides[get_settings]().engine_url == "http://127.0.0.1:8080"
    assert create_app(_TEKKEN).dependency_overrides[get_settings]().engine_url == "http://127.0.0.1:8080"


def test_create_app_default_engine_url_reaches_engine() -> None:
    app = create_app(_TEKKEN)
    tokenizer = app.dependency_overrides[get_settings]().tokenizer.instruct_tokenizer.tokenizer

    with patch("mistral_common.experimental.app.routers.requests.post") as post:
        post.return_value.status_code = 200
        post.return_value.json.return_value = {"tokens": tokenizer.encode("Hello", bos=False, eos=True)}
        response = TestClient(app).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200, response.json()
    assert response.json()["content"] == "Hello"
    assert post.call_args.args == ("http://127.0.0.1:8080/completions",)


@pytest.mark.parametrize("timeout", [0, -5, math.inf, math.nan, "abc", "inf", "-1"])
def test_create_app_rejects_invalid_timeout(timeout: TimeoutType) -> None:
    with pytest.raises(ValueError, match=_TIMEOUT_ERROR):
        create_app(str(_TEKKEN), timeout=timeout)


@pytest.mark.parametrize(("timeout", "expected"), [(1.5, 1.5), ("30", 30.0)])
def test_create_app_accepts_timeout(timeout: TimeoutType, expected: float) -> None:
    app = create_app(str(_TEKKEN), timeout=timeout)
    assert app.dependency_overrides[get_settings]().timeout == expected


@pytest.mark.parametrize(
    ("timeout", "error"),
    [
        ("0", _TIMEOUT_ERROR),
        ("-5", _TIMEOUT_ERROR),
        ("inf", _TIMEOUT_ERROR),
        ("nan", _TIMEOUT_ERROR),
        ("abc", "'abc' is not a valid float."),
    ],
)
def test_serve_rejects_invalid_timeout(timeout: str, error: str, served_apps: list[FastAPI]) -> None:
    result = CliRunner().invoke(cli, ["serve", str(_TEKKEN), "--timeout", timeout])

    assert result.exit_code == 2
    assert f"Invalid value for '--timeout': {error}" in result.output
    assert served_apps == []


@pytest.mark.parametrize(
    ("timeout_args", "expected"), [(["--timeout", "1"], 1.0), (["--timeout", "1.5"], 1.5), ([], 60.0)]
)
def test_serve_timeout(timeout_args: list[str], expected: float, served_apps: list[FastAPI]) -> None:
    result = CliRunner().invoke(cli, ["serve", str(_TEKKEN), *timeout_args])

    assert result.exit_code == 0, result.output
    assert [app.dependency_overrides[get_settings]().timeout for app in served_apps] == [expected]

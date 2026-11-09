from unittest.mock import patch

import pytest
import uvicorn
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mistral_common.experimental.app.main import cli, create_app
from mistral_common.experimental.app.models import Settings, get_settings
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

_TEKKEN = MistralTokenizer._data_path() / "tekken_240911.json"


@pytest.fixture(autouse=True)
def _no_engine_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENGINE_URL", raising=False)


def test_settings_default_engine_url() -> None:
    assert Settings().engine_url == "http://127.0.0.1:8080"


def test_create_app_default_engine_url_matches_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    apps: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))

    result = CliRunner().invoke(cli, ["serve", str(_TEKKEN)])

    assert result.exit_code == 0, result.output
    assert apps[0].dependency_overrides[get_settings]().engine_url == "http://127.0.0.1:8080"
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

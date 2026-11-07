import pytest
import uvicorn
from click.testing import CliRunner
from fastapi import FastAPI
from pydantic import ValidationError

from mistral_common.experimental.app.main import cli, create_app
from mistral_common.experimental.app.models import Settings, get_settings
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

_TEKKEN = MistralTokenizer._data_path() / "tekken_240911.json"


@pytest.mark.parametrize("timeout", [0, -5])
def test_settings_rejects_non_positive_timeout(timeout: int) -> None:
    with pytest.raises(ValidationError, match="timeout\n  Input should be greater than 0"):
        Settings(timeout=timeout)


@pytest.mark.parametrize("timeout", [0, -5])
def test_create_app_rejects_non_positive_timeout(timeout: int) -> None:
    with pytest.raises(ValidationError, match="timeout\n  Input should be greater than 0"):
        create_app(str(_TEKKEN), timeout=timeout)


@pytest.mark.parametrize("timeout", ["0", "-5"])
def test_serve_rejects_non_positive_timeout(timeout: str, monkeypatch: pytest.MonkeyPatch) -> None:
    apps: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))

    result = CliRunner().invoke(cli, ["serve", str(_TEKKEN), "--timeout", timeout])

    assert result.exit_code == 2
    assert f"Invalid value for '--timeout': {timeout} is not in the range x>=1." in result.output
    assert apps == []


def test_serve_accepts_positive_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    apps: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))

    result = CliRunner().invoke(cli, ["serve", str(_TEKKEN), "--timeout", "1"])

    assert result.exit_code == 0, result.output
    assert [app.dependency_overrides[get_settings]().timeout for app in apps] == [1]

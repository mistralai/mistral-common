import pytest

from mistral_common.utils.download import (
    DEFAULT_DOWNLOAD_TIMEOUT_S,
    DOWNLOAD_TIMEOUT_ENV_KEY,
    resolve_download_timeout,
    resolve_env_download_timeout,
    validate_download_timeout,
)


def test_resolve_env_download_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DOWNLOAD_TIMEOUT_ENV_KEY, raising=False)
    assert resolve_env_download_timeout() == DEFAULT_DOWNLOAD_TIMEOUT_S == 10.0


def test_resolve_env_download_timeout_empty_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DOWNLOAD_TIMEOUT_ENV_KEY, "  ")
    assert resolve_env_download_timeout() == 10.0


@pytest.mark.parametrize("value", ["30", "2.5", "1e2"])
def test_resolve_env_download_timeout_accepts_numbers(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(DOWNLOAD_TIMEOUT_ENV_KEY, value)
    assert resolve_env_download_timeout() == float(value)


@pytest.mark.parametrize("value", ["invalid", "0", "-1", "inf", "-inf", "nan"])
def test_resolve_env_download_timeout_rejects_invalid(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(DOWNLOAD_TIMEOUT_ENV_KEY, value)

    with pytest.raises(ValueError, match="expected a positive finite number of seconds"):
        resolve_env_download_timeout()


@pytest.mark.parametrize("timeout", [5, 2.5, "7", "0.5"])
def test_resolve_download_timeout_converts_numbers_to_float(timeout: float | int | str) -> None:
    assert resolve_download_timeout(timeout) == float(timeout)
    assert isinstance(resolve_download_timeout(timeout), float)


def test_resolve_download_timeout_none_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DOWNLOAD_TIMEOUT_ENV_KEY, raising=False)
    assert resolve_download_timeout(None) == 10.0
    monkeypatch.setenv(DOWNLOAD_TIMEOUT_ENV_KEY, "42")
    assert resolve_download_timeout(None) == 42.0


@pytest.mark.parametrize("timeout", [0, -1, 0.0, -0.5, float("inf"), float("nan"), "abc", object()])
def test_resolve_download_timeout_rejects_invalid(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout must be a positive finite number, got timeout="):
        resolve_download_timeout(timeout)  # type: ignore[arg-type]


def test_validate_download_timeout() -> None:
    assert validate_download_timeout(3, error_message="bad") == 3.0
    with pytest.raises(ValueError, match="bad"):
        validate_download_timeout("x", error_message="bad")

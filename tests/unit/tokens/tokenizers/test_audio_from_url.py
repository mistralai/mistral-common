from unittest.mock import Mock, patch

import pytest
import requests

from mistral_common.tokens.tokenizers.audio import (
    _AUDIO_DOWNLOAD_TIMEOUT_ENV_KEY,
    _DEFAULT_AUDIO_DOWNLOAD_TIMEOUT_S,
    Audio,
    _resolve_env_audio_download_timeout,
)


def _wav_bytes() -> bytes:
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16_000, format="WAV")
    return buf.getvalue()


def test_from_url_passes_timeout_and_headers() -> None:
    response = Mock(content=_wav_bytes())
    with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get", return_value=response) as mock_get:
        audio = Audio.from_url("https://example.com/sample.wav", timeout=5)

    assert mock_get.call_args.kwargs["timeout"] == 5
    assert "User-Agent" in mock_get.call_args.kwargs["headers"]
    assert audio.sampling_rate == 16_000


def test_from_url_default_timeout_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_AUDIO_DOWNLOAD_TIMEOUT_ENV_KEY, "42")
    assert _resolve_env_audio_download_timeout() == 42.0

    response = Mock(content=_wav_bytes())
    with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get", return_value=response) as mock_get:
        Audio.from_url("https://example.com/sample.wav")
    assert mock_get.call_args.kwargs["timeout"] == 42.0


def test_from_url_unset_env_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_AUDIO_DOWNLOAD_TIMEOUT_ENV_KEY, raising=False)
    assert _resolve_env_audio_download_timeout() == _DEFAULT_AUDIO_DOWNLOAD_TIMEOUT_S


def test_from_url_invalid_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_AUDIO_DOWNLOAD_TIMEOUT_ENV_KEY, "not-a-number")
    with pytest.raises(ValueError, match=_AUDIO_DOWNLOAD_TIMEOUT_ENV_KEY):
        _resolve_env_audio_download_timeout()


def test_from_url_timeout_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_AUDIO_DOWNLOAD_TIMEOUT_ENV_KEY, raising=False)
    with (
        patch(
            "mistral_common.tokens.tokenizers.audio._requests_lib.get",
            side_effect=requests.Timeout("too slow"),
        ),
        pytest.raises(ValueError, match=r"timed out after 10\.0 seconds") as exc_info,
    ):
        Audio.from_url("https://example.com/sample.wav")

    assert _AUDIO_DOWNLOAD_TIMEOUT_ENV_KEY in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, requests.Timeout)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), "10"])
def test_from_url_invalid_timeout_raises(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout must be a positive finite number"):
        Audio.from_url("https://example.com/sample.wav", timeout=timeout)  # type: ignore[arg-type]

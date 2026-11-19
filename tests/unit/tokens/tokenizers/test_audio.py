from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
import soundfile as sf
from pydantic import ValidationError

from mistral_common.exceptions import AudioDecodeException
from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURL, AudioURLChunk
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
    TranscriptionFormat,
)
from mistral_common.types import TimeoutType

_AUDIO = 24
_BEGIN_AUDIO = 25


def _encode(num_samples: int, chunk_length_s: float | None) -> tuple[np.ndarray, list[int]]:
    # Voxtral audio settings: 16 kHz, 12.5 audio tokens per second, hop 160 and window 400.
    # One audio token covers 16_000 / 12.5 / 160 = 8 spectrogram frames.
    audio_config = AudioConfig(
        sampling_rate=16_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(num_mel_bins=128, window_size=400, hop_length=160),
        chunk_length_s=chunk_length_s,
    )
    special_ids = SpecialAudioIDs(
        audio=_AUDIO, begin_audio=_BEGIN_AUDIO, streaming_pad=None, text_to_audio=None, audio_to_text=None
    )
    audio = Audio(audio_array=np.ones(num_samples, dtype=np.float32), sampling_rate=16_000, format="wav")

    encoding = AudioEncoder(audio_config, special_ids).encode_audio(audio)

    assert encoding.audio is not None
    assert encoding.audio.sampling_rate == 16_000
    return encoding.audio.audio_array, encoding.tokens


@pytest.mark.parametrize(
    ("num_samples", "expected_samples", "expected_audio_tokens"),
    [
        (16_000, 480_000, 375),
        (479_999, 480_000, 375),
        (480_000, 480_000, 375),
        (480_001, 960_000, 750),
        (960_000, 960_000, 750),
    ],
    ids=["shorter_than_one_chunk", "one_sample_short", "one_chunk", "one_sample_over", "two_chunks"],
)
def test_encode_audio_pads_to_whole_chunks(num_samples: int, expected_samples: int, expected_audio_tokens: int) -> None:
    # 30 s chunks of 480_000 samples, each worth 30 * 12.5 = 375 audio tokens.
    audio_array, tokens = _encode(num_samples, chunk_length_s=30.0)

    assert audio_array.shape == (expected_samples,)
    np.testing.assert_array_equal(audio_array[:num_samples], 1.0)
    np.testing.assert_array_equal(audio_array[num_samples:], 0.0)
    assert tokens == [_BEGIN_AUDIO] + [_AUDIO] * expected_audio_tokens


@pytest.mark.parametrize(
    ("num_samples", "expected_samples", "expected_audio_tokens"),
    [
        # shorter than one window: padded to the 400-sample window, which gives 2 frames
        (160, 400, 1),
        (399, 400, 1),
        (400, 400, 1),
        # a partial hop at the end does not add a frame (as with a centered STFT without its last frame)
        (1_279, 1_279, 1),
        (1_280, 1_280, 1),
        (1_439, 1_439, 1),
        (1_440, 1_440, 2),
        (1_441, 1_441, 2),
        (16_000, 16_000, 13),
        (16_159, 16_159, 13),
    ],
)
def test_encode_audio_without_chunks_counts_spectrogram_frames(
    num_samples: int, expected_samples: int, expected_audio_tokens: int
) -> None:
    audio_array, tokens = _encode(num_samples, chunk_length_s=None)

    assert audio_array.shape == (expected_samples,)
    np.testing.assert_array_equal(audio_array[num_samples:], 0.0)
    assert tokens == [_BEGIN_AUDIO] + [_AUDIO] * expected_audio_tokens


def _sin_wave(sampling_rate: int, duration: float) -> np.ndarray:
    return np.sin(np.ones([int(duration * sampling_rate)]))


def _make_dummy_base64() -> str:
    """Create a minimal valid base64-encoded WAV for testing."""
    audio = Audio(audio_array=np.zeros(1600, dtype=np.float32), sampling_rate=16000, format="wav")
    return audio.to_base64("wav")


def _write_wav(path: Path, audio_array: np.ndarray, sampling_rate: int) -> bytes:
    with sf.SoundFile(str(path), "w", samplerate=sampling_rate, channels=1) as f:
        f.write(audio_array)
    return path.read_bytes()


def test_audio_config_rejects_sampling_rate_below_frame_rate() -> None:
    encoding_config = AudioSpectrogramConfig(num_mel_bins=80, hop_length=160, window_size=400)
    with pytest.raises(AssertionError, match="sampling_rate must be >= frame_rate"):
        AudioConfig(sampling_rate=1, frame_rate=2.0, encoding_config=encoding_config)


def test_audio_config_rejects_audio_length_per_tok_of_zero() -> None:
    encoding_config = AudioSpectrogramConfig(num_mel_bins=80, hop_length=160, window_size=400)
    # sampling_rate >= frame_rate holds, but sampling_rate / frame_rate < hop_length floors
    # audio_length_per_tok to 0, which previously made num_audio_tokens raise ZeroDivisionError.
    with pytest.raises(AssertionError, match="must be >= hop_length"):
        AudioConfig(sampling_rate=200, frame_rate=2.0, encoding_config=encoding_config)


@pytest.mark.parametrize(
    ("sampling_rate", "frame_rate", "streaming_kwargs", "match"),
    [
        pytest.param(
            16_000,
            12.5,
            {"transcription_delay_ms": 480.0},
            r"transcription_delay_ms=480\.0 must be None",
            id="non-streaming-rejects-delay",
        ),
        pytest.param(
            16_000,
            12.5,
            {
                "transcription_format": TranscriptionFormat.STREAMING,
                "transcription_delay_ms": 480.0,
                "streaming_look_ahead_ms": 2.5,
                "streaming_look_back_ms": 52.5,
                "streaming_n_left_pad_tokens": 16,
                "chunk_length_s": 30.0,
            },
            r"chunk_length_s=30\.0 cannot be set in streaming",
            id="streaming-rejects-chunk-length",
        ),
    ],
)
def test_audio_config_rejects_inconsistent_streaming_fields(
    sampling_rate: int, frame_rate: float, streaming_kwargs: dict, match: str
) -> None:
    encoding_config = AudioSpectrogramConfig(num_mel_bins=128, hop_length=160, window_size=400)
    with pytest.raises(AssertionError, match=match):
        AudioConfig(
            sampling_rate=sampling_rate, frame_rate=frame_rate, encoding_config=encoding_config, **streaming_kwargs
        )


@pytest.mark.parametrize(
    ("rate", "delay", "num_delay_tokens"),
    [
        (12.5, 360, -1),
        (12.5, 560, 7),
        (12.5, 960, 12),
        (50, 360, 18),
        (50, 560, 28),
        (50, 960, 48),
        (30, 960, -1),
    ],
    ids=[
        "rate-12.5-delay-360-invalid",
        "rate-12.5-delay-560",
        "rate-12.5-delay-960",
        "rate-50-delay-360",
        "rate-50-delay-560",
        "rate-50-delay-960",
        "rate-30-delay-960-invalid",
    ],
)
def test_audio_config_delay(rate: float, delay: int, num_delay_tokens: int) -> None:
    spectrogram_config = AudioSpectrogramConfig(num_mel_bins=128, hop_length=160, window_size=400)

    def audio_config_fn() -> AudioConfig:
        return AudioConfig(
            sampling_rate=16_000,
            frame_rate=rate,
            encoding_config=spectrogram_config,
            transcription_delay_ms=delay,
            transcription_format=TranscriptionFormat.STREAMING,
            streaming_look_ahead_ms=2.5,
            streaming_look_back_ms=52.5,
            streaming_n_left_pad_tokens=16,
        )

    if num_delay_tokens == -1:
        with pytest.raises(AssertionError, match="must be a multiple"):
            audio_config_fn()
    else:
        assert audio_config_fn().get_num_delay_tokens() == num_delay_tokens


def test_audio_resample() -> None:
    sampling_rate = 44_000
    original_array = _sin_wave(sampling_rate, 1)

    audio = Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
        format="wav",
    )

    audio.resample(sampling_rate)
    assert np.allclose(audio.audio_array, original_array, atol=1e-5)

    audio.resample(sampling_rate // 2)
    assert audio.sampling_rate == sampling_rate // 2
    assert len(audio.audio_array) == len(original_array) // 2
    # fmt: off
    expected_resampled_array = [0.622, 0.907, 0.802, 0.870, 0.819, 0.860, 0.826, 0.854, 0.831, 0.850]
    # fmt: on
    assert np.allclose(audio.audio_array[:10], expected_resampled_array, atol=1e-3)


def test_audio_resample_same_rate_is_noop() -> None:
    audio = Audio(audio_array=_sin_wave(16_000, 0.1), sampling_rate=16_000, format="wav")
    before = audio.audio_array.copy()
    audio.resample(16_000)
    np.testing.assert_array_equal(audio.audio_array, before)
    assert audio.sampling_rate == 16_000


def test_from_file(tmp_path: Path) -> None:
    sampling_rate = 44100
    original_array = _sin_wave(sampling_rate, 1)

    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path, original_array, sampling_rate)

    audio_local = Audio.from_file(str(wav_path))
    assert isinstance(audio_local, Audio)
    assert np.allclose(audio_local.audio_array, original_array, atol=1e-5)
    assert audio_local.sampling_rate == sampling_rate

    # file:// URIs are accepted and stripped
    audio_uri = Audio.from_file(wav_path.as_uri())
    assert np.allclose(audio_uri.audio_array, original_array, atol=1e-5)


def test_from_file_missing_path() -> None:
    with pytest.raises(FileNotFoundError, match=r"file='invalid_file_path.wav' does not exist"):
        Audio.from_file("invalid_file_path.wav")


def test_from_bytes_rejects_multichannel_strict() -> None:
    import io

    stereo = np.zeros((1600, 2), dtype=np.float32)
    with io.BytesIO() as buf:
        sf.write(buf, stereo, 16_000, format="WAV")
        wav_bytes = buf.getvalue()

    with pytest.raises(AudioDecodeException, match=r"^Could not decode the audio"):
        Audio.from_bytes(wav_bytes, strict=True)

    # non-strict averages the channels down to mono
    audio = Audio.from_bytes(wav_bytes, strict=False)
    assert audio.audio_array.ndim == 1
    assert audio.audio_array.shape == (1600,)


def test_from_bytes_rejects_undecodable() -> None:
    with pytest.raises(Exception):
        Audio.from_bytes(b"<html>this is not audio</html>")


def test_from_url_transport_failure() -> None:
    with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get") as mock_get:
        mock_get.side_effect = requests.RequestException("connection failed")
        with pytest.raises(
            AudioDecodeException,
            match=r"^Could not download the audio from URL: https://example.com/invalid_audio.wav",
        ):
            Audio.from_url("https://example.com/invalid_audio.wav", strict=True, timeout=None)


def test_from_url_invalid_content() -> None:
    with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html>this is not audio</html>"
        mock_get.return_value = mock_response
        with pytest.raises(
            AudioDecodeException,
            match=r"^Could not decode the audio",
        ):
            Audio.from_url("https://example.com", strict=True, timeout=None)


def test_from_url_valid_content(tmp_path: Path) -> None:
    sampling_rate = 16_000
    audio_array = _sin_wave(sampling_rate, 1)
    wav_bytes = _write_wav(tmp_path / "sample.wav", audio_array, sampling_rate)

    with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = wav_bytes
        mock_get.return_value = mock_response
        audio_url = Audio.from_url("https://fake-audio-url.com/sample.wav", strict=False, timeout=None)
        assert isinstance(audio_url, Audio)
        assert np.allclose(audio_url.audio_array, audio_array, atol=1e-5)
        assert audio_url.sampling_rate == sampling_rate
        # timeout=None falls back to the default download timeout
        assert mock_get.call_args.kwargs.get("timeout") == 10.0


def test_from_url_passes_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from mistral_common.utils.download import DOWNLOAD_TIMEOUT_ENV_KEY

    monkeypatch.setenv(DOWNLOAD_TIMEOUT_ENV_KEY, "25")
    with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get") as mock_get:
        mock_get.side_effect = requests.RequestException("connection failed")
        with pytest.raises(AudioDecodeException, match=r"^Could not download the audio from URL"):
            Audio.from_url("https://example.com/a.wav", strict=True, timeout=None)
    assert mock_get.call_args.kwargs.get("timeout") == 25.0

    with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get") as mock_get:
        mock_get.side_effect = requests.RequestException("connection failed")
        with pytest.raises(AudioDecodeException, match=r"^Could not download the audio from URL"):
            Audio.from_url("https://example.com/a.wav", strict=True, timeout=3)
    assert mock_get.call_args.kwargs.get("timeout") == 3.0


@pytest.mark.parametrize("timeout", [0, -2.5, "nope"], ids=["zero", "negative", "not-a-number"])
def test_from_url_rejects_invalid_timeout(timeout: TimeoutType) -> None:
    with (
        patch("mistral_common.tokens.tokenizers.audio._requests_lib.get") as mock_get,
        pytest.raises(ValueError, match="timeout must be a positive finite number"),
    ):
        Audio.from_url("https://example.com/a.wav", strict=True, timeout=timeout)
    mock_get.assert_not_called()


@pytest.mark.parametrize("prefix", [True, False], ids=["with-data-url-prefix", "bare-base64"])
@pytest.mark.parametrize("format", ["wav", "mp3"], ids=["lossless-wav", "lossy-mp3"])
def test_audio_base64_round_trip(format: str, prefix: bool) -> None:
    sampling_rate = 16_000
    original_array = _sin_wave(sampling_rate, 3.3)

    audio = Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
        format=format,
    )

    base64_str = audio.to_base64(format, prefix)
    new_audio = Audio.from_base64(base64_str)

    if prefix:
        assert base64_str.startswith(f"data:audio/{format};base64,")

    assert audio.sampling_rate == new_audio.sampling_rate
    if format == "wav":
        # lossless: sample values survive the round trip
        assert np.allclose(audio.audio_array, new_audio.audio_array, atol=1e-5)
    else:
        # lossy: codec-specific accuracy criterion (RMSE below 5e-3)
        rmse = float(np.sqrt(np.mean((audio.audio_array - new_audio.audio_array) ** 2)))
        assert rmse < 5e-3


def test_from_base64_rejects_invalid_base64() -> None:
    with pytest.raises(AudioDecodeException, match=r"^Could not decode the audio"):
        Audio.from_base64("data:audio/wav;base64,invalid_base64_string")


def test_from_base64_rejects_non_audio_payload() -> None:
    import base64

    payload = base64.b64encode(b"<html>this is not audio</html>").decode()
    with pytest.raises(AudioDecodeException, match=r"^Could not decode the audio"):
        Audio.from_base64(payload)


def _audio_encoder() -> AudioEncoder:
    audio_config = AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(num_mel_bins=128, window_size=400, hop_length=160),
    )
    special_ids = SpecialAudioIDs(
        audio=_AUDIO, begin_audio=_BEGIN_AUDIO, streaming_pad=None, text_to_audio=None, audio_to_text=None
    )
    return AudioEncoder(audio_config, special_ids)


def _random_audio_chunk(duration: float, sampling_rate: int = 24_000) -> AudioChunk:
    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[int(duration * sampling_rate)]),
        sampling_rate=sampling_rate,
        format="wav",
    )
    return AudioChunk(input_audio=audio.to_base64("wav"))


def test_audio_encoder_chunk_dispatch_inline_base64() -> None:
    chunk = _random_audio_chunk(1.7)
    encoding = _audio_encoder()(chunk)
    assert encoding.audio is not None
    assert isinstance(chunk.input_audio, str)
    source = Audio.from_base64(chunk.input_audio)
    assert np.allclose(encoding.audio.audio_array, source.audio_array, atol=1e-3)
    assert encoding.tokens[:1] == [_BEGIN_AUDIO]


def test_audio_encoder_chunk_dispatch_inline_bytes() -> None:
    import base64

    chunk = _random_audio_chunk(1.7)
    assert isinstance(chunk.input_audio, str)
    assert isinstance(chunk.input_audio, str)
    bytes_chunk = AudioChunk(input_audio=base64.b64decode(chunk.input_audio))
    encoding = _audio_encoder()(bytes_chunk)
    assert encoding.audio is not None
    source = Audio.from_base64(chunk.input_audio)
    assert np.allclose(encoding.audio.audio_array, source.audio_array, atol=1e-3)


def test_audio_encoder_chunk_dispatch_local_path(tmp_path: Path) -> None:
    audio = Audio(audio_array=_sin_wave(24_000, 1.0), sampling_rate=24_000, format="wav")
    wav_path = tmp_path / "sample.wav"
    _write_wav(wav_path, audio.audio_array, audio.sampling_rate)

    for url in (str(wav_path), wav_path.as_uri()):
        encoding = _audio_encoder()(AudioURLChunk(audio_url=url))
        assert encoding.audio is not None
        assert np.allclose(encoding.audio.audio_array, audio.audio_array, atol=1e-5)


def test_audio_encoder_chunk_dispatch_http(tmp_path: Path) -> None:
    audio = Audio(audio_array=_sin_wave(24_000, 1.0), sampling_rate=24_000, format="wav")
    wav_bytes = _write_wav(tmp_path / "sample.wav", audio.audio_array, audio.sampling_rate)

    for scheme in ("http", "https"):
        with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = wav_bytes
            mock_get.return_value = mock_response
            encoding = _audio_encoder()(AudioURLChunk(audio_url=AudioURL(url=f"{scheme}://example.com/audio.wav")))
        assert encoding.audio is not None
        assert np.allclose(encoding.audio.audio_array, audio.audio_array, atol=1e-5)


def test_audio_encoder_chunk_dispatch_base64_and_data_url() -> None:
    audio = Audio(audio_array=_sin_wave(24_000, 1.0), sampling_rate=24_000, format="wav")
    base64_str = audio.to_base64("wav")
    for url in (base64_str, f"data:audio/wav;base64,{base64_str}"):
        encoding = _audio_encoder()(AudioURLChunk(audio_url=url))
        assert encoding.audio is not None
        assert np.allclose(encoding.audio.audio_array, audio.audio_array, atol=1e-5)


def test_audio_encoder_chunk_dispatch_transport_failure() -> None:
    with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get") as mock_get:
        mock_get.side_effect = requests.RequestException("connection failed")
        with pytest.raises(
            AudioDecodeException,
            match=r"^Could not download the audio from URL: https://example.com/invalid_audio.wav",
        ):
            _audio_encoder()(AudioURLChunk(audio_url="https://example.com/invalid_audio.wav"))


def test_audio_encoder_chunk_dispatch_invalid_base64() -> None:
    with pytest.raises(AudioDecodeException, match=r"^Could not decode the audio"):
        _audio_encoder()(AudioURLChunk(audio_url="data:audio/wav;base64,invalid_base64_string"))


def test_audio_encoder_chunk_dispatch_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match=r"file='invalid_file_path.wav' does not exist"):
        _audio_encoder()(AudioURLChunk(audio_url="file://invalid_file_path.wav"))


def test_audio_encoder_rejects_unsupported_chunk_type() -> None:
    from mistral_common.protocol.instruct.chunk import TextChunk

    with pytest.raises(ValueError, match="Unsupported content type"):
        _audio_encoder()(TextChunk(text="hello"))  # type: ignore[arg-type]


class TestAudioChunkInput:
    def test_audiochunk_with_openai_dict(self) -> None:
        b64 = _make_dummy_base64()
        chunk = AudioChunk.model_validate({"input_audio": {"data": b64, "format": "wav"}})
        assert chunk.input_audio == b64

    def test_audiochunk_with_str_new_api(self) -> None:
        chunk = AudioChunk(input_audio="base64data")
        assert chunk.input_audio == "base64data"

    def test_audiochunk_with_bytes_new_api(self) -> None:
        chunk = AudioChunk(input_audio=b"raw_bytes")
        assert chunk.input_audio == b"raw_bytes"

    @pytest.mark.parametrize("empty_audio", ["", "   ", b""], ids=["empty-str", "whitespace-str", "empty-bytes"])
    def test_audiochunk_empty_input_raises_validation_error(self, empty_audio: str | bytes) -> None:
        with pytest.raises(ValidationError, match="should not be empty"):
            AudioChunk(input_audio=empty_audio)


def test_transcription_request_with_openai_dict() -> None:
    b64 = _make_dummy_base64()
    req = TranscriptionRequest.model_validate(
        {"audio": {"data": b64, "format": "wav"}, "language": None, "target_streaming_delay_ms": None}
    )
    assert req.audio == b64


def test_from_base64_strips_data_url_with_non_word_subtype() -> None:
    r"""Strip the data URL prefix for an audio subtype containing `-`."""
    b64 = _make_dummy_base64()
    audio = Audio.from_base64(f"data:audio/x-wav;base64,{b64}")
    assert audio.format == "wav"


def test_from_base64_rejects_image_payload() -> None:
    r"""Reject valid base64 of a valid image passed as audio."""
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    png_b64 = base64.b64encode(buffer.getvalue()).decode()
    with pytest.raises(AudioDecodeException, match=r"^Could not decode the audio"):
        Audio.from_base64(f"data:audio/wav;base64,{png_b64}")


def test_from_base64_rejects_malformed_data_url() -> None:
    r"""Reject a malformed data URL before attempting to decode."""
    with pytest.raises(ValueError, match="Invalid base64 data URL"):
        Audio.from_base64("data:text/plain,hello")

import numpy as np
import pytest

from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)

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

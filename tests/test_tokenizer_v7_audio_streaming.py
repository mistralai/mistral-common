import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import (
    OFFLINE_STREAMING_BUFFER_TOKENS,
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
    TranscriptionFormat,
)
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV7,
)
from mistral_common.tokens.tokenizers.mistral import (
    MistralTokenizer,
    load_audio_encoder,
)
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab

_audio_spectrogram_config = {
    "num_mel_bins": 128,
    "hop_length": 160,
    "window_size": 400,
}


def add_special_tokens(
    special_tokens: list[dict[str, Any]], tokens: list[str], start_idx: int, is_control: bool = True
) -> None:
    """Insert `tokens` into `special_tokens` starting at `start_idx`.

    Raises if the target slot is already populated to avoid silent overrides.
    """
    for offset, token in enumerate(tokens):
        idx = start_idx + offset
        try:
            token_entry = special_tokens[idx]
        except IndexError as exc:
            raise ValueError(f"special_tokens is missing index {idx}") from exc

        existing = token_entry.get("token_str")
        placeholder = f"<SPECIAL_{idx}>"
        if existing not in (None, placeholder):
            raise ValueError(f"special_tokens[{idx}] already set to {existing}")

        token_entry["token_str"] = token
        token_entry["is_control"] = is_control


def load_audio_streaming() -> Tekkenizer:
    special_tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v7, add_audio=True)
    for i in range(35, 39):
        special_tokens += [SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True)]

    special_tokens += [SpecialTokenInfo(rank=39, token_str=f"[STREAMING_PAD]", is_control=True)]
    special_tokens += [SpecialTokenInfo(rank=40, token_str=f"[STREAMING_WORD]", is_control=True)]
    tokenizer = Tekkenizer(
        quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v7,
    )

    audio_config = AudioConfig(
        sampling_rate=16_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(
            num_mel_bins=128,
            window_size=400,
            hop_length=160,
        ),
        transcription_delay_ms=960.0,
        transcription_format=TranscriptionFormat.STREAMING,
    )
    tokenizer._audio_config = audio_config
    return tokenizer

@pytest.fixture(scope="session")
def tokenizer() -> InstructTokenizerV7:
    tekkenizer = load_audio_streaming()
    assert tekkenizer.audio is not None
    audio_encoder = load_audio_encoder(tekkenizer.audio, tekkenizer)
    return InstructTokenizerV7(tekkenizer, image_encoder=None, audio_encoder=audio_encoder)

def test_special_audio_streaming_tokens(tokenizer: InstructTokenizerV7) -> None:
    assert tokenizer.tokenizer.get_special_token(SpecialTokens.streaming_pad.value) == 39
    assert tokenizer.tokenizer.get_special_token(SpecialTokens.streaming_word.value) == 40

    decoded_ids = [211, 218, 39, 194, 189, 39, 40, 172, 39, 191, 40, 2]
    assert tokenizer.decode(decoded_ids) == "ov^YH["
    assert tokenizer.decode(decoded_ids, special_token_policy=SpecialTokenPolicy.KEEP) == "ov[STREAMING_PAD]^Y[STREAMING_PAD][STREAMING_WORD]H[STREAMING_PAD][[STREAMING_WORD]</s>"

@pytest.mark.parametrize(
    ("mode", "expected_array_len"), [(StreamingMode.OFFLINE, 57600), (StreamingMode.ONLINE, 28160)]
)
def test_tokenize_streaming_request(
    tokenizer: InstructTokenizerV7, mode: StreamingMode, expected_array_len: int
) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24_000
    signal_length = int(duration * sampling_rate)

    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
        format="wav",
    )

    streaming_request = TranscriptionRequest(audio=RawAudio(data=audio.to_base64("wav"), format="wav"), streaming=mode, language=None)

    tokenized = tokenizer.encode_transcription(streaming_request)
    assert tokenizer.audio_encoder is not None
    delay_n_tokens = tokenizer.audio_encoder.audio_config.num_delay_tokens

    BOS = tokenizer.tokenizer.get_special_token(SpecialTokens.bos.value)
    STREAMING_PAD = tokenizer.tokenizer.get_special_token(SpecialTokens.streaming_pad.value)

    assert tokenized.tokens == ([BOS] + delay_n_tokens * [STREAMING_PAD]), f"{tokenized.tokens}"
    assert tokenized.text == "<s>" + delay_n_tokens * "[STREAMING_PAD]"

    audio_array = tokenized.audios[0].audio_array
    assert audio_array.shape == (expected_array_len,)

    audio_len_in_tokens = tokenizer.audio_encoder.audio_config.num_audio_tokens(audio_array.shape[0])
    expected_audio_len_in_tokens = int(math.ceil(duration * tokenizer.audio_encoder.audio_config.frame_rate))

    if mode == StreamingMode.OFFLINE:
        expected_audio_len_in_tokens += +len(tokenized.tokens) + OFFLINE_STREAMING_BUFFER_TOKENS

        assert audio_len_in_tokens == expected_audio_len_in_tokens


@pytest.mark.parametrize(
    ("rate", "delay", "num_delay_tokens"),
    [(12.5, 360, -1), (12.5, 560, 7), (12.5, 960, 12), (50, 360, 18), (50, 560, 28), (50, 960, 48), (30, 960, -1)],
)
def test_audio_config_delay(rate: float, delay: int, num_delay_tokens: int) -> None:
    spectogram_config = AudioSpectrogramConfig(**_audio_spectrogram_config)

    def audio_config_fn() -> AudioConfig:
        return AudioConfig(
            sampling_rate=16_000,
            frame_rate=rate,
            encoding_config=spectogram_config,
            transcription_delay_ms=delay,
            transcription_format=TranscriptionFormat.STREAMING,
        )

    if num_delay_tokens == -1:
        with pytest.raises(AssertionError, match="must be a multiple"):
            audio_config_fn()
    else:
        assert audio_config_fn().num_delay_tokens == num_delay_tokens

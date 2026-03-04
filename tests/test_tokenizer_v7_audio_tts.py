import math

import numpy as np
import pytest

from mistral_common.tokens.audio.speech.request import BaseSpeechRequest
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
)
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV7
from mistral_common.tokens.tokenizers.mistral import load_audio_encoder
from mistral_common.tokens.tokenizers.special_tokens import SpecialTokenPolicy, SpecialTokens

from .test_tokenizer_v7_audio import load_audio_v7_tokenizer


@pytest.fixture(scope="session")
def tts_tokenizer() -> InstructTokenizerV7:
    mm_tekkenizer = load_audio_v7_tokenizer()
    audio_encoder = load_audio_encoder(
        AudioConfig(
            sampling_rate=24000,
            frame_rate=12.5,
            audio_spectrogram_config=AudioSpectrogramConfig(
                num_mel_bins=128,
                window_size=400,
                hop_length=160,
            ),
            voice_num_audio_tokens={
                "female": 52,
                "male": 76,
            },
        ),
        mm_tekkenizer,
    )
    assert isinstance(audio_encoder, AudioEncoder), type(audio_encoder)
    return InstructTokenizerV7(tokenizer=mm_tekkenizer, audio_encoder=audio_encoder)


def _make_fake_audio(duration: float, sampling_rate: int = 24000) -> Audio:
    rng = np.random.default_rng(42)
    audio_array = rng.uniform(low=-1, high=1, size=int(duration * sampling_rate))
    return Audio(audio_array=audio_array, sampling_rate=sampling_rate, format="wav")


def test_encode_speech_request_with_ref_audio(tts_tokenizer: InstructTokenizerV7) -> None:
    duration = 1.5
    sampling_rate = 24000
    audio = _make_fake_audio(duration, sampling_rate)
    request = BaseSpeechRequest(input="Hello world", ref_audio=audio.to_base64())
    tokenized = tts_tokenizer.encode_speech_request(request)

    assert isinstance(tts_tokenizer.audio_encoder, AudioEncoder)
    frame_rate = tts_tokenizer.audio_encoder.audio_config.frame_rate
    num_audio_tokens = math.ceil(duration * frame_rate) + 1  # +1 for eoa

    BOS = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.bos.value)
    AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.begin_audio.value)
    NEXT_AUDIO_TEXT = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.text_to_audio.value)
    REPEAT_AUDIO_TEXT = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio_to_text.value)

    text_tokens = tts_tokenizer.tokenizer.encode("Hello world", bos=False, eos=False)

    expected = (
        [BOS]
        + [BEGIN_AUDIO]
        + [AUDIO] * num_audio_tokens
        + [NEXT_AUDIO_TEXT]
        + text_tokens
        + [REPEAT_AUDIO_TEXT]
        + [BEGIN_AUDIO]
    )
    assert tokenized.tokens == expected, f"{tokenized.tokens=} != {expected=}"
    assert len(tokenized.audios) == 1
    assert np.allclose(tokenized.audios[0].audio_array, audio.audio_array, atol=1e-3)


def test_encode_speech_request_with_voice_female(tts_tokenizer: InstructTokenizerV7) -> None:
    request = BaseSpeechRequest(input="Hello world", voice="female")
    tokenized = tts_tokenizer.encode_speech_request(request)

    BOS = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.bos.value)
    AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.begin_audio.value)
    NEXT_AUDIO_TEXT = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.text_to_audio.value)
    REPEAT_AUDIO_TEXT = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio_to_text.value)

    text_tokens = tts_tokenizer.tokenizer.encode("Hello world", bos=False, eos=False)

    expected = (
        [BOS] + [BEGIN_AUDIO] + [AUDIO] * 52 + [NEXT_AUDIO_TEXT] + text_tokens + [REPEAT_AUDIO_TEXT] + [BEGIN_AUDIO]
    )
    assert tokenized.tokens == expected, f"{tokenized.tokens=} != {expected=}"
    assert tokenized.audios == []


def test_encode_speech_request_with_voice_male(tts_tokenizer: InstructTokenizerV7) -> None:
    request = BaseSpeechRequest(input="Hello world", voice="male")
    tokenized = tts_tokenizer.encode_speech_request(request)

    BOS = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.bos.value)
    AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.begin_audio.value)
    NEXT_AUDIO_TEXT = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.text_to_audio.value)
    REPEAT_AUDIO_TEXT = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio_to_text.value)

    text_tokens = tts_tokenizer.tokenizer.encode("Hello world", bos=False, eos=False)

    expected = (
        [BOS] + [BEGIN_AUDIO] + [AUDIO] * 76 + [NEXT_AUDIO_TEXT] + text_tokens + [REPEAT_AUDIO_TEXT] + [BEGIN_AUDIO]
    )
    assert tokenized.tokens == expected, f"{tokenized.tokens=} != {expected=}"
    assert tokenized.audios == []


def test_encode_speech_request_ref_audio_takes_precedence_over_voice(
    tts_tokenizer: InstructTokenizerV7,
) -> None:
    audio = _make_fake_audio(1.0)
    request = BaseSpeechRequest(input="text", ref_audio=audio.to_base64(), voice="female")
    tokenized = tts_tokenizer.encode_speech_request(request)

    # ref_audio takes precedence: audio is not None so the audio path is used
    assert len(tokenized.audios) == 1
    assert np.allclose(tokenized.audios[0].audio_array, audio.audio_array, atol=1e-3)

    # Token count should match ref_audio duration, not the preset 52 for "female"
    assert isinstance(tts_tokenizer.audio_encoder, AudioEncoder)
    frame_rate = tts_tokenizer.audio_encoder.audio_config.frame_rate
    num_audio_tokens = math.ceil(1.0 * frame_rate) + 1
    AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio.value)
    audio_token_count = tokenized.tokens.count(AUDIO)
    assert audio_token_count == num_audio_tokens, f"{audio_token_count=} != {num_audio_tokens=}"


def test_encode_speech_request_neither_audio_nor_voice_fails(
    tts_tokenizer: InstructTokenizerV7,
) -> None:
    request = BaseSpeechRequest(input="Hello world")
    with pytest.raises(AssertionError, match="Either ref_audio or voice must be defined"):
        tts_tokenizer.encode_speech_request(request)


def test_encode_speech_request_text_tokens_correct(tts_tokenizer: InstructTokenizerV7) -> None:
    input_text = "The quick brown fox"
    request = BaseSpeechRequest(input=input_text, voice="female")
    tokenized = tts_tokenizer.encode_speech_request(request)

    NEXT_AUDIO_TEXT = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.text_to_audio.value)
    REPEAT_AUDIO_TEXT = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio_to_text.value)

    # Extract text tokens between [NEXT_AUDIO_TEXT] and [REPEAT_AUDIO_TEXT]
    next_idx = tokenized.tokens.index(NEXT_AUDIO_TEXT)
    repeat_idx = tokenized.tokens.index(REPEAT_AUDIO_TEXT)
    text_tokens = tokenized.tokens[next_idx + 1 : repeat_idx]

    decoded = tts_tokenizer.tokenizer.decode(text_tokens, special_token_policy=SpecialTokenPolicy.IGNORE)
    assert decoded == input_text, f"{decoded=} != {input_text=}"


def test_encode_speech_request_no_audio_encoder_fails() -> None:
    mm_tekkenizer = load_audio_v7_tokenizer()
    tokenizer_no_encoder = InstructTokenizerV7(tokenizer=mm_tekkenizer, audio_encoder=None)

    request = BaseSpeechRequest(input="Hello world", voice="female")
    with pytest.raises(AssertionError, match="Audio encoder must be defined"):
        tokenizer_no_encoder.encode_speech_request(request)


def test_encode_speech_request_audio_resampled(tts_tokenizer: InstructTokenizerV7) -> None:
    # Create audio at 16000 Hz — different from the encoder's 24000 Hz
    duration = 2.0
    source_sr = 16000
    audio = _make_fake_audio(duration, source_sr)
    request = BaseSpeechRequest(input="Resample test", ref_audio=audio.to_base64())
    tokenized = tts_tokenizer.encode_speech_request(request)

    assert isinstance(tts_tokenizer.audio_encoder, AudioEncoder)
    target_sr = tts_tokenizer.audio_encoder.audio_config.sampling_rate
    frame_rate = tts_tokenizer.audio_encoder.audio_config.frame_rate

    # After resampling the duration stays the same, but length changes to target_sr * duration
    resampled_length = int(duration * target_sr)
    num_audio_tokens = math.ceil((resampled_length / target_sr) * frame_rate) + 1

    AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio.value)
    audio_token_count = tokenized.tokens.count(AUDIO)
    assert audio_token_count == num_audio_tokens, f"{audio_token_count=} != {num_audio_tokens=}"
    assert len(tokenized.audios) == 1

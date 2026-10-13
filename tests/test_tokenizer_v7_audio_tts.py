import math
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from mistral_common.exceptions import (
    InvalidRequestException,
    TokenizerException,
    UnsupportedTokenizerFeatureException,
)
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
)
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV7
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer, load_audio_encoder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer

from .test_tokenizer_v7_audio import (
    SYNTHETIC_V7_SPEECH,
    SYNTHETIC_V7_SPEECH_NO_AUDIO,
    SYNTHETIC_V7_SPEECH_NO_AUDIO_TO_TEXT,
    SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO,
    SYNTHETIC_V7_SPEECH_NO_MARKER,
    SYNTHETIC_V7_SPEECH_NO_VOICE_MAP,
    SyntheticV7AudioProfile,
    build_synthetic_v7_audio_tokenizer,
    get_tekkenizer_with_audio,
    load_v7_audio_error_tokenizer,
    preset_speech_request,
    reference_speech_request,
    unknown_voice_speech_request,
    valid_reference_audio,
)


def test_audio_encoder_speech_request_requires_source() -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_SPEECH)
    assert isinstance(tokenizer.instruct_tokenizer.audio_encoder, AudioEncoder)

    with pytest.raises(
        InvalidRequestException,
        match=r"^Either audio or voice must be defined to encode speech\.$",
    ):
        tokenizer.instruct_tokenizer.audio_encoder.encode_audio_for_speech_request(audio=None, voice=None)


@pytest.fixture(scope="session")
def tts_tokenizer() -> InstructTokenizerV7:
    mm_tekkenizer = get_tekkenizer_with_audio().tokenizer
    assert isinstance(mm_tekkenizer, Tekkenizer)
    audio_encoder = load_audio_encoder(
        AudioConfig(
            sampling_rate=24000,
            frame_rate=12.5,
            encoding_config=AudioSpectrogramConfig(
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
    request = SpeechRequest(input="Hello world", ref_audio=audio.to_base64("wav"))
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
    request = SpeechRequest(input="Hello world", voice="female")
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
    request = SpeechRequest(input="Hello world", voice="male")
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
    request = SpeechRequest(input="text", ref_audio=audio.to_base64("wav"), voice="female")
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


@pytest.mark.parametrize(
    "speech_input",
    [pytest.param("Hello world", id="audio-v7-speech-source-required-test")],
)
def test_encode_speech_request_neither_audio_nor_voice_fails(speech_input: str) -> None:
    with pytest.raises(
        ValidationError,
        match=r"Value error, Either ref_audio or voice must be defined\. \[type=value_error",
    ):
        SpeechRequest(input=speech_input)


def test_encode_speech_request_text_tokens_correct(tts_tokenizer: InstructTokenizerV7) -> None:
    input_text = "The quick brown fox"
    request = SpeechRequest(input=input_text, voice="female")
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
    mm_tekkenizer = get_tekkenizer_with_audio().tokenizer
    assert isinstance(mm_tekkenizer, Tekkenizer)
    tokenizer_no_encoder = InstructTokenizerV7(tokenizer=mm_tekkenizer, audio_encoder=None)

    request = SpeechRequest(input="Hello world", voice="female")
    with pytest.raises(
        UnsupportedTokenizerFeatureException,
        match=r"Tokenizer v7 does not provide an audio encoder for speech",
    ):
        tokenizer_no_encoder.encode_speech_request(request)


def test_encode_speech_request_audio_resampled(tts_tokenizer: InstructTokenizerV7) -> None:
    # Create audio at 16000 Hz — different from the encoder's 24000 Hz
    duration = 2.0
    source_sr = 16000
    audio = _make_fake_audio(duration, source_sr)
    request = SpeechRequest(input="Resample test", ref_audio=audio.to_base64("wav"))
    tokenized = tts_tokenizer.encode_speech_request(request)

    assert isinstance(tts_tokenizer.audio_encoder, AudioEncoder)
    target_sr = tts_tokenizer.audio_encoder.audio_config.sampling_rate
    frame_rate = tts_tokenizer.audio_encoder.audio_config.frame_rate

    assert tokenized.audios[0].sampling_rate == target_sr, f"{frame_rate=}, {tokenized.audios[0].sampling_rate=}"

    # After resampling the duration stays the same, but length changes to target_sr * duration
    num_audio_tokens = math.ceil(duration * frame_rate) + 1

    AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio.value)
    audio_token_count = tokenized.tokens.count(AUDIO)
    assert audio_token_count == num_audio_tokens, f"{audio_token_count=} != {num_audio_tokens=}"
    assert len(tokenized.audios) == 1


PUBLIC_SPEECH_ERROR_CASES = (
    pytest.param(
        None,
        preset_speech_request,
        UnsupportedTokenizerFeatureException,
        r"Tokenizer v7 does not provide an audio encoder for speech",
        id="audio-v7-speech-no-encoder-test",
    ),
    pytest.param(
        SYNTHETIC_V7_SPEECH_NO_MARKER,
        reference_speech_request,
        UnsupportedTokenizerFeatureException,
        r"the text_to_audio marker required for speech",
        id="audio-v7-speech-no-marker-test",
    ),
    pytest.param(
        SYNTHETIC_V7_SPEECH_NO_AUDIO_TO_TEXT,
        reference_speech_request,
        UnsupportedTokenizerFeatureException,
        r"the audio_to_text marker required for speech",
        id="audio-v7-speech-no-audio-to-text-test",
    ),
    pytest.param(
        SYNTHETIC_V7_SPEECH_NO_VOICE_MAP,
        preset_speech_request,
        UnsupportedTokenizerFeatureException,
        r"(?i)preset voices.*not configured",
        id="audio-v7-speech-no-voice-map-test",
    ),
    pytest.param(
        SYNTHETIC_V7_SPEECH_NO_AUDIO,
        reference_speech_request,
        UnsupportedTokenizerFeatureException,
        r"the audio marker required for speech",
        id="audio-v7-speech-no-audio-token-test",
    ),
    pytest.param(
        SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO,
        reference_speech_request,
        UnsupportedTokenizerFeatureException,
        r"the begin_audio marker required for speech",
        id="audio-v7-speech-no-begin-audio-test",
    ),
    pytest.param(
        SYNTHETIC_V7_SPEECH,
        unknown_voice_speech_request,
        InvalidRequestException,
        r"Unknown voice 'not-configured', expected one of \['preset'\]",
        id="audio-v7-speech-unknown-voice-test",
    ),
)


@pytest.mark.parametrize(("profile", "request_factory", "exception_type", "message"), PUBLIC_SPEECH_ERROR_CASES)
def test_public_speech_error(
    profile: SyntheticV7AudioProfile | None,
    request_factory: Callable[[], SpeechRequest],
    exception_type: type[Exception],
    message: str,
) -> None:
    tokenizer = load_v7_audio_error_tokenizer(profile=profile)
    assert tokenizer.mode == ValidationMode.test

    request = request_factory()
    with pytest.raises(exception_type, match=message):
        tokenizer.encode_speech_request(request)


@pytest.mark.parametrize(
    ("profile", "request_factory", "message"),
    [
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_MARKER,
            reference_speech_request,
            r"the text_to_audio marker required for speech",
            id="v7-speech-profile-without-required-marker",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_AUDIO_TO_TEXT,
            reference_speech_request,
            r"the audio_to_text marker required for speech",
            id="v7-speech-profile-without-audio-to-text-marker",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_AUDIO,
            reference_speech_request,
            r"the audio marker required for speech",
            id="v7-speech-profile-without-audio-marker",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO,
            reference_speech_request,
            r"the begin_audio marker required for speech",
            id="v7-speech-profile-without-begin-audio-marker",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_VOICE_MAP,
            preset_speech_request,
            r"(?i)preset voices.*not configured",
            id="v7-speech-profile-without-voice-map",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_VOICE_MAP,
            unknown_voice_speech_request,
            r"(?i)preset voices.*not configured",
            id="v7-speech-absent-voice-map-precedes-unknown-voice",
        ),
    ],
)
def test_direct_unsupported_speech_capability(
    profile: SyntheticV7AudioProfile | None,
    request_factory: Callable[[], SpeechRequest],
    message: str,
) -> None:
    tokenizer = load_v7_audio_error_tokenizer(profile=profile)

    request = request_factory()
    with pytest.raises(UnsupportedTokenizerFeatureException, match=message):
        tokenizer.instruct_tokenizer.encode_speech_request(request)


@pytest.mark.parametrize(
    ("request_factory", "message"),
    [
        pytest.param(
            unknown_voice_speech_request,
            r"Unknown voice 'not-configured', expected one of \['preset'\]",
            id="speech-unknown-configured-voice",
        ),
    ],
)
def test_direct_invalid_speech_request(request_factory: Callable[[], SpeechRequest], message: str) -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_SPEECH)
    request = request_factory()

    with pytest.raises(InvalidRequestException, match=message):
        tokenizer.instruct_tokenizer.encode_speech_request(request)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        pytest.param(MistralTokenizer.v1, r"Speech request not available for tokenizer v1", id="v1-speech"),
        pytest.param(MistralTokenizer.v2, r"Speech request not available for tokenizer v2", id="v2-speech"),
        pytest.param(MistralTokenizer.v3, r"Speech request not available for tokenizer v3", id="v3-speech"),
    ],
)
def test_pre_v7_speech_remains_tokenizer_error(factory: Callable[[], MistralTokenizer], message: str) -> None:
    tokenizer = factory()
    request = SpeechRequest(input="hello", voice="preset")
    with pytest.raises(TokenizerException, match=message):
        tokenizer.instruct_tokenizer.encode_speech_request(request)


def test_reference_audio_ignores_unknown_voice_with_direct_encoder() -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_SPEECH)
    encoded_audio = valid_reference_audio()
    request = SpeechRequest(input="hello", ref_audio=encoded_audio, voice="not-configured")

    tokenized = tokenizer.instruct_tokenizer.encode_speech_request(request)

    expected_audio = Audio.from_base64(encoded_audio)
    assert len(tokenized.audios) == 1
    assert np.allclose(tokenized.audios[0].audio_array, expected_audio.audio_array, atol=1e-3)

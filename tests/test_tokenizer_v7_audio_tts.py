import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest

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
    BUNDLED_V7_NO_AUDIO_CONFIGURATION_ID,
    SYNTHETIC_V7_SPEECH,
    SYNTHETIC_V7_SPEECH_NO_AUDIO,
    SYNTHETIC_V7_SPEECH_NO_AUDIO_TO_TEXT,
    SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO,
    SYNTHETIC_V7_SPEECH_NO_MARKER,
    SYNTHETIC_V7_SPEECH_NO_VOICE_MAP,
    SyntheticV7AudioProfile,
    build_synthetic_v7_audio_tokenizer,
    get_tekkenizer_with_audio,
    load_bundled_v7_no_audio_tokenizer,
    valid_reference_audio,
)


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


def test_encode_speech_request_neither_audio_nor_voice_fails(
    tts_tokenizer: InstructTokenizerV7,
) -> None:
    request = SpeechRequest(input="Hello world")
    with pytest.raises(InvalidRequestException, match="Either ref_audio or voice must be defined"):
        tts_tokenizer.encode_speech_request(request)


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
    with pytest.raises(UnsupportedTokenizerFeatureException, match="audio encoder.*speech"):
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


@dataclass(frozen=True)
class PublicSpeechErrorCase:
    case_id: str
    configuration_id: str
    mode: ValidationMode
    profile: SyntheticV7AudioProfile | None
    request_recipe: Literal["speech-no-source", "speech-reference", "speech-preset", "speech-unknown-voice"]
    exception_type: type[UnsupportedTokenizerFeatureException] | type[InvalidRequestException]
    message: str


PUBLIC_SPEECH_ERROR_CASES = (
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-no-encoder-test",
        configuration_id=BUNDLED_V7_NO_AUDIO_CONFIGURATION_ID,
        mode=ValidationMode.test,
        profile=None,
        request_recipe="speech-no-source",
        exception_type=UnsupportedTokenizerFeatureException,
        message=r"audio encoder.*speech",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-no-marker-test",
        configuration_id=SYNTHETIC_V7_SPEECH_NO_MARKER.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_SPEECH_NO_MARKER,
        request_recipe="speech-reference",
        exception_type=UnsupportedTokenizerFeatureException,
        message=r"text_to_audio marker.*speech",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-no-voice-map-test",
        configuration_id=SYNTHETIC_V7_SPEECH_NO_VOICE_MAP.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_SPEECH_NO_VOICE_MAP,
        request_recipe="speech-preset",
        exception_type=UnsupportedTokenizerFeatureException,
        message=r"(?i)preset voices.*not configured",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-no-audio-token-test",
        configuration_id=SYNTHETIC_V7_SPEECH_NO_AUDIO.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_SPEECH_NO_AUDIO,
        request_recipe="speech-reference",
        exception_type=UnsupportedTokenizerFeatureException,
        message=r"audio marker.*speech",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-no-begin-audio-test",
        configuration_id=SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO,
        request_recipe="speech-reference",
        exception_type=UnsupportedTokenizerFeatureException,
        message=r"begin_audio marker.*speech",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-source-required-test",
        configuration_id=SYNTHETIC_V7_SPEECH.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_SPEECH,
        request_recipe="speech-no-source",
        exception_type=InvalidRequestException,
        message=r"Either ref_audio or voice must be defined",
    ),
    PublicSpeechErrorCase(
        case_id="audio-v7-speech-unknown-voice-test",
        configuration_id=SYNTHETIC_V7_SPEECH.configuration_id,
        mode=ValidationMode.test,
        profile=SYNTHETIC_V7_SPEECH,
        request_recipe="speech-unknown-voice",
        exception_type=InvalidRequestException,
        message=r"Unknown voice.*expected one of \['preset'\]",
    ),
)


def _build_speech_error_request(
    recipe: Literal["speech-no-source", "speech-reference", "speech-preset", "speech-unknown-voice"],
) -> SpeechRequest:
    if recipe == "speech-no-source":
        return SpeechRequest(input="hello")
    if recipe == "speech-reference":
        return SpeechRequest(input="hello", ref_audio=valid_reference_audio())
    if recipe == "speech-preset":
        return SpeechRequest(input="hello", voice="preset")
    assert recipe == "speech-unknown-voice"
    return SpeechRequest(input="hello", voice="not-configured")


@pytest.mark.parametrize("case", PUBLIC_SPEECH_ERROR_CASES, ids=lambda case: case.case_id)
def test_public_speech_error(case: PublicSpeechErrorCase) -> None:
    if case.profile is None:
        assert case.configuration_id == BUNDLED_V7_NO_AUDIO_CONFIGURATION_ID
        tokenizer = load_bundled_v7_no_audio_tokenizer(mode=case.mode)
    else:
        assert case.configuration_id == case.profile.configuration_id
        tokenizer = build_synthetic_v7_audio_tokenizer(profile=case.profile, mode=case.mode)
    assert tokenizer.mode == case.mode

    request = _build_speech_error_request(case.request_recipe)
    with pytest.raises(case.exception_type, match=case.message):
        tokenizer.encode_speech_request(request)


@pytest.mark.parametrize(
    ("profile", "request_recipe", "message"),
    [
        pytest.param(
            None,
            "speech-no-source",
            r"audio encoder.*speech",
            id="v7-speech-without-encoder-takes-priority-over-missing-source",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_MARKER,
            "speech-reference",
            r"text_to_audio marker.*speech",
            id="v7-speech-profile-without-required-marker",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_AUDIO_TO_TEXT,
            "speech-reference",
            r"audio_to_text marker.*speech",
            id="v7-speech-profile-without-audio-to-text-marker",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_AUDIO,
            "speech-reference",
            r"audio marker.*speech",
            id="v7-speech-profile-without-audio-marker",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO,
            "speech-reference",
            r"begin_audio marker.*speech",
            id="v7-speech-profile-without-begin-audio-marker",
        ),
        pytest.param(
            SYNTHETIC_V7_SPEECH_NO_VOICE_MAP,
            "speech-preset",
            r"(?i)preset voices.*not configured",
            id="v7-speech-profile-without-voice-map",
        ),
    ],
)
def test_direct_unsupported_speech_capability(
    profile: SyntheticV7AudioProfile | None,
    request_recipe: Literal["speech-no-source", "speech-reference", "speech-preset", "speech-unknown-voice"],
    message: str,
) -> None:
    if profile is None:
        tokenizer = load_bundled_v7_no_audio_tokenizer(mode=ValidationMode.test)
    else:
        tokenizer = build_synthetic_v7_audio_tokenizer(profile=profile, mode=ValidationMode.test)

    request = _build_speech_error_request(request_recipe)
    with pytest.raises(UnsupportedTokenizerFeatureException, match=message):
        tokenizer.instruct_tokenizer.encode_speech_request(request)


@pytest.mark.parametrize(
    ("request_recipe", "message"),
    [
        pytest.param(
            "speech-no-source",
            r"Either ref_audio or voice must be defined",
            id="speech-source-required",
        ),
        pytest.param(
            "speech-unknown-voice",
            r"Unknown voice.*expected one of \['preset'\]",
            id="speech-unknown-configured-voice",
        ),
    ],
)
def test_direct_invalid_speech_request(
    request_recipe: Literal["speech-no-source", "speech-unknown-voice"], message: str
) -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_SPEECH, mode=ValidationMode.test)
    request = _build_speech_error_request(request_recipe)

    with pytest.raises(InvalidRequestException, match=message):
        tokenizer.instruct_tokenizer.encode_speech_request(request)


@pytest.mark.parametrize(
    ("version", "message"),
    [
        pytest.param("v1", r"Speech request not available for tokenizer v1", id="v1-speech"),
        pytest.param("v2", r"Speech request not available for tokenizer v2", id="v2-speech"),
        pytest.param("v3", r"Speech request not available for tokenizer v3", id="v3-speech"),
    ],
)
def test_pre_v7_speech_remains_tokenizer_error(version: str, message: str) -> None:
    if version == "v1":
        tokenizer = MistralTokenizer.v1()
    elif version == "v2":
        tokenizer = MistralTokenizer.v2()
    else:
        assert version == "v3"
        tokenizer = MistralTokenizer.v3()

    request = SpeechRequest(input="hello")
    with pytest.raises(TokenizerException, match=message):
        tokenizer.instruct_tokenizer.encode_speech_request(request)


def test_reference_audio_ignores_unknown_voice_with_direct_encoder() -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_SPEECH, mode=ValidationMode.test)
    encoded_audio = valid_reference_audio()
    request = SpeechRequest(input="hello", ref_audio=encoded_audio, voice="not-configured")

    tokenized = tokenizer.instruct_tokenizer.encode_speech_request(request)

    expected_audio = Audio.from_base64(encoded_audio)
    assert len(tokenized.audios) == 1
    assert np.allclose(tokenized.audios[0].audio_array, expected_audio.audio_array, atol=1e-3)

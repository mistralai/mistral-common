r"""Direct synthetic v7 audio encoding: chat audio, transcription, streaming and speech.

The synthetic Tekken vocabularies used here establish formatting behavior only;
their token IDs are not released-vocabulary evidence.
"""

import base64
import hashlib
import io
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import requests
from pydantic import ValidationError
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.exceptions import (
    AudioDecodeException,
    InvalidRequestException,
    TokenizerException,
    UnsupportedTokenizerFeatureException,
)
from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURL, AudioURLChunk, ContentChunk, TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import (
    OFFLINE_STREAMING_BUFFER_TOKENS,
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
    TranscriptionFormat,
)
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV7
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer, load_audio_encoder
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.utils import decode_keep

BUNDLED_V7_NO_AUDIO_SHA256 = "1b968b8dc352f42192367337c78ccc61e1eaddc6d641a579372d4f20694beb7a"


@dataclass(frozen=True)
class SyntheticV7AudioProfile:
    configuration_id: str
    omitted_markers: tuple[str, ...]
    streaming: bool
    voices: tuple[tuple[str, int], ...] | None


SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-instruct-no-transcribe-test",
    omitted_markers=(SpecialTokens.transcribe.value,),
    streaming=False,
    voices=(("preset", 5),),
)
SYNTHETIC_V7_SPEECH_NO_MARKER = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-speech-no-marker-test",
    omitted_markers=(SpecialTokens.text_to_audio.value,),
    streaming=False,
    voices=(("preset", 5),),
)
SYNTHETIC_V7_SPEECH_NO_AUDIO_TO_TEXT = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-speech-no-audio-to-text-test",
    omitted_markers=(SpecialTokens.audio_to_text.value,),
    streaming=False,
    voices=(("preset", 5),),
)
SYNTHETIC_V7_INSTRUCT_NO_AUDIO = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-instruct-no-audio-test",
    omitted_markers=(SpecialTokens.audio.value,),
    streaming=False,
    voices=(("preset", 5),),
)
SYNTHETIC_V7_INSTRUCT_NO_BEGIN_AUDIO = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-instruct-no-begin-audio-test",
    omitted_markers=(SpecialTokens.begin_audio.value,),
    streaming=False,
    voices=(("preset", 5),),
)
SYNTHETIC_V7_STREAMING_NO_PAD = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-streaming-no-streaming-pad-test",
    omitted_markers=(SpecialTokens.streaming_pad.value,),
    streaming=True,
    voices=(("preset", 5),),
)
SYNTHETIC_V7_SPEECH_NO_AUDIO = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-speech-no-audio-test",
    omitted_markers=(SpecialTokens.audio.value,),
    streaming=False,
    voices=(("preset", 5),),
)
SYNTHETIC_V7_SPEECH_NO_BEGIN_AUDIO = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-speech-no-begin-audio-test",
    omitted_markers=(SpecialTokens.begin_audio.value,),
    streaming=False,
    voices=(("preset", 5),),
)
SYNTHETIC_V7_SPEECH_NO_VOICE_MAP = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-speech-no-map-test",
    omitted_markers=(),
    streaming=False,
    voices=None,
)
SYNTHETIC_V7_INSTRUCT = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-instruct-test",
    omitted_markers=(),
    streaming=False,
    voices=(("preset", 5),),
)
SYNTHETIC_V7_STREAMING = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-streaming-test",
    omitted_markers=(),
    streaming=True,
    voices=(("preset", 5),),
)
SYNTHETIC_V7_SPEECH = SyntheticV7AudioProfile(
    configuration_id="synthetic-v7-speech-test",
    omitted_markers=(),
    streaming=False,
    voices=(("preset", 5),),
)


def load_bundled_v7_no_audio_tokenizer() -> MistralTokenizer:
    path = MistralTokenizer._data_path() / "mistral_instruct_tokenizer_241114.model.v7"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == BUNDLED_V7_NO_AUDIO_SHA256

    tokenizer = MistralTokenizer.from_file(tokenizer_filename=path, mode=ValidationMode.test)
    assert tokenizer.version == TokenizerVersion.v7
    assert tokenizer.mode == ValidationMode.test
    assert tokenizer.instruct_tokenizer.audio_encoder is None
    return tokenizer


def build_synthetic_v7_audio_tokenizer(profile: SyntheticV7AudioProfile) -> MistralTokenizer:
    special_tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v7, add_audio=True)
    if profile.streaming:
        special_tokens.extend(
            [
                SpecialTokenInfo(rank=37, token_str="<SPCECIAL_37>", is_control=True),
                SpecialTokenInfo(rank=38, token_str="<SPCECIAL_38>", is_control=True),
                SpecialTokenInfo(rank=39, token_str=SpecialTokens.streaming_pad.value, is_control=True),
                SpecialTokenInfo(rank=40, token_str=SpecialTokens.streaming_word.value, is_control=True),
            ]
        )
    special_tokens = [
        SpecialTokenInfo(
            rank=token["rank"],
            token_str=f"<OMITTED_{token['rank']}>",
            is_control=True,
        )
        if token["token_str"] in profile.omitted_markers
        else token
        for token in special_tokens
    ]
    audio_config = AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(num_mel_bins=128, window_size=400, hop_length=160),
        transcription_format=(TranscriptionFormat.STREAMING if profile.streaming else TranscriptionFormat.INSTRUCT),
        transcription_delay_ms=480.0 if profile.streaming else None,
        streaming_look_ahead_ms=2.5 if profile.streaming else None,
        streaming_look_back_ms=52.5 if profile.streaming else None,
        streaming_n_left_pad_tokens=16 if profile.streaming else None,
        voice_num_audio_tokens=(dict(profile.voices) if profile.voices is not None else None),
    )
    tekkenizer = Tekkenizer(
        vocab=quick_vocab([b"synthetic"]),
        special_tokens=special_tokens,
        pattern=r".+",
        vocab_size=356,
        num_special_tokens=100,
        version=TokenizerVersion.v7,
        audio_config=audio_config,
    )
    audio_encoder = load_audio_encoder(audio_config=audio_config, tokenizer=tekkenizer)
    if SpecialTokens.transcribe.value in profile.omitted_markers:
        # The V7 constructor only binds TRANSCRIBE when an audio encoder is passed.
        instruct_tokenizer = InstructTokenizerV7(tokenizer=tekkenizer, audio_encoder=None)
        instruct_tokenizer.audio_encoder = audio_encoder
    else:
        instruct_tokenizer = InstructTokenizerV7(tokenizer=tekkenizer, audio_encoder=audio_encoder)
    return MistralTokenizer(
        instruct_tokenizer=instruct_tokenizer,
        validator=get_validator(version=TokenizerVersion.v7, mode=ValidationMode.test),
        request_normalizer=get_normalizer(version=TokenizerVersion.v7),
    )


def load_v7_audio_error_tokenizer(profile: SyntheticV7AudioProfile | None) -> MistralTokenizer:
    if profile is None:
        return load_bundled_v7_no_audio_tokenizer()
    return build_synthetic_v7_audio_tokenizer(profile=profile)


def valid_reference_audio() -> str:
    audio = Audio(
        audio_array=np.zeros(shape=(2_400,), dtype=np.float32),
        sampling_rate=24_000,
        format="wav",
    )
    return audio.to_base64("wav")


def valid_reference_audio_bytes() -> bytes:
    return base64.b64decode(valid_reference_audio(), validate=True)


def reference_speech_request() -> SpeechRequest:
    return SpeechRequest(input="hello", ref_audio=valid_reference_audio())


def preset_speech_request() -> SpeechRequest:
    return SpeechRequest(input="hello", voice="preset")


def unknown_voice_speech_request() -> SpeechRequest:
    return SpeechRequest(input="hello", voice="not-configured")


def undecodable_transcription_request() -> TranscriptionRequest:
    return TranscriptionRequest(
        audio=b"not decoded before capability check", language=None, target_streaming_delay_ms=None
    )


def valid_transcription_request() -> TranscriptionRequest:
    return TranscriptionRequest(audio=valid_reference_audio(), language=None, target_streaming_delay_ms=None)


def get_tekkenizer_with_audio() -> InstructTokenizerV7:
    special_tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v7, add_audio=True)
    tokenizer = Tekkenizer(
        quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v7,
    )

    audio_config = AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(
            num_mel_bins=128,
            window_size=400,
            hop_length=160,
        ),
    )
    special_audio_ids = SpecialAudioIDs(
        audio=tokenizer.get_special_token(SpecialTokens.audio.value),
        begin_audio=tokenizer.get_special_token(SpecialTokens.begin_audio.value),
        streaming_pad=None,
        text_to_audio=None,
        audio_to_text=None,
    )
    audio_encoder = AudioEncoder(audio_config, special_audio_ids)

    return InstructTokenizerV7(tokenizer, audio_encoder=audio_encoder)


@pytest.fixture(scope="module")
def tekkenizer() -> InstructTokenizerV7:
    return get_tekkenizer_with_audio()


def _get_audio_chunk(duration: float) -> AudioChunk:
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)
    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
        format="wav",
    )
    return AudioChunk(input_audio=audio.to_base64("wav"))


def _get_specials(tekkenizer: InstructTokenizerV7) -> tuple[int, ...]:
    BOS = tekkenizer.tokenizer.get_special_token(SpecialTokens.bos.value)
    EOS = tekkenizer.tokenizer.get_special_token(SpecialTokens.eos.value)
    BEGIN_INST = tekkenizer.tokenizer.get_special_token(SpecialTokens.begin_inst.value)
    END_INST = tekkenizer.tokenizer.get_special_token(SpecialTokens.end_inst.value)
    AUDIO = tekkenizer.tokenizer.get_special_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = tekkenizer.tokenizer.get_special_token(SpecialTokens.begin_audio.value)
    TRANSCRIBE = tekkenizer.tokenizer.get_special_token(SpecialTokens.transcribe.value)
    return BOS, EOS, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, TRANSCRIBE


DUMMY_AUDIO = _get_audio_chunk(1.7)


def test_tokenize_user_assistant_message(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration)

    num_expected_frames = int(np.ceil(duration * frame_rate))
    tokenized = tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(
                    content=[
                        TextChunk(
                            text="a",
                        ),
                        audio_chunk,
                    ]
                ),
                AssistantMessage(content="c b d"),
            ],
        )
    )

    BOS, EOS, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, _ = _get_specials(tekkenizer)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        197,  # a
        *audio_toks,
        END_INST,
        199,  # "c"
        132,  # " "
        198,  # "b"
        132,  # " "
        200,  # "d"
        EOS,
    ]
    text = decode_keep(tekkenizer, tokenized)
    assert text == ("<s>[INST]a[BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST]c b d</s>")
    assert len(tokenized.audios) == 1
    base64_str = audio_chunk.input_audio
    assert isinstance(base64_str, str)
    audio_array = Audio.from_base64(base64_str).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


@pytest.mark.parametrize("audio_first", [True, False], ids=["audio-first", "text-first"])
def test_tokenize_user_message(tekkenizer: InstructTokenizerV7, audio_first: bool) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration)
    text_chunk = TextChunk(text="a")

    num_expected_frames = int(np.ceil(duration * frame_rate))
    chunks: list[ContentChunk] = [audio_chunk, text_chunk] if audio_first else [text_chunk, audio_chunk]

    tokenized = tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content=chunks),
            ],
        )
    )

    BOS, EOS, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, _ = _get_specials(tekkenizer)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    if audio_first:
        assert tokenized.tokens == [
            BOS,
            BEGIN_INST,
            *audio_toks,
            197,  # "a"
            END_INST,
        ]
        text = decode_keep(tekkenizer, tokenized)
        assert text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "a[/INST]")
    else:
        assert tokenized.tokens == [
            BOS,
            BEGIN_INST,
            197,  # "a"
            *audio_toks,
            END_INST,
        ]
        text = decode_keep(tekkenizer, tokenized)
        assert text == ("<s>[INST]a[BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST]")
    assert len(tokenized.audios) == 1
    base64_str = audio_chunk.input_audio
    assert isinstance(base64_str, str)
    audio_array = Audio.from_base64(base64_str).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


def test_tokenize_multi_turn(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration)
    text_chunk = TextChunk(text="a")

    num_expected_frames = int(np.ceil(duration * frame_rate))
    chunks: list[ContentChunk] = [audio_chunk, text_chunk]

    tokenized = tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content=chunks),
                AssistantMessage(content="c b"),
                UserMessage(content=[audio_chunk]),
                AssistantMessage(content="a f"),
                UserMessage(content=[text_chunk]),
                AssistantMessage(content="g"),
                UserMessage(content=chunks),
            ],
        )
    )

    BOS, EOS, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, _ = _get_specials(tekkenizer)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        *audio_toks,
        197,  # "a"
        END_INST,
        199,  # "c"
        132,  # " "
        198,  # "b"
        EOS,
        BEGIN_INST,
        *audio_toks,
        END_INST,
        197,  # "a"
        132,  # " "
        202,  # "f"
        EOS,
        BEGIN_INST,
        197,  # "a"
        END_INST,
        203,  # "g"
        EOS,
        BEGIN_INST,
        *audio_toks,
        197,  # "a"
        END_INST,
    ]
    assert len(tokenized.audios) == 3
    # each returned waveform matches the independently decoded source signal, in order
    assert isinstance(audio_chunk.input_audio, str)
    source_array = Audio.from_base64(audio_chunk.input_audio).audio_array
    for returned in tokenized.audios:
        assert np.allclose(returned.audio_array, source_array, atol=1e-3)


def test_tokenize_audio_raise(tekkenizer: InstructTokenizerV7) -> None:
    # the request constructs fine; the encoder rejects system+audio at encode time
    request: InstructRequest = InstructRequest(
        messages=[
            SystemMessage(content="a b c d"),
            UserMessage(content=[TextChunk(text="a"), DUMMY_AUDIO]),
        ],
    )
    with pytest.raises(ValueError, match="System messages are not yet allowed when audio is present"):
        tekkenizer.encode_instruct(request)


@pytest.mark.parametrize(
    "url_kind",
    ["path", "path_file", "http", "https", "base64", "base64_prefix"],
)
def test_tokenize_audio_url_chunk(tekkenizer: InstructTokenizerV7, tmp_path: Path, url_kind: str) -> None:
    sampling_rate = 44100
    original_array = np.sin(np.ones([sampling_rate]))
    sample_audio = Audio(audio_array=original_array, sampling_rate=sampling_rate, format="wav")
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(_wav_bytes(sample_audio))

    url_by_kind = {
        "path": str(wav_path),
        "path_file": f"file://{wav_path}",
        "http": "http://example.com/audio.wav",
        "https": "https://example.com/audio.wav",
        "base64": sample_audio.to_base64("wav"),
        "base64_prefix": f"data:audio/wav;base64,{sample_audio.to_base64('wav')}",
    }
    audio_url_chunk = AudioURLChunk(audio_url=AudioURL(url=url_by_kind[url_kind]))

    with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get") as mock_get:
        mock_get.return_value.content = wav_path.read_bytes()
        mock_get.return_value.status_code = 200

        tokenized = tekkenizer.encode_instruct(
            InstructRequest(
                messages=[
                    UserMessage(content=[audio_url_chunk]),
                    AssistantMessage(content="c b d"),
                ],
            )
        )

    BOS, EOS, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, _ = _get_specials(tekkenizer)

    num_expected_frames = int(np.ceil(1 * 12.5))

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    assert len(tokenized.audios) == 1
    assert isinstance(tokenized.audios[0], Audio)
    assert tokenized.audios[0].sampling_rate == 24000
    assert tokenized.audios[0].format == "wav"
    assert tokenized.audios[0].audio_array.shape == (24000,)
    # the returned waveform matches the resampled source, not just its shape
    expected = Audio(audio_array=original_array, sampling_rate=sampling_rate, format="wav")
    expected.resample(24_000)
    assert np.allclose(tokenized.audios[0].audio_array, expected.audio_array, atol=1e-3)
    text = decode_keep(tekkenizer, tokenized)
    assert text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * (len(audio_toks) - 1) + "[/INST]c b d</s>")
    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        *audio_toks,
        END_INST,
        199,  # "c"
        132,  # " "
        198,  # "b"
        132,  # " "
        200,  # "d"
        EOS,
    ]


def _wav_bytes(audio: Audio) -> bytes:
    import io

    import soundfile as sf

    with io.BytesIO() as buf:
        sf.write(buf, audio.audio_array, audio.sampling_rate, format="WAV")
        return buf.getvalue()


def test_encode_invalid_audio_url_chunk(tekkenizer: InstructTokenizerV7) -> None:
    assert tekkenizer.audio_encoder is not None
    # Mock an invalid URL
    with patch("mistral_common.tokens.tokenizers.audio._requests_lib.get") as mock_get:
        mock_get.side_effect = requests.RequestException("connection failed")
        with pytest.raises(AudioDecodeException, match=r"^Could not download the audio from URL"):
            tekkenizer.audio_encoder(AudioURLChunk(audio_url="https://example.com/invalid_audio.wav"))

    # Test with an invalid base64 string
    with pytest.raises(AudioDecodeException, match=r"^Could not decode the audio"):
        tekkenizer.audio_encoder(AudioURLChunk(audio_url="data:audio/wav;base64,invalid_base64_string"))

    # Test with an invalide file path
    with pytest.raises(FileNotFoundError, match=r"file='invalid_file_path.wav' does not exist"):
        tekkenizer.audio_encoder(AudioURLChunk(audio_url="file://invalid_file_path.wav"))


# --- transcription ---


def get_transcription_request(duration: float, language: LanguageAlpha2 | None = None) -> TranscriptionRequest:
    audio_chunk = _get_audio_chunk(duration)

    return TranscriptionRequest(
        model="dummy", audio=audio_chunk.input_audio, language=language, target_streaming_delay_ms=None
    )


def test_tokenize_transcribe(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    num_expected_frames = int(np.ceil(duration * frame_rate))

    request = get_transcription_request(duration)

    tokenized = tekkenizer.encode_transcription(request)
    BOS, _, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, TRANSCRIBE = _get_specials(tekkenizer)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        *audio_toks,
        END_INST,
        TRANSCRIBE,
    ]
    text = decode_keep(tekkenizer, tokenized)
    assert text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST][TRANSCRIBE]")
    assert len(tokenized.audios) == 1
    base64_audio = request.audio
    assert isinstance(base64_audio, str)
    audio_array = Audio.from_base64(base64_audio).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


def test_tokenize_transcribe_with_lang(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    num_expected_frames = int(np.ceil(duration * frame_rate))

    request = get_transcription_request(duration, language=LanguageAlpha2("en"))

    tokenized = tekkenizer.encode_transcription(request)
    BOS, _, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, TRANSCRIBE = _get_specials(tekkenizer)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        *audio_toks,
        END_INST,
        208,
        197,
        210,
        203,
        158,
        201,
        210,
        TRANSCRIBE,
    ]
    text = decode_keep(tekkenizer, tokenized)
    assert text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST]lang:en[TRANSCRIBE]")
    assert len(tokenized.audios) == 1
    base64_audio = request.audio
    assert isinstance(base64_audio, str)
    audio_array = Audio.from_base64(base64_audio).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


TRANSCRIPTION_ERROR_CASES = (
    pytest.param(
        None,
        undecodable_transcription_request,
        UnsupportedTokenizerFeatureException,
        r"Tokenizer v7 does not provide an audio encoder for transcription",
        id="audio-v7-transcription-no-encoder-test",
    ),
    pytest.param(
        SYNTHETIC_V7_INSTRUCT_NO_TRANSCRIBE,
        undecodable_transcription_request,
        UnsupportedTokenizerFeatureException,
        r"the TRANSCRIBE marker required for transcription",
        id="audio-v7-transcription-no-marker-test",
    ),
    pytest.param(
        SYNTHETIC_V7_INSTRUCT_NO_AUDIO,
        valid_transcription_request,
        UnsupportedTokenizerFeatureException,
        r"the audio marker required for transcription",
        id="audio-v7-instruct-transcription-no-audio-test",
    ),
    pytest.param(
        SYNTHETIC_V7_INSTRUCT_NO_BEGIN_AUDIO,
        valid_transcription_request,
        UnsupportedTokenizerFeatureException,
        r"the begin_audio marker required for transcription",
        id="audio-v7-instruct-transcription-no-begin-audio-test",
    ),
    pytest.param(
        SYNTHETIC_V7_INSTRUCT,
        lambda: TranscriptionRequest(
            audio=b"not decodable audio",
            streaming=StreamingMode.OFFLINE,
            language=None,
            target_streaming_delay_ms=None,
        ),
        InvalidRequestException,
        r"^Instruct transcription requires streaming mode DISABLED, got OFFLINE\.$",
        id="audio-v7-instruct-offline-rejected-test",
    ),
    pytest.param(
        SYNTHETIC_V7_INSTRUCT,
        lambda: TranscriptionRequest(
            audio=valid_reference_audio(),
            streaming=StreamingMode.ONLINE,
            language=None,
            target_streaming_delay_ms=None,
        ),
        InvalidRequestException,
        r"Instruct transcription.*DISABLED.*ONLINE",
        id="audio-v7-instruct-online-rejected-test",
    ),
)


@pytest.mark.parametrize(("profile", "request_factory", "error_type", "message"), TRANSCRIPTION_ERROR_CASES)
def test_direct_transcription_errors(
    profile: SyntheticV7AudioProfile | None,
    request_factory: Callable[[], TranscriptionRequest],
    error_type: type[Exception],
    message: str,
) -> None:
    tokenizer = load_v7_audio_error_tokenizer(profile=profile)
    assert tokenizer.mode == ValidationMode.test
    request = request_factory()

    with pytest.raises(error_type, match=message):
        tokenizer.instruct_tokenizer.encode_transcription(request)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        pytest.param(
            MistralTokenizer.v1,
            r"^Transcription not available for (?:TokenizerVersion\.)?v1$",
            id="v1-transcription",
        ),
        pytest.param(
            MistralTokenizer.v2,
            r"^Transcription not available for (?:TokenizerVersion\.)?v2$",
            id="v2-transcription",
        ),
        pytest.param(
            MistralTokenizer.v3,
            r"^Transcription not available for (?:TokenizerVersion\.)?v3$",
            id="v3-transcription",
        ),
    ],
)
def test_pre_v7_transcription_remains_tokenizer_error(factory: Callable[[], MistralTokenizer], message: str) -> None:
    tokenizer = factory()
    request = TranscriptionRequest(audio=b"audio", language=None, target_streaming_delay_ms=None)
    with pytest.raises(TokenizerException, match=message):
        tokenizer.instruct_tokenizer.encode_transcription(request)


# --- streaming ---


def load_audio_streaming() -> Tekkenizer:
    special_tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v7, add_audio=True)
    for i in range(37, 39):
        special_tokens += [SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True)]

    special_tokens += [SpecialTokenInfo(rank=39, token_str="[STREAMING_PAD]", is_control=True)]
    special_tokens += [SpecialTokenInfo(rank=40, token_str="[STREAMING_WORD]", is_control=True)]
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
        transcription_delay_ms=480.0,
        transcription_format=TranscriptionFormat.STREAMING,
        streaming_look_ahead_ms=2.5,
        streaming_look_back_ms=52.5,
        streaming_n_left_pad_tokens=16,
    )
    tokenizer._audio_config = audio_config
    return tokenizer


@pytest.fixture(scope="module")
def streaming_tokenizer() -> InstructTokenizerV7:
    tekkenizer = load_audio_streaming()
    assert tekkenizer.audio is not None
    audio_encoder = load_audio_encoder(tekkenizer.audio, tekkenizer)
    return InstructTokenizerV7(tekkenizer, image_encoder=None, audio_encoder=audio_encoder)


def test_special_audio_streaming_tokens(streaming_tokenizer: InstructTokenizerV7) -> None:
    assert streaming_tokenizer.tokenizer.get_special_token(SpecialTokens.streaming_pad.value) == 39
    assert streaming_tokenizer.tokenizer.get_special_token(SpecialTokens.streaming_word.value) == 40

    decoded_ids = [211, 218, 39, 194, 189, 39, 40, 172, 39, 191, 40, 2]
    assert streaming_tokenizer.decode(decoded_ids) == "ov^YH["
    assert (
        streaming_tokenizer.decode(decoded_ids, special_token_policy=SpecialTokenPolicy.KEEP)
        == "ov[STREAMING_PAD]^Y[STREAMING_PAD][STREAMING_WORD]H[STREAMING_PAD][[STREAMING_WORD]</s>"
    )


@pytest.mark.parametrize("target_streaming_delay_ms", [None, 0, 1, 480, 960, 2400])
@pytest.mark.parametrize(
    ("mode", "duration"),
    [
        (StreamingMode.OFFLINE, 1.7),  # normal
        (StreamingMode.ONLINE, 1.7),  # normal (would correspond to long delay)
        (StreamingMode.OFFLINE, 8.44),
        (StreamingMode.OFFLINE, 0.24),  # very short
        (StreamingMode.ONLINE, 0.24),  # very short delay
        (StreamingMode.OFFLINE, 2.312231),  # weird length can correctly be padded
        (StreamingMode.ONLINE, 2.312231),  # irregular length is still a successful encode
    ],
)
def test_tokenize_streaming_request(
    streaming_tokenizer: InstructTokenizerV7,
    mode: StreamingMode,
    duration: float,
    target_streaming_delay_ms: int | None,
) -> None:
    sampling_rate = 24_000
    signal_length = int(duration * sampling_rate)

    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
        format="wav",
    )

    streaming_request = TranscriptionRequest(
        audio=audio.to_base64("wav") if mode == StreamingMode.OFFLINE else "",
        streaming=mode,
        target_streaming_delay_ms=target_streaming_delay_ms,
        language=None,
    )

    tokenized = streaming_tokenizer.encode_transcription(streaming_request)
    assert streaming_tokenizer.audio_encoder is not None
    config = streaming_tokenizer.audio_encoder.audio_config
    delay_n_tokens = config.get_num_delay_tokens(target_streaming_delay_ms) + config.n_left_pad_tokens

    BOS = streaming_tokenizer.tokenizer.get_special_token(SpecialTokens.bos.value)
    STREAMING_PAD = streaming_tokenizer.tokenizer.get_special_token(SpecialTokens.streaming_pad.value)
    assert tokenized.tokens == ([BOS] + delay_n_tokens * [STREAMING_PAD]), f"{tokenized.tokens}"
    text = decode_keep(streaming_tokenizer, tokenized)
    assert text == "<s>" + delay_n_tokens * "[STREAMING_PAD]"

    if mode == StreamingMode.OFFLINE:
        rounded_duration = math.ceil(duration / (config.frame_duration_ms / 1000)) * (config.frame_duration_ms / 1000)
        audio_array = tokenized.audios[0].audio_array
        assert audio_array.shape == (
            rounded_duration * config.sampling_rate
            + (config.n_left_pad_tokens + config.n_right_pad_tokens(target_streaming_delay_ms))
            * config.raw_audio_length_per_tok,
        )
        audio_len_in_tokens = streaming_tokenizer.audio_encoder.audio_config.num_audio_tokens(int(audio_array.shape[0]))
        assert audio_len_in_tokens == (
            rounded_duration * streaming_tokenizer.audio_encoder.audio_config.frame_rate
            + len(tokenized.tokens)
            + OFFLINE_STREAMING_BUFFER_TOKENS
        )
        # the offline waveform content matches the fresh source signal after resampling to the
        # encoder rate, embedded between the left and right zero padding
        expected = Audio(audio_array=audio.audio_array.copy(), sampling_rate=sampling_rate, format="wav")
        expected.resample(config.sampling_rate)
        left = config.n_left_pad_tokens * config.raw_audio_length_per_tok
        assert np.allclose(audio_array[left : left + len(expected.audio_array)], expected.audio_array, atol=1e-3)
        np.testing.assert_array_equal(audio_array[:left], 0.0)
        np.testing.assert_array_equal(audio_array[left + len(expected.audio_array) :], 0.0)
    elif mode == StreamingMode.ONLINE:
        assert len(tokenized.audios) == 2, tokenized.audios
        left_audio, right_audio = tokenized.audios
        assert left_audio.duration == config.n_left_pad_tokens / config.frame_rate
        assert right_audio.duration == config.n_right_pad_tokens(target_streaming_delay_ms) / config.frame_rate
        # online pad audios are silent and ordered left then right
        np.testing.assert_array_equal(left_audio.audio_array, 0.0)
        np.testing.assert_array_equal(right_audio.audio_array, 0.0)
    else:
        raise ValueError(f"Unknown mode {mode}")


STREAMING_TRANSCRIPTION_ERROR_CASES = (
    pytest.param(
        SYNTHETIC_V7_STREAMING_NO_PAD,
        "",
        StreamingMode.ONLINE,
        UnsupportedTokenizerFeatureException,
        r"the streaming_pad marker required for transcription",
        id="audio-v7-streaming-transcription-no-streaming-pad-test",
    ),
    pytest.param(
        SYNTHETIC_V7_STREAMING,
        b"not decodable audio",
        StreamingMode.DISABLED,
        InvalidRequestException,
        r"^Streaming transcription requires mode OFFLINE or ONLINE, got DISABLED\.$",
        id="audio-v7-streaming-disabled-rejected-test",
    ),
)


@pytest.mark.parametrize(
    ("profile", "audio", "streaming", "error_type", "message"),
    STREAMING_TRANSCRIPTION_ERROR_CASES
    + (
        pytest.param(
            SYNTHETIC_V7_STREAMING_NO_PAD,
            valid_reference_audio(),
            StreamingMode.OFFLINE,
            UnsupportedTokenizerFeatureException,
            r"^Audio encoder does not provide the streaming_pad marker required for transcription\.$",
            id="v7-streaming-profile-without-streaming-pad-marker-offline",
        ),
    ),
)
def test_direct_streaming_transcription_errors(
    profile: SyntheticV7AudioProfile,
    audio: str | bytes,
    streaming: StreamingMode,
    error_type: type[Exception],
    message: str,
) -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=profile)
    request = TranscriptionRequest(
        audio=audio,
        streaming=streaming,
        language=None,
        target_streaming_delay_ms=None,
    )

    with pytest.raises(error_type, match=message):
        tokenizer.instruct_tokenizer.encode_transcription(request)


def test_online_nonempty_bytes_rejected_at_construction() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            r"Value error, ONLINE streaming audio must be a base64 text value or empty prefill; received bytes\. "
            r"\[type=value_error"
        ),
    ):
        TranscriptionRequest(
            audio=valid_reference_audio_bytes(),
            streaming=StreamingMode.ONLINE,
            language=None,
            target_streaming_delay_ms=None,
        )


def test_online_empty_bytes_prefill_still_constructs() -> None:
    request = TranscriptionRequest(
        audio=b"",
        streaming=StreamingMode.ONLINE,
        language=None,
        target_streaming_delay_ms=None,
    )

    assert request.audio == b""


def test_online_from_openai_request_still_constructs() -> None:
    request = TranscriptionRequest.from_openai(
        openai_request={"file": io.BytesIO(valid_reference_audio_bytes()), "streaming": "online"}
    )

    assert isinstance(request.audio, str)
    assert request.audio
    assert request.streaming == StreamingMode.ONLINE


def test_online_base64_audio_is_rejected_by_the_tokenizer() -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_STREAMING)
    input_audio = Audio(
        audio_array=np.linspace(-0.75, 0.75, 2_400, dtype=np.float32),
        sampling_rate=24_000,
        format="wav",
    )
    request = TranscriptionRequest(
        audio=input_audio.to_base64("wav"),
        streaming=StreamingMode.ONLINE,
        language=None,
        target_streaming_delay_ms=None,
    )

    with pytest.raises(InvalidRequestException, match="Online streaming transcription requests must not contain audio"):
        tokenizer.instruct_tokenizer.encode_transcription(request)


# --- speech ---


def test_audio_encoder_speech_request_requires_source() -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_SPEECH)
    assert isinstance(tokenizer.instruct_tokenizer.audio_encoder, AudioEncoder)

    with pytest.raises(
        InvalidRequestException,
        match=r"^Either audio or voice must be defined to encode speech\.$",
    ):
        tokenizer.instruct_tokenizer.audio_encoder.encode_audio_for_speech_request(audio=None, voice=None)


@pytest.fixture(scope="module")
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


@pytest.mark.parametrize(
    ("voice", "num_voice_tokens"),
    [("female", 52), ("male", 76)],
    ids=["voice-female", "voice-male"],
)
def test_encode_speech_request_with_voice_preset(
    tts_tokenizer: InstructTokenizerV7, voice: str, num_voice_tokens: int
) -> None:
    request = SpeechRequest(input="Hello world", voice=voice)
    tokenized = tts_tokenizer.encode_speech_request(request)

    BOS = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.bos.value)
    AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.begin_audio.value)
    NEXT_AUDIO_TEXT = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.text_to_audio.value)
    REPEAT_AUDIO_TEXT = tts_tokenizer.tokenizer.get_special_token(SpecialTokens.audio_to_text.value)

    text_tokens = tts_tokenizer.tokenizer.encode("Hello world", bos=False, eos=False)

    expected = (
        [BOS]
        + [BEGIN_AUDIO]
        + [AUDIO] * num_voice_tokens
        + [NEXT_AUDIO_TEXT]
        + text_tokens
        + [REPEAT_AUDIO_TEXT]
        + [BEGIN_AUDIO]
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


def test_encode_speech_request_neither_audio_nor_voice_fails() -> None:
    with pytest.raises(
        ValidationError,
        match=r"Value error, Either ref_audio or voice must be defined\. \[type=value_error",
    ):
        SpeechRequest(input="Hello world")


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
    # the returned waveform matches the resampled source samples
    expected = _make_fake_audio(duration, source_sr)
    expected.resample(target_sr)
    assert np.allclose(tokenized.audios[0].audio_array, expected.audio_array, atol=1e-3)


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
        AudioDecodeException,
        r"Unknown voice 'not-configured', expected one of \['preset'\]",
        id="audio-v7-speech-unknown-voice-test",
    ),
)


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


def test_direct_invalid_speech_request() -> None:
    tokenizer = build_synthetic_v7_audio_tokenizer(profile=SYNTHETIC_V7_SPEECH)
    request = unknown_voice_speech_request()

    with pytest.raises(InvalidRequestException, match=r"Unknown voice 'not-configured', expected one of \['preset'\]"):
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

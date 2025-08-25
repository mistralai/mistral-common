import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURL,
    AudioURLChunk,
    RawAudio,
    TextChunk,
    UserContentChunk,
)
from mistral_common.protocol.instruct.messages import (
    UATS,
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import (
    InstructRequest,
    SpecialTokens,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV7,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import _quick_vocab, get_special_tokens


def _get_tekkenizer_with_audio() -> InstructTokenizerV7:
    special_tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v7, add_audio=True)
    tokenizer = Tekkenizer(
        _quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
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
        audio=tokenizer.get_control_token(SpecialTokens.audio.value),
        begin_audio=tokenizer.get_control_token(SpecialTokens.begin_audio.value),
    )
    audio_encoder = AudioEncoder(audio_config, special_audio_ids)

    return InstructTokenizerV7(tokenizer, audio_encoder=audio_encoder)


@pytest.fixture(scope="session")
def tekkenizer() -> InstructTokenizerV7:
    return _get_tekkenizer_with_audio()


def sin_wave(sampling_rate: int, duration: float) -> np.ndarray:
    return np.sin(np.ones([int(duration * sampling_rate)]))


@pytest.fixture(scope="session")
def sample_audio() -> Audio:
    sampling_rate = 44100
    original_array = sin_wave(sampling_rate, 1)

    audio = Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
        format="wav",
    )
    return audio


@pytest.fixture(scope="session")
def sample_audio_file(sample_audio: Audio) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, sample_audio.audio_array, sample_audio.sampling_rate)
        return Path(f.name)


@pytest.fixture(scope="session")
def sample_audio_url_chunk_path(sample_audio_file: Path) -> AudioURLChunk:
    return AudioURLChunk(audio_url=str(sample_audio_file))


@pytest.fixture(scope="session")
def sample_audio_url_chunk_path_file(sample_audio_file: Path) -> AudioURLChunk:
    return AudioURLChunk(audio_url=AudioURL(url=f"file://{sample_audio_file}"))


@pytest.fixture(scope="session")
def sample_audio_url_chunk_http() -> AudioURLChunk:
    return AudioURLChunk(audio_url=AudioURL(url="http://example.com/audio.wav"))


@pytest.fixture(scope="session")
def sample_audio_url_chunk_https() -> AudioURLChunk:
    return AudioURLChunk(audio_url=AudioURL(url="https://example.com/audio.wav"))


@pytest.fixture(scope="session")
def sample_audio_url_chunk_base64(sample_audio: Audio) -> AudioURLChunk:
    return AudioURLChunk(audio_url=AudioURL(url=sample_audio.to_base64("wav")))


@pytest.fixture(scope="session")
def sample_audio_url_chunk_base64_prefix(sample_audio: Audio) -> AudioURLChunk:
    return AudioURLChunk(audio_url=AudioURL(url=f"data:audio/wav;base64,{sample_audio.to_base64('wav')}"))


def _get_audio_chunk(duration: float) -> AudioChunk:
    format = "wav"

    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)
    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
        format=format,
    )

    return AudioChunk(
        input_audio=RawAudio(
            data=audio.to_base64(format),
            format=format,
        ),
    )


def _get_specials(tekkenizer: InstructTokenizerV7) -> tuple[int, ...]:
    BOS = tekkenizer.tokenizer.get_control_token(SpecialTokens.bos.value)
    EOS = tekkenizer.tokenizer.get_control_token(SpecialTokens.eos.value)
    BEGIN_INST = tekkenizer.tokenizer.get_control_token(SpecialTokens.begin_inst.value)
    END_INST = tekkenizer.tokenizer.get_control_token(SpecialTokens.end_inst.value)
    AUDIO = tekkenizer.tokenizer.get_control_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = tekkenizer.tokenizer.get_control_token(SpecialTokens.begin_audio.value)
    TRANSCRIBE = tekkenizer.tokenizer.get_control_token(SpecialTokens.transcribe.value)
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
    assert tokenized.text == ("<s>[INST]a[BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST]c b d</s>")
    assert len(tokenized.audios) == 1
    base64_str = audio_chunk.input_audio.data
    assert isinstance(base64_str, str)
    audio_array = Audio.from_base64(base64_str).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


@pytest.mark.parametrize("audio_first", [True, False])
def test_tokenize_user_message(tekkenizer: InstructTokenizerV7, audio_first: bool) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration)
    text_chunk = TextChunk(text="a")

    num_expected_frames = int(np.ceil(duration * frame_rate))
    chunks: List[UserContentChunk] = [audio_chunk, text_chunk] if audio_first else [text_chunk, audio_chunk]

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
        assert tokenized.text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "a[/INST]")
    else:
        assert tokenized.tokens == [
            BOS,
            BEGIN_INST,
            197,  # "a"
            *audio_toks,
            END_INST,
        ]
        assert tokenized.text == ("<s>[INST]a[BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST]")
    assert len(tokenized.audios) == 1
    base64_str = audio_chunk.input_audio.data
    assert isinstance(base64_str, str)
    audio_array = Audio.from_base64(base64_str).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


def test_tokenize_multi_turn(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration)
    text_chunk = TextChunk(text="a")

    num_expected_frames = int(np.ceil(duration * frame_rate))
    chunks: List[UserContentChunk] = [audio_chunk, text_chunk]

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


def test_no_audio_in_system_message_before_v7() -> None:
    path = str(MistralTokenizer._data_path() / "tekken_240911.json")
    tokenizer = MistralTokenizer.from_file(path).instruct_tokenizer

    duration = 1.7  # seconds
    audio_chunk = _get_audio_chunk(duration)
    with pytest.raises(NotImplementedError):
        # we don't allow system
        tokenizer.encode_instruct(
            InstructRequest(
                messages=[SystemMessage(content="hello")],
            )
        )

    with pytest.raises(AssertionError):
        # we don't allow audio
        tokenizer.encode_instruct(
            InstructRequest(
                messages=[
                    UserMessage(content=[audio_chunk]),
                ]
            )
        )


@pytest.mark.parametrize(
    ("messages", "match_regex"),
    [
        (
            [
                SystemMessage(content="a b c d"),
                UserMessage(content=[TextChunk(text="a"), DUMMY_AUDIO]),
            ],
            "System messages are not yet allowed when audio is present",
        ),
    ],
)
def test_tokenize_audio_raise(tekkenizer: InstructTokenizerV7, messages: List[UATS], match_regex: str) -> None:
    with pytest.raises(ValueError, match=match_regex):
        tekkenizer.encode_instruct(InstructRequest(messages=messages))


@pytest.mark.parametrize(
    "audio_url_chunk",
    [
        pytest.param("sample_audio_url_chunk_path", id="path"),
        pytest.param("sample_audio_url_chunk_path_file", id="path_file"),
        pytest.param("sample_audio_url_chunk_http", id="http"),
        pytest.param("sample_audio_url_chunk_https", id="https"),
        pytest.param("sample_audio_url_chunk_base64", id="base64"),
        pytest.param("sample_audio_url_chunk_base64_prefix", id="base64_prefix"),
    ],
)
def test_tokenize_audio_url_chunk(
    tekkenizer: InstructTokenizerV7, request: pytest.FixtureRequest, audio_url_chunk: str
) -> None:
    audio_url_chunk_fixture: AudioURLChunk = request.getfixturevalue(audio_url_chunk)
    audio_url_chunk_path_fixture: AudioURLChunk = request.getfixturevalue("sample_audio_url_chunk_path")

    with patch("mistral_common.audio.requests.get") as mock_get:
        mock_get.return_value.content = Path(audio_url_chunk_path_fixture.url).read_bytes()
        mock_get.return_value.status_code = 200

        tokenized = tekkenizer.encode_instruct(
            InstructRequest(
                messages=[
                    UserMessage(content=[audio_url_chunk_fixture]),
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
    assert tokenized.text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * (len(audio_toks) - 1) + "[/INST]c b d</s>")
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


def test_encode_invalid_audio_url_chunk(tekkenizer: InstructTokenizerV7) -> None:
    assert tekkenizer.audio_encoder is not None
    # Test with an invalid URL
    with pytest.raises(ValueError, match=r"Failed to download audio from URL: https://example.com/invalid_audio.wav"):
        tekkenizer.audio_encoder(AudioURLChunk(audio_url="https://example.com/invalid_audio.wav"))

    # Test with an invalid base64 string
    with pytest.raises(ValueError, match=r"base64 decoding failed. Please check the input string is a valid base64."):
        tekkenizer.audio_encoder(AudioURLChunk(audio_url="data:audio/wav;base64,invalid_base64_string"))

    # Test with an invalide file path
    with pytest.raises(FileNotFoundError, match=r"file='invalid_file_path.wav' does not exist"):
        tekkenizer.audio_encoder(AudioURLChunk(audio_url="file://invalid_file_path.wav"))

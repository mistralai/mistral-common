from typing import List

import numpy as np
import pytest

from mistral_common.audio import AudioFormat
from mistral_common.protocol.instruct.messages import (
    UATS,
    AssistantMessage,
    AudioChunk,
    RawAudio,
    SystemMessage,
    TextChunk,
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
        audio_encoding_config=AudioSpectrogramConfig(
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
    audio_array = Audio.from_base64(audio_chunk.input_audio.data).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


@pytest.mark.parametrize("audio_first", [True, False])
def test_tokenize_user_message(tekkenizer: InstructTokenizerV7, audio_first: bool) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration)
    text_chunk = TextChunk(text="a")

    num_expected_frames = int(np.ceil(duration * frame_rate))
    chunks = [audio_chunk, text_chunk] if audio_first else [text_chunk, audio_chunk]

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
    audio_array = Audio.from_base64(audio_chunk.input_audio.data).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


def test_tokenize_multi_turn(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration)
    text_chunk = TextChunk(text="a")

    num_expected_frames = int(np.ceil(duration * frame_rate))
    chunks = [audio_chunk, text_chunk]

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

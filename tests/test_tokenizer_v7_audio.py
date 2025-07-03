import numpy as np
import pytest
from typing import Optional

from mistral_common.audio import AudioFormat
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    AudioChunk,
    RawAudio,
    SystemMessage,
    TextChunk,
    TranscriptionParams,
    UserMessage,
)
from mistral_common.tokens.tokenizers.audio import (
    Audio,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import (
    InstructRequest,
    SpecialTokens,
)
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV7,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.audio import (
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import InstructRequest, TokenizerVersion
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


def _get_audio_chunk(duration: float, add_transciption: bool = False, language: Optional[str] = None) -> AudioChunk:
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)
    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    format = AudioFormat("WAV")
    transcription_params = TranscriptionParams(language=language) if add_transciption else None

    return AudioChunk(
        input_audio=RawAudio(
            format=format,
            data=audio.to_base64(format=format),
        ),
        transcription_params=transcription_params,
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



def test_tokenize_assistant_message(tekkenizer: InstructTokenizerV7) -> None:
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
                AssistantMessage(
                    content="c b d"
                ),
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
        199, # "c"
        132, # " "
        198, # "b"
        132, # " "
        200, # "d"
        EOS,
    ]
    assert tokenized.text == (
        "<s>[INST]a[BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST]c b d</s>"
    )
    assert len(tokenized.audios) == 1
    audio_array = Audio.from_base64(audio_chunk.input_audio.data).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


def test_no_audio_in_system_message_before_v7() -> None:
    path = str(MistralTokenizer._data_path() / "tekken_240911.json")
    tokenizer = MistralTokenizer.from_file(path).instruct_tokenizer

    duration = 1.7  # seconds
    audio_chunk = _get_audio_chunk(duration)
    text_chunk = TextChunk(text="you are a helpful voice assistant")

    with pytest.raises(NotImplementedError):
        # we don't allow system
        tokenizer.encode_instruct(
            InstructRequest(
                messages=[
                    SystemMessage(content="hello")
                ],
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


def test_tokenize_transcribe(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration, add_transciption=True)
    num_expected_frames = int(np.ceil(duration * frame_rate))

    tokenized = tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(
                    content=[audio_chunk]
                ),
                AssistantMessage(
                    content="a b c d",
                ),
            ],
        )
    )

    BOS, EOS, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, TRANSCRIBE = _get_specials(tekkenizer)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    print(tokenized.tokens)
    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        *audio_toks,
        END_INST,
        TRANSCRIBE,
        1097,  # "a"
        1289,  # " b"
        1272,  # " c"
        1266,  # " d"
        EOS,
    ]
    assert tokenized.text == (
        "<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST][TRANSCRIBE]a b c d</s>"
    )
    assert len(tokenized.audios) == 1
    audio_array = Audio.from_base64(audio_chunk.input_audio.data).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)
    assert len(tokenized.audios_tokens_with_pattern) == 0  # Only used in training.
    assert tokenized.audios_segment_token_sizes == [
        [num_expected_frames],
    ]


def test_tokenize_transcribe_with_lang(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration, add_transciption=True, language="en")
    num_expected_frames = int(np.ceil(duration * frame_rate))

    tokenized = tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(
                    content=[audio_chunk]
                ),
                AssistantMessage(
                    content="a b c d",
                ),
            ],
        )
    )

    BOS, EOS, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, TRANSCRIBE = _get_specials(tekkenizer)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        *audio_toks,
        END_INST,
        9909,  # "lang"
        1058,  # ":"
        1262,  # "en"
        TRANSCRIBE,
        1097,  # "a"
        1289,  # " b"
        1272,  # " c"
        1266,  # " d"
        EOS,
    ]
    assert tokenized.text == (
        "<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST]lang:en[TRANSCRIBE]a b c d</s>"
    )
    assert len(tokenized.audios) == 1
    audio_array = Audio.from_base64(audio_chunk.input_audio.data).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)
    assert len(tokenized.audios_tokens_with_pattern) == 0  # Only used in training.
    assert tokenized.audios_segment_token_sizes == [
        [num_expected_frames],
    ]


def test_tokenize_transcribe_with_lang_and_text_prompt(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration, add_transciption=True, language="en")
    num_expected_frames = int(np.ceil(duration * frame_rate))

    tokenized = tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(
                    content=[
                        audio_chunk,
                        TextChunk(text="a"),
                    ]
                ),
                AssistantMessage(
                    content="a b c d",
                ),
            ],
        )
    )

    BOS, EOS, BEGIN_INST, END_INST, AUDIO, BEGIN_AUDIO, TRANSCRIBE = _get_specials(tekkenizer)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_expected_frames

    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        *audio_toks,
        1097,  # "a"
        END_INST,
        9909,  # "lang"
        1058,  # ":"
        1262,  # "en"
        TRANSCRIBE,
        1097,  # "a"
        1289,  # " b"
        1272,  # " c"
        1266,  # " d"
        EOS,
    ]
    assert tokenized.text == (
        "<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "a[/INST]lang:en[TRANSCRIBE]a b c d</s>"
    )
    assert len(tokenized.audios) == 1
    audio_array = Audio.from_base64(audio_chunk.input_audio.data).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)
    assert len(tokenized.audios_tokens_with_pattern) == 0  # Only used in training.
    assert tokenized.audios_segment_token_sizes == [
        [num_expected_frames],
    ]

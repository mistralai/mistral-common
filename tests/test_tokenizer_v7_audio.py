import json
import re
import tempfile
from pathlib import Path
from typing import Type

import numpy as np
import pytest

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    AudioChunk,
    RawAudio,
    SystemMessage,
    TextChunk,
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
    InstructTokenizerBase,
    InstructTokenizerV3,
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
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer
from tests.test_tekken import _quick_vocab


@pytest.fixture(scope="session")
def tekkenizer() -> InstructTokenizerV7:
    special_tokens = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
    special_tokens += [
        SpecialTokenInfo(rank=24, token_str=SpecialTokens.audio, is_control=True),
        SpecialTokenInfo(rank=25, token_str=SpecialTokens.begin_audio, is_control=True),
        SpecialTokenInfo(rank=34, token_str=SpecialTokens.transcribe, is_control=True),
    ]
    tokenizer = Tekkenizer(
        _quick_vocab([b"a", b"b", b"c", b"f", b"de"]),
        special_tokens,
        pattern=r".+",  # single token, whole string
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v7,
    )
    audio_config = AudioConfig(
    sampling_rate=16000,
        frame_rate=12.5,
        audio_encoding_config=AudioSpectrogramConfig(
            num_mel_bins=128,
            hop_length=160,
            window_size=400,
        ),
        chunk_length_s=30.0,
    )
    special_audio_ids = SpecialAudioIDs(
        audio=tokenizer.get_control_token(SpecialTokens.audio.value),
        begin_audio=tokenizer.get_control_token(SpecialTokens.begin_audio.value),
    )
    audio_encoder = AudioEncoder(audio_config, special_audio_ids)
    return InstructTokenizerV7(tokenizer, audio_encoder=audio_encoder)


def test_tokenize_assistant_message(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)
    frame_rate = 12.5
    assert isinstance(tekkenizer.audio_encoder, AudioEncoder)
    num_expected_frames = int(np.ceil(duration * frame_rate))
    num_exp_audio_special_toks = num_expected_frames

    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    audio_chunk = AudioChunk(
        input_audio=RawAudio(
            format="wav",
            data=audio.to_base64(format="wav"),
        )
    )
    text_chunk = TextChunk(text="can you hear me")

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
                    content=[text_chunk],
                ),
            ],
            output_modalities=[Modality.AUDIO],
        )
    )

    BOS = tekkenizer.tokenizer.get_control_token(SpecialTokens.bos.value)
    EOS = tekkenizer.tokenizer.get_control_token(SpecialTokens.eos.value)
    BEGIN_INST = tekkenizer.tokenizer.get_control_token(SpecialTokens.begin_inst.value)
    END_INST = tekkenizer.tokenizer.get_control_token(SpecialTokens.end_inst.value)
    AUDIO = tekkenizer.tokenizer.get_control_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = tekkenizer.tokenizer.get_control_token(SpecialTokens.begin_audio.value)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_exp_audio_special_toks

    print(tokenized.tokens)
    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        1097,  # a
        *audio_toks,
        END_INST,
        8495,  # can
        1636,  # you
        12459,  # hear
        1639,  # me
        EOS,
    ]
    assert tokenized.text == (
        "<s>[INST]a[BEGIN_AUDIO]" + "[AUDIO]" * num_exp_audio_special_toks + "[/INST]can you hear me</s>"
    )
    assert len(tokenized.audios) == 1
    assert np.allclose(tokenized.audios[0].audio_array, audio.audio_array, atol=1e-3)
    assert len(tokenized.audios_tokens_with_pattern) == 0  # Only used in training
    assert tokenized.audios_segment_token_sizes == [
        [num_expected_frames],
    ]


# AudioTranscriptChunk is not supported when interleaving with spectrogram tokenizers
def test_spectrogram_tokenizer_transcript_chunk_assistant_message(
    tekkenizer_with_spectrogram_and_transcribe: InstructTokenizerV7,
) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)

    rng = np.random.default_rng(0)
    audio_1 = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    audio_2 = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    audio_chunk_1 = AudioChunk(
        input_audio=RawAudio(
            format="wav",
            data=audio_1.to_base64(format="wav"),
        )
    )
    audio_chunk_w_transcript = AudioTranscriptChunk(
        output_audio=RawAudio(
            format="wav",
            data=audio_2.to_base64(format="wav"),
        ),
        transcript="can you hear me",
    )
    assert isinstance(tekkenizer_with_spectrogram_and_transcribe.audio_encoder, AudioEncoder)
    assert isinstance(tekkenizer_with_spectrogram_and_transcribe.audio_encoder.encoding_config, AudioSpectrogramConfig)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Encoding AudioTranscriptChunk requires using a codebook config for encoding_config, which is mutually exclusive with a spectrogram config. Got a spectrogram config: {tekkenizer_with_spectrogram_and_transcribe.audio_encoder.encoding_config}"
        ),
    ):
        tekkenizer_with_spectrogram_and_transcribe.encode_instruct(
            InstructRequest(
                messages=[
                    UserMessage(
                        content=[
                            TextChunk(
                                text="a",
                            ),
                            audio_chunk_1,
                        ]
                    ),
                    AssistantMessage(
                        content=[
                            audio_chunk_w_transcript,
                        ],
                    ),
                ],
                output_modalities=[Modality.AUDIO],
            )
        )


# AudioTranscriptChunk is only supported when interleaving with audio codebook patterns
def test_codebook_tokenizer_transcript_chunk_assistant_message(
    tekkenizer_with_codebook_and_transcribe: InstructTokenizerV7,
) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)
    frame_rate = 12.5
    num_codebooks = 9
    num_expected_frames = int(np.ceil(duration * frame_rate))
    num_exp_audio_special_toks = num_expected_frames + (num_codebooks - 1)

    rng = np.random.default_rng(0)
    audio_1 = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    audio_2 = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    audio_chunk_1 = AudioChunk(
        input_audio=RawAudio(
            format="wav",
            data=audio_1.to_base64(format="wav"),
        )
    )
    audio_chunk_w_transcript = AudioTranscriptChunk(
        output_audio=RawAudio(
            format="wav",
            data=audio_2.to_base64(format="wav"),
        ),
        transcript="can you hear me",
    )

    tokenized = tekkenizer_with_codebook_and_transcribe.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(
                    content=[
                        TextChunk(
                            text="a",
                        ),
                        audio_chunk_1,
                    ]
                ),
                AssistantMessage(
                    content=[
                        audio_chunk_w_transcript,
                    ],
                ),
            ],
            output_modalities=[Modality.AUDIO],
        )
    )

    BOS = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.bos.value)
    EOS = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.eos.value)
    BEGIN_INST = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.begin_inst.value)
    END_INST = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.end_inst.value)
    AUDIO = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.begin_audio.value)
    OUTPUT_AUDIO = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.output_audio.value)
    BEGIN_TRANSCRIPT = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(
        SpecialTokens.begin_transcript.value
    )
    END_TRANSCRIPT = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(
        SpecialTokens.end_transcript.value
    )

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_exp_audio_special_toks
    audio_segment_toks = [BEGIN_AUDIO] + [AUDIO] * (3 + 8)
    trailing_audio_segment_toks = [BEGIN_AUDIO] + [AUDIO] * (num_exp_audio_special_toks - 3 * (1 + 6))

    print(tokenized.tokens)
    assert tokenized.tokens == [
        BOS,
        BEGIN_INST,
        1097,  # a
        *audio_toks,
        END_INST,
        OUTPUT_AUDIO,
        BEGIN_TRANSCRIPT,
        8495,
        1636,
        *audio_segment_toks,
        12459,
        1639,
        END_TRANSCRIPT,
        *audio_segment_toks,
        *audio_segment_toks,
        *audio_segment_toks,
        *audio_segment_toks,
        *audio_segment_toks,
        *audio_segment_toks,
        *trailing_audio_segment_toks,
        EOS,
    ]
    tokenized_audio_segment_text = "[BEGIN_AUDIO]" + "[AUDIO]" * (3 + 8)
    assert tokenized.text == (
        "<s>[INST]a[BEGIN_AUDIO]"
        + "[AUDIO]" * num_exp_audio_special_toks
        + "[/INST][OUTPUT_AUDIO][BEGIN_TRANSCRIPT]can you"
        + tokenized_audio_segment_text
        + " hear me"
        + "[END_TRANSCRIPT]"
        + tokenized_audio_segment_text
        + tokenized_audio_segment_text
        + tokenized_audio_segment_text
        + tokenized_audio_segment_text
        + tokenized_audio_segment_text
        + tokenized_audio_segment_text
        + "[BEGIN_AUDIO]"
        # remaining tokens after 1 interleaved audio chunk
        + "[AUDIO]" * (num_exp_audio_special_toks - 3 * (1 + 6))
        + "</s>"
    )
    assert len(tokenized.audios) == 2
    assert np.allclose(tokenized.audios[0].audio_array, audio_1.audio_array, atol=1e-3)
    assert np.allclose(tokenized.audios[1].audio_array, audio_2.audio_array, atol=1e-3)
    assert len(tokenized.audios_tokens_with_pattern) == 0  # Only used in training.
    assert tokenized.audios_segment_token_sizes == [
        [num_expected_frames],
        [3, 3, 3, 3, 3, 3, 3, 1],
    ]


def test_tokenize_audio_system_message(tekkenizer_with_codebook_and_transcribe: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)
    frame_rate = 12.5
    num_codebooks = 9
    num_expected_frames = int(np.ceil(duration * frame_rate))
    num_exp_audio_special_toks = num_expected_frames + (num_codebooks - 1)

    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    text_chunk_1 = TextChunk(text="a")
    text_chunk_2 = TextChunk(text="b")
    text_chunk_3 = TextChunk(text="c")
    audio_chunk = AudioChunk(
        input_audio=RawAudio(
            format="wav",
            data=audio.to_base64(format="wav"),
        )
    )
    audio_chunk_w_transcript = AudioTranscriptChunk(
        output_audio=RawAudio(
            format="wav",
            data=audio.to_base64(format="wav"),
        ),
        transcript="can you hear me",
    )

    tokenized = tekkenizer_with_codebook_and_transcribe.encode_instruct(
        InstructRequest(
            messages=[
                SystemMessage(content=[text_chunk_1, audio_chunk, text_chunk_2]),
                UserMessage(content=[text_chunk_3]),
                AssistantMessage(content=[audio_chunk_w_transcript]),
            ],
            output_modalities=[Modality.AUDIO],
        )
    )
    BOS = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.bos.value)
    EOS = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.eos.value)
    BEGIN_INST = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.begin_inst.value)
    END_INST = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.end_inst.value)
    AUDIO = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.begin_audio.value)
    OUTPUT_AUDIO = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.output_audio.value)
    BEGIN_TRANSCRIPT = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(
        SpecialTokens.begin_transcript.value
    )
    END_TRANSCRIPT = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(
        SpecialTokens.end_transcript.value
    )
    BEGIN_SYS = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.begin_system.value)
    END_SYS = tekkenizer_with_codebook_and_transcribe.tokenizer.get_control_token(SpecialTokens.end_system.value)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_exp_audio_special_toks
    audio_segment_toks = [BEGIN_AUDIO] + [AUDIO] * (3 + 8)
    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_exp_audio_special_toks
    audio_segment_toks = [BEGIN_AUDIO] + [AUDIO] * (3 + 8)
    trailing_audio_segment_toks = [BEGIN_AUDIO] + [AUDIO] * (num_exp_audio_special_toks - 3 * (1 + 6))
    assert tokenized.tokens == [
        BOS,
        BEGIN_SYS,
        1097,
        *audio_toks,
        1098,
        END_SYS,
        BEGIN_INST,
        1099,  # a
        END_INST,
        OUTPUT_AUDIO,
        BEGIN_TRANSCRIPT,
        8495,
        1636,
        *audio_segment_toks,
        12459,
        1639,
        END_TRANSCRIPT,
        *audio_segment_toks,
        *audio_segment_toks,
        *audio_segment_toks,
        *audio_segment_toks,
        *audio_segment_toks,
        *audio_segment_toks,
        *trailing_audio_segment_toks,
        EOS,
    ]
    assert tokenized.audios_segment_token_sizes == [[22], [3, 3, 3, 3, 3, 3, 3, 1]]
    assert len(tokenized.audios) == 2


def test_tokenize_audio_system_message_needs_audio_output(parameterized_tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)

    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    text_chunk_1 = TextChunk(text="you are a helpful voice assistant")
    text_chunk_2 = TextChunk(text="speak in this natural voice")
    audio_chunk = AudioChunk(
        input_audio=RawAudio(
            format="wav",
            data=audio.to_base64(format="wav"),
        )
    )

    with pytest.raises(AssertionError, match=r".*`output_modalities` should be set to audio.*"):
        # This should raise an error
        parameterized_tekkenizer.encode_instruct(
            InstructRequest(
                messages=[
                    SystemMessage(content=[text_chunk_1, audio_chunk, text_chunk_2]),
                    UserMessage(content=[text_chunk_1]),
                ],
                output_modalities=[Modality.TEXT],
            )
        )

    # This should not raise an error
    parameterized_tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                SystemMessage(content=[text_chunk_1, audio_chunk, text_chunk_2]),
                UserMessage(content=[text_chunk_1]),
            ],
            output_modalities=[Modality.AUDIO],
        )
    )


def test_tokenize_audio_system_message_multiple_audios_should_fail(
    parameterized_tekkenizer: InstructTokenizerV7,
) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)

    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    text_chunk = TextChunk(text="you are a helpful voice assistant")
    audio_chunk = AudioChunk(
        input_audio=RawAudio(
            format="wav",
            data=audio.to_base64(format="wav"),
        )
    )

    with pytest.raises(AssertionError, match=r".*System message can only have one audio chunk*"):
        parameterized_tekkenizer.encode_instruct(
            InstructRequest(
                messages=[
                    SystemMessage(content=[audio_chunk, text_chunk, audio_chunk]),
                ],
                output_modalities=[Modality.AUDIO],
            )
        )


@pytest.mark.parametrize(
    "tokenizer_cls",
    [InstructTokenizerV3]
)
def test_no_audio_in_system_message_before_v7(tokenizer_cls: Type[InstructTokenizerBase]) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)

    tokenizer = tokenizer_cls(tokenizer=Tekkenizer.from_file(MistralTokenizer._data_path() / "tekken_mm_240930.json"))
    rng = np.random.default_rng(0)
    audio = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    text_chunk = TextChunk(text="you are a helpful voice assistant")
    audio_chunk = AudioChunk(
        input_audio=RawAudio(  # noqa: F821
            format="wav",
            data=audio.to_base64(format="wav"),
        )
    )

    with pytest.raises(AssertionError):
        tokenizer.encode_instruct(
            InstructRequest(
                messages=[
                    SystemMessage(content=[audio_chunk, text_chunk]),
                ],
                output_modalities=[Modality.AUDIO],
            )
        )

    with pytest.raises(AssertionError):
        # We also don't allow text chunks - old tokenizer only supports string content.
        tokenizer.encode_instruct(
            InstructRequest(
                messages=[
                    SystemMessage(content=[text_chunk]),
                ]
            )
        )


def test_tokenize_transcribe(parameterized_tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)
    frame_rate = 12.5
    assert isinstance(parameterized_tekkenizer.audio_encoder, AudioEncoder)
    num_codebooks = 9 if isinstance(parameterized_tekkenizer.audio_encoder.encoding_config, AudioCodebookConfig) else 1
    num_expected_frames = int(np.ceil(duration * frame_rate))
    num_exp_audio_special_toks = num_expected_frames + (num_codebooks - 1)

    rng = np.random.default_rng(0)
    audio_1 = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    audio_chunk_1 = AudioChunk(
        input_audio=RawAudio(
            format="wav",
            data=audio_1.to_base64(format="wav"),
        )
    )

    tokenized = parameterized_tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(
                    content=[
                        audio_chunk_1,
                    ]
                ),
                AssistantMessage(
                    content="a b c d",
                ),
            ],
            transcription_params=TranscriptionParams(language=None),
        )
    )

    BOS = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.bos.value)
    EOS = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.eos.value)
    BEGIN_INST = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.begin_inst.value)
    END_INST = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.end_inst.value)
    AUDIO = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.begin_audio.value)
    TRANSCRIBE = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.transcribe.value)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_exp_audio_special_toks

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
        "<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_exp_audio_special_toks + "[/INST][TRANSCRIBE]a b c d</s>"
    )
    assert len(tokenized.audios) == 1
    assert np.allclose(tokenized.audios[0].audio_array, audio_1.audio_array, atol=1e-3)
    assert len(tokenized.audios_tokens_with_pattern) == 0  # Only used in training.
    assert tokenized.audios_segment_token_sizes == [
        [num_expected_frames],
    ]


def test_tokenize_transcribe_with_lang(parameterized_tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)
    frame_rate = 12.5
    assert isinstance(parameterized_tekkenizer.audio_encoder, AudioEncoder)
    num_codebooks = 9 if isinstance(parameterized_tekkenizer.audio_encoder.encoding_config, AudioCodebookConfig) else 1
    num_expected_frames = int(np.ceil(duration * frame_rate))
    num_exp_audio_special_toks = num_expected_frames + (num_codebooks - 1)

    rng = np.random.default_rng(0)
    audio_1 = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    audio_chunk_1 = AudioChunk(
        input_audio=RawAudio(
            format="wav",
            data=audio_1.to_base64(format="wav"),
        )
    )

    tokenized = parameterized_tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(
                    content=[
                        audio_chunk_1,
                    ]
                ),
                AssistantMessage(
                    content="a b c d",
                ),
            ],
            transcription_params=TranscriptionParams(language="en"),
        )
    )

    BOS = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.bos.value)
    EOS = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.eos.value)
    BEGIN_INST = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.begin_inst.value)
    END_INST = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.end_inst.value)
    AUDIO = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.begin_audio.value)
    TRANSCRIBE = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.transcribe.value)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_exp_audio_special_toks

    print(tokenized.tokens)
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
        "<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_exp_audio_special_toks + "[/INST]lang:en[TRANSCRIBE]a b c d</s>"
    )
    assert len(tokenized.audios) == 1
    assert np.allclose(tokenized.audios[0].audio_array, audio_1.audio_array, atol=1e-3)
    assert len(tokenized.audios_tokens_with_pattern) == 0  # Only used in training.
    assert tokenized.audios_segment_token_sizes == [
        [num_expected_frames],
    ]


def test_tokenize_transcribe_with_lang_and_text_prompt(parameterized_tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    sampling_rate = 24000
    signal_length = int(duration * sampling_rate)
    frame_rate = 12.5
    assert isinstance(parameterized_tekkenizer.audio_encoder, AudioEncoder)
    num_codebooks = 9 if isinstance(parameterized_tekkenizer.audio_encoder.encoding_config, AudioCodebookConfig) else 1
    num_expected_frames = int(np.ceil(duration * frame_rate))
    num_exp_audio_special_toks = num_expected_frames + (num_codebooks - 1)

    rng = np.random.default_rng(0)
    audio_1 = Audio(
        audio_array=rng.uniform(low=-1, high=1, size=[signal_length]),
        sampling_rate=sampling_rate,
    )
    audio_chunk_1 = AudioChunk(
        input_audio=RawAudio(
            format="wav",
            data=audio_1.to_base64(format="wav"),
        )
    )

    tokenized = parameterized_tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(
                    content=[
                        audio_chunk_1,
                        TextChunk(text="a"),
                    ]
                ),
                AssistantMessage(
                    content="a b c d",
                ),
            ],
            transcription_params=TranscriptionParams(language="en"),
        )
    )

    BOS = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.bos.value)
    EOS = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.eos.value)
    BEGIN_INST = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.begin_inst.value)
    END_INST = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.end_inst.value)
    AUDIO = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.audio.value)
    BEGIN_AUDIO = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.begin_audio.value)
    TRANSCRIBE = parameterized_tekkenizer.tokenizer.get_control_token(SpecialTokens.transcribe.value)

    audio_toks = [BEGIN_AUDIO] + [AUDIO] * num_exp_audio_special_toks

    print(tokenized.tokens)
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
        "<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_exp_audio_special_toks + "a[/INST]lang:en[TRANSCRIBE]a b c d</s>"
    )
    assert len(tokenized.audios) == 1
    assert np.allclose(tokenized.audios[0].audio_array, audio_1.audio_array, atol=1e-3)
    assert len(tokenized.audios_tokens_with_pattern) == 0  # Only used in training.
    assert tokenized.audios_segment_token_sizes == [
        [num_expected_frames],
    ]

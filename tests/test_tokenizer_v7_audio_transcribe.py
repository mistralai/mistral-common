from typing import List

import numpy as np
import pytest

from mistral_common.protocol.instruct.messages import (
    UATS,
    AssistantMessage,
    SystemMessage,
    TextChunk,
    UserMessage,
)
from mistral_common.tokens.tokenizers.audio import (
    Audio,
)
from mistral_common.tokens.tokenizers.base import (
    InstructRequest,
)
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV7,
)
from tests.test_tokenizer_v7_audio import (
    DUMMY_AUDIO_WITH_TRANSCRIPTION as DUMMY_AUDIO,
)
from tests.test_tokenizer_v7_audio import (
    _get_audio_chunk,
    _get_specials,
    _get_tekkenizer_with_audio,
)


@pytest.fixture(scope="session")
def tekkenizer() -> InstructTokenizerV7:
    return _get_tekkenizer_with_audio()


def test_tokenize_transcribe(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration, add_transciption=True)
    num_expected_frames = int(np.ceil(duration * frame_rate))

    tokenized = tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content=[audio_chunk]),
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
        TRANSCRIBE,
        197,
        132,
        198,
        132,
        199,
        132,
        200,
        EOS,
    ]
    assert tokenized.text == (
        "<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST][TRANSCRIBE]a b c d</s>"
    )
    assert len(tokenized.audios) == 1
    audio_array = Audio.from_base64(audio_chunk.input_audio.data).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


def test_tokenize_transcribe_with_lang(tekkenizer: InstructTokenizerV7) -> None:
    duration = 1.7  # seconds
    frame_rate = 12.5
    audio_chunk = _get_audio_chunk(duration, add_transciption=True, language="en")
    num_expected_frames = int(np.ceil(duration * frame_rate))

    tokenized = tekkenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content=[audio_chunk]),
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
        208,
        197,
        210,
        203,
        158,
        201,
        210,
        TRANSCRIBE,
        197,
        132,
        198,
        132,
        199,
        132,
        200,
        EOS,
    ]
    assert tokenized.text == (
        "<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST]lang:en[TRANSCRIBE]a b c d</s>"
    )
    assert len(tokenized.audios) == 1
    audio_array = Audio.from_base64(audio_chunk.input_audio.data).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


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
        197,  # "a"
        END_INST,
        208,
        197,
        210,
        203,
        158,
        201,
        210,
        TRANSCRIBE,
        197,
        132,
        198,
        132,
        199,
        132,
        200,
        EOS,
    ]
    assert tokenized.text == (
        "<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "a[/INST]lang:en[TRANSCRIBE]a b c d</s>"
    )
    assert len(tokenized.audios) == 1
    audio_array = Audio.from_base64(audio_chunk.input_audio.data).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


@pytest.mark.parametrize(
    ("messages", "match_regex"),
    [
        (
            [
                UserMessage(content=[TextChunk(text="a"), DUMMY_AUDIO]),
                AssistantMessage(content="a b c d"),
                UserMessage(content="a"),
            ],
            "Transcription request should have at most two messages, not 3",
        ),
        (
            [
                SystemMessage(content="a b c d"),
                UserMessage(content=[TextChunk(text="a"), DUMMY_AUDIO]),
            ],
            "Expected first message to be UserMessage, got system",
        ),
        (
            [
                UserMessage(content=[TextChunk(text="a"), DUMMY_AUDIO, TextChunk(text="a")]),
            ],
            "Transcription request should have at most one text chunk in the user message, not 2",
        ),
        (
            [
                UserMessage(content=[DUMMY_AUDIO, TextChunk(text="a"), DUMMY_AUDIO]),
            ],
            "Only one transcription params is allowed, not 2",
        ),
    ],
)
def test_tokenize_transcribe_raise(tekkenizer: InstructTokenizerV7, messages: List[UATS], match_regex: str) -> None:
    with pytest.raises(ValueError, match=match_regex):
        tekkenizer.encode_instruct(InstructRequest(messages=messages))

from typing import Optional

import numpy as np
import pytest
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import (
    Audio,
)
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV7,
)
from tests.test_tokenizer_v7_audio import (
    _get_audio_chunk,
    _get_specials,
    _get_tekkenizer_with_audio,
)


@pytest.fixture(scope="session")
def tekkenizer() -> InstructTokenizerV7:
    return _get_tekkenizer_with_audio()


def get_transcription_request(duration: float, language: Optional[LanguageAlpha2] = None) -> TranscriptionRequest:
    audio_chunk = _get_audio_chunk(duration)

    return TranscriptionRequest(model="dummy", audio=audio_chunk.input_audio, language=language)


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
    assert tokenized.text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST][TRANSCRIBE]")
    assert len(tokenized.audios) == 1
    base64_audio = request.audio.data
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
    assert tokenized.text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST]lang:en[TRANSCRIBE]")
    assert len(tokenized.audios) == 1
    base64_audio = request.audio.data
    assert isinstance(base64_audio, str)
    audio_array = Audio.from_base64(base64_audio).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)

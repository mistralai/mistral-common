import numpy as np
import pytest
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import (
    Audio,
)
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV7,
)
from tests.test_tokenizer_v7_audio import (
    _get_audio_chunk,
    _get_specials,
    get_tekkenizer_with_audio,
)


@pytest.fixture(scope="session")
def tekkenizer() -> InstructTokenizerV7:
    return get_tekkenizer_with_audio()


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
    assert tokenized.text == ("<s>[INST][BEGIN_AUDIO]" + "[AUDIO]" * num_expected_frames + "[/INST][TRANSCRIBE]")
    assert len(tokenized.audios) == 1
    base64_audio = request.audio
    assert isinstance(base64_audio, str)
    audio_array = Audio.from_base64(base64_audio).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)


def test_to_openai_drops_mistral_specific_fields() -> None:
    audio_chunk = _get_audio_chunk(1.0)
    request = TranscriptionRequest(
        id="some-id",
        model="dummy",
        audio=audio_chunk.input_audio,
        language=LanguageAlpha2("en"),
        temperature=0.5,
        max_tokens=128,
        streaming=StreamingMode.OFFLINE,
        target_streaming_delay_ms=480,
    )

    openai_request = request.to_openai()

    # mistral-common / Mistral-specific fields must not leak into the OpenAI payload.
    for field in ("id", "max_tokens", "strict_audio_validation", "streaming", "target_streaming_delay_ms", "audio"):
        assert field not in openai_request, field

    # OpenAI-valid fields are preserved and audio is exposed as a file buffer.
    assert openai_request["model"] == "dummy"
    assert openai_request["language"] == "en"
    assert openai_request["temperature"] == 0.5
    assert "file" in openai_request


def test_to_openai_exclude_strips_compatible_extension_fields() -> None:
    audio_chunk = _get_audio_chunk(1.0)
    request = TranscriptionRequest(model="dummy", audio=audio_chunk.input_audio, target_streaming_delay_ms=None)

    # `seed` and `top_p` are OpenAI-compatible extension fields (e.g. accepted by vLLM) but not part
    # of OpenAI's hosted transcription API. `exclude` can drop them to get a strictly OpenAI-valid payload.
    openai_request = request.to_openai(exclude=("seed", "top_p"))

    assert "seed" not in openai_request
    assert "top_p" not in openai_request
    # Unrelated OpenAI-valid fields are still present.
    assert openai_request["model"] == "dummy"
    assert "file" in openai_request


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
    base64_audio = request.audio
    assert isinstance(base64_audio, str)
    audio_array = Audio.from_base64(base64_audio).audio_array
    assert np.allclose(tokenized.audios[0].audio_array, audio_array, atol=1e-3)

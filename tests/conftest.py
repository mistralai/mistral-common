import numpy as np
import pytest

from mistral_common.audio import Audio
from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURL, AudioURLChunk, RawAudio


@pytest.fixture(scope="session")
def audio_chunk() -> AudioChunk:
    sampling_rate = 16000
    audio_array = np.zeros(1600)  # 0.1 seconds of silence

    audio = Audio(
        audio_array=audio_array,
        sampling_rate=sampling_rate,
        format="wav",
    )

    return AudioChunk(
        input_audio=RawAudio(
            data=audio.to_base64("wav"),
            format="wav",
        ),
    )


@pytest.fixture(scope="session")
def audio_url_chunk(audio_chunk: AudioChunk) -> AudioURLChunk:
    return AudioURLChunk(audio_url=AudioURL(url=str(audio_chunk.input_audio.data)))

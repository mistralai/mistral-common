import numpy as np

from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURL, AudioURLChunk
from mistral_common.tokens.tokenizers.audio import Audio


def get_dummy_audio_chunk() -> AudioChunk:
    sampling_rate = 16000
    audio_array = np.zeros(1600)

    audio = Audio(
        audio_array=audio_array,
        sampling_rate=sampling_rate,
        format="wav",
    )

    return AudioChunk(input_audio=audio.to_base64("wav"))


def get_dummy_audio_url_chunk() -> AudioURLChunk:
    audio_chunk = get_dummy_audio_chunk()
    return AudioURLChunk(audio_url=AudioURL(url=str(audio_chunk.input_audio)))

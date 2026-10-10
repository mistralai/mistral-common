from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.tokens.tokenizers.audio import Audio

def test_image_chunk_data_url_subtypes():
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    mimes = ["image/png", "image/jpeg", "image/x-icon", "image/svg+xml", "image/vnd.microsoft.icon"]
    for mime in mimes:
        chunk = ImageChunk.from_openai({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{png}"}})
        assert chunk.image is not None

def test_audio_data_url_subtypes():
    fake_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    mimes = ["audio/wav", "audio/x-wav", "audio/mp3", "audio/mpeg"]
    for mime in mimes:
        audio = Audio.from_base64(f"data:{mime};base64,{fake_b64}")
        assert audio is not None

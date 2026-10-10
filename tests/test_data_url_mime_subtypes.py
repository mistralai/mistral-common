from mistral_common.protocol.instruct.chunk import ImageChunk

def test_data_url_mime_subtypes_supported():
    png_payload = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    for mime in ["image/png", "image/jpeg", "image/x-icon", "image/svg+xml", "image/vnd.microsoft.icon"]:
        chunk = ImageChunk.from_openai({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{png_payload}"}})
        assert chunk is not None

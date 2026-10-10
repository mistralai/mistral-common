import io
from PIL import Image
from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.tokens.tokenizers.image import image_from_chunk

def test_exif_orientation_correction():
    img = Image.new('RGB', (200, 100), color='red')
    exif = img.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format='JPEG', exif=exif)
    buf.seek(0)
    
    loaded_img = Image.open(buf)
    chunk = ImageChunk(image=loaded_img)
    processed_img = image_from_chunk(chunk)

    assert processed_img.size == (100, 200)

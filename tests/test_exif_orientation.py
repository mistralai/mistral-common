import io
from PIL import Image
from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.tokens.tokenizers.image import image_from_chunk

def test_exif_orientation_is_applied():
    # Creiamo un'immagine landscape 200x100
    img = Image.new('RGB', (200, 100), color='red')
    exif = img.getexif()
    exif[0x0112] = 6  # Simuliamo che la fotocamera fosse ruotata di 90 gradi (portrait originale 100x200)
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG', exif=exif)
    buf.seek(0)
    loaded_img = Image.open(buf)

    chunk = ImageChunk(image=loaded_img)
    processed_img = image_from_chunk(chunk)

    # L'immagine processata DEVE essere stata raddrizzata automaticamente dal nostro helper
    assert processed_img.size == (100, 200), f"Expected (100, 200), got {processed_img.size}"

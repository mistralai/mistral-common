# Images

Most of the recently released [Mistral models](https://huggingface.co/mistralai/models) support image inputs. Images are represented as [BaseContentChunk][mistral_common.protocol.instruct.messages.BaseContentChunk] objects within the `messages` field of the [ChatCompletionRequest][mistral_common.protocol.instruct.request.ChatCompletionRequest]. Encoding an image via a [ImageEncoder][mistral_common.tokens.tokenizers.image.ImageEncoder] will return:

- a sequence of special tokens representing the image.
- the image normalized as a numpy array.

## Supported image formats

Mistral Image encoders use Pillow to decode images and OpenCV to encode. Hence, the supported formats are the same as Pillow's. The images can be provided as:
- an [ImageURLChunk][mistral_common.protocol.instruct.messages.ImageURLChunk]: a pydantic model containing an image URL from which the image will be downloaded.
- an [ImageChunk][mistral_common.protocol.instruct.messages.ImageChunk]: a pydantic model containing a serialized image that can be either a base64 string or a pillow image.

## Use an Image encoder with our tokenizer

Our tokenizers can an [ImageEncoder][mistral_common.tokens.tokenizers.image.ImageEncoder] that is configured with [ImageConfig][mistral_common.tokens.tokenizers.image.ImageConfig].

The attributes of the [ImageConfig][mistral_common.tokens.tokenizers.image.ImageConfig] configure how the images will be patched into tokens:

- `image_patch_size`: the square size of a patch in pixels to form one token. E.g if the image is 224x224 and the patch size is 14, then the image will be divided into 16x16 patches.
- `max_image_size`: the maximum size of the image in pixels. If the image is larger, it will be resized to this size.
- `spatial_merge_size`: the number of patches to merge into one token. This is useful to reduce the number of redundant tokens in the image. E.g if the image is 224x224 and the patch size is 14, then the image will be divided into 16x16 patches. If the spatial merge size is 2, then the image will be divided into 8x8 patches.

```python
from mistral_common.protocol.instruct.messages import ImageURLChunk
from mistral_common.tokens.tokenizers.image import ImageEncoder, ImageConfig, SpecialImageIDs

special_ids = SpecialImageIDs(img=10, img_break=11, img_end=12)  # These are normally automatically set by the tokenizer

config = ImageConfig(image_patch_size=14, max_image_size=224, spatial_merge_size=2)

image = ImageURLChunk(image_url="https://live.staticflickr.com/7250/7534338696_b33e941b7d_b.jpg")

encoder = ImageEncoder(config, special_ids)
encoder(image)
```

## Tokenize an image

Let's load the tekken tokenizer used for [Mistral Small 3.1's](https://mistral.ai/news/mistral-small-3-1) tokenizer to encode and tokenize an image.

```python
from huggingface_hub import hf_hub_download

from mistral_common.protocol.instruct.messages import ImageURLChunk, TextChunk, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

tokenizer = MistralTokenizer.from_hf_hub(repo_id=model_id, token="your_hf_token")

tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    ImageURLChunk(image_url="https://live.staticflickr.com/7250/7534338696_b33e941b7d_b.jpg"),
                    TextChunk(text="What is displayed in this image?"),
                ]
            )
        ],
    )
)
# Tokenized(tokens=[1, 3, 10, 10, ...], text='<s>[INST][IMG][IMG][IMG][IMG]...', prefix_ids=None, images=[array[[(0.95238595, 0.95238795, 0.95224484, ...,)]]])
```

The output contains:

- the **text**: the string equivalent of the tokens. The image is represented into a sequence of special `[IMG]` tokens with `[IMG_BREAK]` at regular intervals and `[IMG_END]` at the end. An image is a grid and each `[IMG]` represent a patch, the `IMG_BREAK` tokens the end of a row. The `IMG_END` token is used to mark the end of the image.
- the **tokens**: identifier used by the model for the text. The special tokens of images are not directly used by the model, but replaced by the features of an image encoder.
- the **prefix_ids**: Used for FIM (Fill-In-the-Middle) tasks, here it is `None`.
- the **images**: the images normalized as a numpy array.


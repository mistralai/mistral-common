import sys
from typing import Any

import pytest
from PIL import Image

from mistral_common.protocol.instruct.chunk import (
    ContentChunk,
    ImageChunk,
    TextChunk,
)
from mistral_common.protocol.instruct.messages import (
    SystemMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.image import ImageEncoder
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils.requests.instruct import (
    image_alignment_requests,
    single_user_message,
    text_alignment_requests,
    text_requests,
)
from tests.utils.tokenizers import image_token_ids, image_token_spans

v3_tekken_mm = pytest.mark.parametrize("shipped_mistral_tokenizer", ["v3_tekken_mm_small_patch"], indirect=True)
v3_tekken_text = pytest.mark.parametrize("keyed_mistral_tokenizer", ["v3_tekken"], indirect=True)


class TestInstructTokenizerV3Multimodal:
    @pytest.mark.parametrize("r", text_alignment_requests())
    @v3_tekken_text
    @v3_tekken_mm
    def test_agreement_with_text_only(
        self,
        shipped_mistral_tokenizer: MistralTokenizer,
        keyed_mistral_tokenizer: MistralTokenizer,
        r: ChatCompletionRequest,
    ) -> None:
        text_output = keyed_mistral_tokenizer.encode_chat_completion(r)
        mm_output = shipped_mistral_tokenizer.encode_chat_completion(r)

        assert mm_output.tokens == text_output.tokens, f"mm output: {mm_output.tokens}\nexpected: {text_output.tokens}"

    @v3_tekken_mm
    def test_swap_text_image_special_case(self, shipped_mistral_tokenizer: MistralTokenizer) -> None:
        img = Image.new("RGB", (4, 4), "red")
        prompt = "What is on this image?"

        request_text_first: ChatCompletionRequest = ChatCompletionRequest(
            messages=[
                UserMessage(content=[ImageChunk(image=img), TextChunk(text=prompt)]),
            ],
        )
        request_img_first: ChatCompletionRequest = ChatCompletionRequest(
            messages=[
                UserMessage(content=[TextChunk(text=prompt), ImageChunk(image=img)]),
            ],
        )
        assert _are_requests_same(shipped_mistral_tokenizer, [request_text_first, request_img_first])

        # adding one more text or image will lead to different results though
        prompt_2 = "more"

        request_text_first.messages[0].content.append(TextChunk(text=prompt_2))
        request_img_first.messages[0].content.append(TextChunk(text=prompt_2))

        assert not _are_requests_same(shipped_mistral_tokenizer, [request_text_first, request_img_first])

    @pytest.mark.parametrize("r", image_alignment_requests() + text_requests())
    @v3_tekken_mm
    def test_mm_normalizer(
        self,
        shipped_mistral_tokenizer: MistralTokenizer,
        r: ChatCompletionRequest,
    ) -> None:
        r_norm = shipped_mistral_tokenizer._instruct_request_normalizer.from_chat_completion_request(r)

        # filter system messages
        messages = [m for m in r.messages if not isinstance(m, SystemMessage)]
        norm_messages = [m for m in r_norm.messages]

        assert len(messages) == len(norm_messages)
        for message, norm_message in zip(messages, norm_messages):
            if all(isinstance(c, TextChunk) for c in message.content):
                # text-only is collapsed into a single str
                assert isinstance(norm_message.content, str)
            else:
                # image
                if not isinstance(message.content, str):
                    assert not isinstance(message.content, str)
                    assert _count_expected_chunks(message.content) == len(norm_message.content)

    @v3_tekken_mm
    def test_image_tokenization_integration(self, shipped_mistral_tokenizer: MistralTokenizer) -> None:
        # we'll put the test cases inside the test body so we don't have too much PIL stuff
        # outside of functions
        # fmt: off

        # Test cases validated by manually comparing to what you get from the language
        # only tokenizer when you remove all the images from the request. The two should
        # match other than the difference in token ids when something follows vs doesn't
        # follow a \n (which is easy to check)
        requests = image_alignment_requests()
        _im = 10
        _im_break = 12
        _im_end = 13
        img_toks = [_im, _im, _im_break, _im, _im, _im_end]
        expected = [
            [1, 3, *img_toks, 1097, 4],
            [1, 3, 1065, 1398, 1267, *img_toks, 1067, 4],
            [1, 3, 1065, 1398, 1267, *img_toks, 1067, 4],
            [1, 3, *img_toks, *img_toks, 1067, 4, 1068, 2, 3, 1065, 1398, 1267, *img_toks, 1069, *img_toks, 4],
            [1, 3, 1065, *img_toks, 1066, 1267, 1067, *img_toks, 1068, 1267, 1069, 4],
        ]
        # fmt: on
        image_encoder = shipped_mistral_tokenizer.instruct_tokenizer.image_encoder
        assert isinstance(image_encoder, ImageEncoder)

        kw_args: dict[str, Any] = dict(strict=True) if sys.version_info >= (3, 10) else {}
        for r, expected_tokens in zip(requests, expected, **kw_args):
            output: Tokenized = shipped_mistral_tokenizer.encode_chat_completion(r)
            assert output.tokens == expected_tokens, f"Incorrect tokens for request {r}"

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(
                [
                    TextChunk(text=""),
                    ImageChunk(image=Image.new("RGB", (4, 4), "red")),
                    ImageChunk(image=Image.new("RGB", (6, 4), "blue")),
                ],
                id="empty-text-then-two-images",
            ),
            pytest.param(
                [
                    TextChunk(text="x"),
                    ImageChunk(image=Image.new("RGB", (4, 4), "red")),
                    ImageChunk(image=Image.new("RGB", (6, 4), "blue")),
                ],
                id="text-then-two-images",
            ),
            pytest.param(
                [
                    ImageChunk(image=Image.new("RGB", (4, 4), "red")),
                    ImageChunk(image=Image.new("RGB", (6, 4), "blue")),
                ],
                id="two-images",
            ),
        ],
    )
    @v3_tekken_mm
    def test_multi_image_order_is_preserved(
        self, shipped_mistral_tokenizer: MistralTokenizer, content: list[ContentChunk]
    ) -> None:
        image_encoder = shipped_mistral_tokenizer.instruct_tokenizer.image_encoder
        assert isinstance(image_encoder, ImageEncoder)
        tokenized = shipped_mistral_tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=single_user_message(content=content))
        )
        assert image_token_spans(tokenized.tokens, image_encoder.special_ids) == [
            image_token_ids(2, 2, image_encoder.special_ids),
            image_token_ids(3, 2, image_encoder.special_ids),
        ]

    @v3_tekken_mm
    def test_single_trailing_image_moves_first(self, shipped_mistral_tokenizer: MistralTokenizer) -> None:
        image_encoder = shipped_mistral_tokenizer.instruct_tokenizer.image_encoder
        assert isinstance(image_encoder, ImageEncoder)
        tokenized = shipped_mistral_tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=single_user_message(
                    content=[TextChunk(text="x"), ImageChunk(image=Image.new("RGB", (4, 4), "red"))]
                )
            )
        )
        assert image_token_spans(tokenized.tokens, image_encoder.special_ids) == [
            image_token_ids(2, 2, image_encoder.special_ids)
        ]
        x_token = shipped_mistral_tokenizer.instruct_tokenizer.tokenizer.encode("x", bos=False, eos=False)[0]
        assert tokenized.tokens.index(image_encoder.special_ids.img) < tokenized.tokens.index(x_token)

    @v3_tekken_mm
    def test_single_leading_image_remains_first(self, shipped_mistral_tokenizer: MistralTokenizer) -> None:
        image_encoder = shipped_mistral_tokenizer.instruct_tokenizer.image_encoder
        assert isinstance(image_encoder, ImageEncoder)
        tokenized = shipped_mistral_tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=single_user_message(
                    content=[ImageChunk(image=Image.new("RGB", (4, 4), "red")), TextChunk(text="x")]
                )
            )
        )
        assert image_token_spans(tokenized.tokens, image_encoder.special_ids) == [
            image_token_ids(2, 2, image_encoder.special_ids)
        ]
        x_token = shipped_mistral_tokenizer.instruct_tokenizer.tokenizer.encode("x", bos=False, eos=False)[0]
        assert tokenized.tokens.index(image_encoder.special_ids.img) < tokenized.tokens.index(x_token)


def _are_requests_same(mm_tokenizer: MistralTokenizer, requests: list[ChatCompletionRequest]) -> bool:
    assert mm_tokenizer.instruct_tokenizer.image_encoder is not None
    outputs: list[Tokenized] = []
    for request in requests:
        outputs.append(mm_tokenizer.encode_chat_completion(request))

    token_same = outputs[0].tokens == outputs[1].tokens

    return token_same


def _count_expected_chunks(elements: list[ContentChunk]) -> int:
    """
    Count the number of chunks in the list, treating consecutive TextChunks as a single chunk.
    """
    count = 0
    previous_was_text = False

    for element in elements:
        if isinstance(element, TextChunk):
            if not previous_was_text:
                count += 1
                previous_was_text = True
        else:
            count += 1
            previous_was_text = False

    return count

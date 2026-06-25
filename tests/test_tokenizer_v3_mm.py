import sys

import pytest
from PIL import Image

from mistral_common.protocol.instruct.chunk import (
    ContentChunk,
    ImageChunk,
    TextChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import Tokenized
from mistral_common.tokens.tokenizers.image import ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

text_alignment_requests: list[ChatCompletionRequest] = [
    ChatCompletionRequest(
        messages=[
            UserMessage(content="hello"),
            UserMessage(content=[TextChunk(text="bbb"), TextChunk(text="ccc")]),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ],
    ),
    ChatCompletionRequest(
        messages=[
            UserMessage(content="hello"),
        ],
    ),
    ChatCompletionRequest(messages=[UserMessage(content=[TextChunk(text="")])]),
    ChatCompletionRequest(
        messages=[
            SystemMessage(content="You are an AI assistant"),
            UserMessage(content=[TextChunk(text="aaa"), TextChunk(text="bbb")]),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ]
    ),
]

img = Image.new("RGB", (4, 4), "red")
img_requests: list[ChatCompletionRequest] = [
    ChatCompletionRequest(
        messages=[
            UserMessage(content=[TextChunk(text="a"), ImageChunk(image=img)]),
        ],
    ),
    ChatCompletionRequest(
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[TextChunk(text="C"), ImageChunk(image=img)]),
        ]
    ),
    ChatCompletionRequest(
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[ImageChunk(image=img), TextChunk(text="C")]),
        ]
    ),
    ChatCompletionRequest(
        messages=[
            SystemMessage(content="A B"),
            UserMessage(
                content=[
                    ImageChunk(image=img),
                    ImageChunk(image=img),
                    TextChunk(text="C"),
                ]
            ),
            AssistantMessage(content="D"),
            UserMessage(
                content=[
                    ImageChunk(image=img),
                    TextChunk(text="E"),
                    ImageChunk(image=img),
                ]
            ),
        ]
    ),
    ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="A"),
                    ImageChunk(image=img),
                    TextChunk(text="B"),
                    TextChunk(text="C"),
                    ImageChunk(image=img),
                    TextChunk(text="D"),
                    TextChunk(text="E"),
                ]
            ),
        ]
    ),
]
text_requests: list[ChatCompletionRequest] = [
    ChatCompletionRequest(
        messages=[
            UserMessage(content="hello"),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ],
    ),
    ChatCompletionRequest(
        messages=[
            UserMessage(content="hello"),
        ],
    ),
    ChatCompletionRequest(messages=[UserMessage(content=[TextChunk(text="")])]),
    ChatCompletionRequest(
        messages=[
            SystemMessage(content="You are an AI assistant"),
            UserMessage(content=[TextChunk(text="aaa"), TextChunk(text="bbb")]),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ]
    ),
]


def _image_tokens(width: int, height: int, special_ids: SpecialImageIDs) -> list[int]:
    image_tokens = ([special_ids.img] * width + [special_ids.img_break]) * height
    image_tokens[-1] = special_ids.img_end
    return image_tokens


def _set_test_image_patch_size(tokenizer: MistralTokenizer) -> ImageEncoder:
    image_encoder = tokenizer.instruct_tokenizer.image_encoder
    assert isinstance(image_encoder, ImageEncoder)
    image_encoder.image_config.image_patch_size = 2
    return image_encoder


def _image_tokenizer_spans(tokens: list[int], special_ids: SpecialImageIDs) -> list[list[int]]:
    spans: list[list[int]] = []
    start_idx: int | None = None
    for idx, token in enumerate(tokens):
        if start_idx is None:
            if token == special_ids.img:
                start_idx = idx
        elif token == special_ids.img_end:
            spans.append(tokens[start_idx : idx + 1])
            start_idx = None
    return spans


@pytest.fixture
def mm_tokenizer() -> MistralTokenizer:
    path = str(MistralTokenizer._data_path() / "tekken_240911.json")
    tokenizer = MistralTokenizer.from_file(path)
    return tokenizer


@pytest.fixture
def text_tokenizer() -> MistralTokenizer:
    path = str(MistralTokenizer._data_path() / "tekken_240718.json")
    tokenizer = MistralTokenizer.from_file(path)
    return tokenizer


@pytest.fixture
def spm_tokenizer() -> MistralTokenizer:
    return MistralTokenizer.v7(is_mm=True)


@pytest.mark.parametrize("r", text_alignment_requests)
def test_agreement_with_text_only(
    mm_tokenizer: MistralTokenizer,
    text_tokenizer: MistralTokenizer,
    r: ChatCompletionRequest,
) -> None:
    text_output = text_tokenizer.encode_chat_completion(r)
    mm_output = mm_tokenizer.encode_chat_completion(r)

    assert mm_output.tokens == text_output.tokens, f"mm output: {mm_output.tokens}\nexpected: {text_output.tokens}"


def test_swap_text_image_special_case(mm_tokenizer: MistralTokenizer) -> None:
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
    assert are_requests_same(mm_tokenizer, [request_text_first, request_img_first])

    # adding one more text or image will lead to different results though
    prompt_2 = "more"

    request_text_first.messages[0].content.append(TextChunk(text=prompt_2))
    request_img_first.messages[0].content.append(TextChunk(text=prompt_2))

    assert not are_requests_same(mm_tokenizer, [request_text_first, request_img_first])


def are_requests_same(mm_tokenizer: MistralTokenizer, requests: list[ChatCompletionRequest]) -> bool:
    assert mm_tokenizer.instruct_tokenizer.image_encoder is not None
    outputs: list[Tokenized] = []
    for request in requests:
        outputs.append(mm_tokenizer.encode_chat_completion(request))

    token_same = outputs[0].tokens == outputs[1].tokens

    return token_same


@pytest.mark.parametrize("r", img_requests + text_requests)
def test_mm_normalizer(
    mm_tokenizer: MistralTokenizer,
    r: ChatCompletionRequest,
) -> None:
    r_norm = mm_tokenizer._instruct_request_normalizer.from_chat_completion_request(r)

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
                assert count_expected_chunks(message.content) == len(norm_message.content)


def count_expected_chunks(elements: list[ContentChunk]) -> int:
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


def test_image_tokenization_integration(mm_tokenizer: MistralTokenizer) -> None:
    # we'll put the test cases inside the test body so we don't have too much PIL stuff
    # outside of functions
    # fmt: off

    # Test cases validated by manually comparing to what you get from the language
    # only tokenizer when you remove all the images from the request. The two should
    # match other than the difference in token ids when something follows vs doesn't
    # follow a \n (which is easy to check)
    requests = img_requests
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
    image_encoder = mm_tokenizer.instruct_tokenizer.image_encoder
    assert isinstance(image_encoder, ImageEncoder)
    # hardcode image_patch_size = 2
    image_encoder.image_config.image_patch_size = 2

    kw_args = dict(strict=True) if sys.version_info >= (3, 10) else {}
    for r, expected_tokens in zip(requests, expected, **kw_args):
        output: Tokenized = mm_tokenizer.encode_chat_completion(r)
        assert output.tokens == expected_tokens, f"Incorrect tokens for request {r}"


@pytest.mark.parametrize("tokenizer_fixture_name", ["mm_tokenizer", "spm_tokenizer"])
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
def test_multi_image_order_is_preserved(
    request: pytest.FixtureRequest,
    tokenizer_fixture_name: str,
    content: list[ContentChunk],
) -> None:
    tokenizer = request.getfixturevalue(tokenizer_fixture_name)
    image_encoder = _set_test_image_patch_size(tokenizer)
    tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=content)]))
    assert _image_tokenizer_spans(tokenized.tokens, image_encoder.special_ids) == [
        _image_tokens(2, 2, image_encoder.special_ids),
        _image_tokens(3, 2, image_encoder.special_ids),
    ]


@pytest.mark.parametrize("tokenizer_fixture_name", ["mm_tokenizer", "spm_tokenizer"])
def test_single_trailing_image_moves_first(request: pytest.FixtureRequest, tokenizer_fixture_name: str) -> None:
    tokenizer = request.getfixturevalue(tokenizer_fixture_name)
    image_encoder = _set_test_image_patch_size(tokenizer)
    tokenized = tokenizer.encode_chat_completion(
        ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        TextChunk(text="x"),
                        ImageChunk(image=Image.new("RGB", (4, 4), "red")),
                    ]
                )
            ]
        )
    )
    assert _image_tokenizer_spans(tokenized.tokens, image_encoder.special_ids) == [
        _image_tokens(2, 2, image_encoder.special_ids)
    ]
    x_token = tokenizer.instruct_tokenizer.tokenizer.encode("x", bos=False, eos=False)[0]
    assert tokenized.tokens.index(image_encoder.special_ids.img) < tokenized.tokens.index(x_token)


@pytest.mark.parametrize("tokenizer_fixture_name", ["mm_tokenizer", "spm_tokenizer"])
def test_single_leading_image_remains_first(request: pytest.FixtureRequest, tokenizer_fixture_name: str) -> None:
    tokenizer = request.getfixturevalue(tokenizer_fixture_name)
    image_encoder = _set_test_image_patch_size(tokenizer)
    tokenized = tokenizer.encode_chat_completion(
        ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        ImageChunk(image=Image.new("RGB", (4, 4), "red")),
                        TextChunk(text="x"),
                    ]
                )
            ]
        )
    )
    assert _image_tokenizer_spans(tokenized.tokens, image_encoder.special_ids) == [
        _image_tokens(2, 2, image_encoder.special_ids)
    ]
    x_token = tokenizer.instruct_tokenizer.tokenizer.encode("x", bos=False, eos=False)[0]
    assert tokenized.tokens.index(image_encoder.special_ids.img) < tokenized.tokens.index(x_token)

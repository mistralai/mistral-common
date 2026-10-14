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
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

img = Image.new(mode="RGB", size=(4, 4), color="red")
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


@pytest.fixture
def mm_tokenizer() -> MistralTokenizer:
    path = str(MistralTokenizer._data_path() / "tekken_240911.json")
    tokenizer = MistralTokenizer.from_file(path)
    return tokenizer


@pytest.mark.parametrize(argnames="r", argvalues=img_requests + text_requests)
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

r"""Normalizer tests for the bundled v3 multimodal Tekken façade.

Each case runs a `ChatCompletionRequest` through the v3 request normalizer and
checks the project-owned aggregation rules: system messages are folded into the
request-level system prompt, text-only content collapses to a single string and
multimodal content keeps one chunk per run of consecutive `TextChunk`s.
"""

from collections.abc import Callable

import pytest

from mistral_common.protocol.instruct.chunk import ContentChunk, TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.unit.tokens.tokenizers.instruct.request_builders import build_red_image_chunk


def _build_text_then_image_request() -> ChatCompletionRequest:
    r"""Build a request whose user message holds text followed by an image."""
    return ChatCompletionRequest(
        messages=[UserMessage(content=[TextChunk(text="a"), build_red_image_chunk()])],
    )


def _build_system_text_then_image_request() -> ChatCompletionRequest:
    r"""Build a request with a system message and a text-then-image user message."""
    return ChatCompletionRequest(
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[TextChunk(text="C"), build_red_image_chunk()]),
        ]
    )


def _build_system_image_then_text_request() -> ChatCompletionRequest:
    r"""Build a request with a system message and an image-then-text user message."""
    return ChatCompletionRequest(
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[build_red_image_chunk(), TextChunk(text="C")]),
        ]
    )


def _build_system_multiturn_mixed_request() -> ChatCompletionRequest:
    r"""Build a multi-turn request mixing images and text across user messages."""
    return ChatCompletionRequest(
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[build_red_image_chunk(), build_red_image_chunk(), TextChunk(text="C")]),
            AssistantMessage(content="D"),
            UserMessage(content=[build_red_image_chunk(), TextChunk(text="E"), build_red_image_chunk()]),
        ]
    )


def _build_interleaved_text_image_request() -> ChatCompletionRequest:
    r"""Build a request whose user message interleaves text chunks and images."""
    return ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="A"),
                    build_red_image_chunk(),
                    TextChunk(text="B"),
                    TextChunk(text="C"),
                    build_red_image_chunk(),
                    TextChunk(text="D"),
                    TextChunk(text="E"),
                ]
            ),
        ]
    )


def _build_plain_multiturn_request() -> ChatCompletionRequest:
    r"""Build a plain text multi-turn request."""
    return ChatCompletionRequest(
        messages=[
            UserMessage(content="hello"),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ],
    )


def _build_single_user_request() -> ChatCompletionRequest:
    r"""Build a single plain-text user request."""
    return ChatCompletionRequest(messages=[UserMessage(content="hello")])


def _build_empty_text_chunk_request() -> ChatCompletionRequest:
    r"""Build a request whose user message holds a single empty text chunk."""
    return ChatCompletionRequest(messages=[UserMessage(content=[TextChunk(text="")])])


def _build_system_two_text_chunks_request() -> ChatCompletionRequest:
    r"""Build a request with a system message and two consecutive text chunks in the user message."""
    return ChatCompletionRequest(
        messages=[
            SystemMessage(content="You are an AI assistant"),
            UserMessage(content=[TextChunk(text="aaa"), TextChunk(text="bbb")]),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ]
    )


def _count_expected_chunks(content: list[ContentChunk]) -> int:
    r"""Count the chunks a multimodal message normalizes to, joining consecutive `TextChunk`s."""
    count = 0
    previous_was_text = False
    for element in content:
        if isinstance(element, TextChunk):
            if not previous_was_text:
                count += 1
                previous_was_text = True
        else:
            count += 1
            previous_was_text = False
    return count


@pytest.fixture(scope="module")
def mm_tokenizer() -> MistralTokenizer:
    r"""Provide the bundled v3 multimodal Tekken façade."""
    return MistralTokenizer.v3(is_tekken=True, is_mm=True)


@pytest.mark.parametrize(
    "request_builder",
    [
        pytest.param(_build_text_then_image_request, id="text-then-image"),
        pytest.param(_build_system_text_then_image_request, id="system-text-then-image"),
        pytest.param(_build_system_image_then_text_request, id="system-image-then-text"),
        pytest.param(_build_system_multiturn_mixed_request, id="system-multiturn-mixed"),
        pytest.param(_build_interleaved_text_image_request, id="interleaved-text-image"),
        pytest.param(_build_plain_multiturn_request, id="plain-multiturn"),
        pytest.param(_build_single_user_request, id="single-user"),
        pytest.param(_build_empty_text_chunk_request, id="empty-text-chunk"),
        pytest.param(_build_system_two_text_chunks_request, id="system-two-text-chunks"),
    ],
)
def test_v3_mm_normalizer_aggregates_content(
    mm_tokenizer: MistralTokenizer,
    request_builder: Callable[[], ChatCompletionRequest],
) -> None:
    r"""Fold system messages away, collapse text-only content to a string and keep multimodal chunk counts."""
    request = request_builder()
    normalized = mm_tokenizer._instruct_request_normalizer.from_chat_completion_request(request)

    expected_messages = [m for m in request.messages if not isinstance(m, SystemMessage)]
    assert len(expected_messages) == len(normalized.messages)

    for message, norm_message in zip(expected_messages, normalized.messages):
        content = message.content
        if isinstance(content, str):
            assert norm_message.content == content
        elif all(isinstance(chunk, TextChunk) for chunk in content):
            # text-only chunk lists collapse into a single string
            assert isinstance(norm_message.content, str)
        else:
            assert _count_expected_chunks(content) == len(norm_message.content)

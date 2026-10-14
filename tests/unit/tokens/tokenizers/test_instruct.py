import pytest
from PIL import Image

from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, UserMessage
from mistral_common.tokens.tokenizers.base import (
    InstructRequest,
    InstructTokenizer,
    SpecialTokens,
    Tokenized,
)
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV7
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.fixtures.audio import get_dummy_audio_chunk
from tests.test_tokenizer_v7_audio import get_tekkenizer_with_audio


@pytest.fixture(scope="module")
def image_tokenizer() -> InstructTokenizerV7:
    r"""A V7 tokenizer whose images encode to few tokens, to keep budgets small.

    Returns:
        The instruct tokenizer with a 2 pixel image patch size.
    """
    tokenizer = MistralTokenizer.v7(is_mm=True).instruct_tokenizer
    assert isinstance(tokenizer, InstructTokenizerV7)
    assert tokenizer.image_encoder is not None
    tokenizer.image_encoder.image_config.image_patch_size = 2
    return tokenizer


@pytest.fixture(scope="module")
def audio_tokenizer() -> InstructTokenizerV7:
    r"""A V7 tokenizer with audio support, built on a test vocabulary.

    Returns:
        The instruct tokenizer with an audio encoder.
    """
    return get_tekkenizer_with_audio()


def _special_token_count(tokenizer: InstructTokenizer, tokenized: Tokenized, special_token: SpecialTokens) -> int:
    r"""Count how often a special token appears in a tokenized request.

    Args:
        tokenizer: The tokenizer that produced the tokens.
        tokenized: The tokenized request to inspect.
        special_token: The special token to count.

    Returns:
        The number of occurrences.
    """
    return tokenized.tokens.count(tokenizer.tokenizer.get_special_token(special_token.value))


def _image_message(size: tuple[int, int] = (8, 8)) -> UserMessage:
    return UserMessage(content=[TextChunk(text="a"), ImageChunk(image=Image.new("RGB", size, "red"))])


def test_truncation_drops_the_image_of_a_dropped_message(image_tokenizer: InstructTokenizerV7) -> None:
    messages: list[ChatMessage] = [_image_message(), AssistantMessage(content="b"), UserMessage(content="c" * 30)]
    full = image_tokenizer.encode_instruct(InstructRequest(messages=messages))
    assert len(full.images) == 1
    image_tokens = _special_token_count(image_tokenizer, full, SpecialTokens.img)

    # One token below everything the image takes: the image message cannot fit.
    tokenized = image_tokenizer.encode_instruct(
        InstructRequest(messages=messages, truncate_at_max_tokens=len(full.tokens) - image_tokens - 1)
    )

    assert len(tokenized.tokens) < len(full.tokens)
    assert len(tokenized.images) == 0
    assert _special_token_count(image_tokenizer, tokenized, SpecialTokens.img) == 0


def test_truncation_keeps_the_image_of_a_kept_message(image_tokenizer: InstructTokenizerV7) -> None:
    messages: list[ChatMessage] = [_image_message(), AssistantMessage(content="b"), UserMessage(content="c" * 30)]
    full = image_tokenizer.encode_instruct(InstructRequest(messages=messages))

    tokenized = image_tokenizer.encode_instruct(
        InstructRequest(messages=messages, truncate_at_max_tokens=len(full.tokens))
    )

    assert tokenized.tokens == full.tokens
    assert len(tokenized.images) == 1


def test_truncation_drops_only_the_images_of_dropped_messages(image_tokenizer: InstructTokenizerV7) -> None:
    messages: list[ChatMessage] = [
        _image_message(),
        AssistantMessage(content="b"),
        _image_message((16, 16)),
        AssistantMessage(content="b"),
        UserMessage(content="c" * 30),
    ]
    full = image_tokenizer.encode_instruct(InstructRequest(messages=messages))
    assert len(full.images) == 2
    image_tokens = _special_token_count(image_tokenizer, full, SpecialTokens.img)

    # Exactly the budget of everything after the first turn, so only the first
    # image message and the turn it belongs to have to go.
    tail = image_tokenizer.encode_instruct(InstructRequest(messages=messages[2:]))
    tokenized = image_tokenizer.encode_instruct(
        InstructRequest(messages=messages, truncate_at_max_tokens=len(tail.tokens))
    )

    assert len(tokenized.tokens) < len(full.tokens)
    assert len(tokenized.images) == 1
    assert 0 < _special_token_count(image_tokenizer, tokenized, SpecialTokens.img) < image_tokens


def test_truncation_drops_the_audio_of_a_dropped_message(audio_tokenizer: InstructTokenizerV7) -> None:
    messages: list[ChatMessage] = [
        UserMessage(content=[TextChunk(text="a"), get_dummy_audio_chunk()]),
        AssistantMessage(content="b"),
        UserMessage(content="c"),
    ]
    full = audio_tokenizer.encode_instruct(InstructRequest(messages=messages))
    assert len(full.audios) == 1

    tokenized = audio_tokenizer.encode_instruct(
        InstructRequest(messages=messages, truncate_at_max_tokens=len(full.tokens) - 2)
    )

    assert len(tokenized.tokens) < len(full.tokens)
    assert len(tokenized.audios) == 0

r"""V3 multimodal public chat cases.

Replaces the public selectors of ``tests/test_tokenizer_v3_mm.py``: the
text/multimodal agreement rows, the ordered image/text swap comparison, the
five hidden image requests with their exact legacy token ids, and the five
image-ordering rows. Each public encode gets its own case and reviewed
manifest; the legacy cross-output equality or difference is retained as a
relational assertion after the exact comparisons.
"""

from dataclasses import dataclass

from PIL import Image

from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from tests.integration.chat_cases import PublicChatSuccessCase
from tests.integration.chat_recipes import ChatRecipe
from tests.integration.tokenizer_configurations import (
    BUNDLED_TEKKEN_V3_MM_PATCH2_TEST,
    BUNDLED_TEKKEN_V3_MM_TEST,
    BUNDLED_TEKKEN_V3_TEXT_TEST,
)

# Expected relation between the complete token sequences of a paired case.
EQUAL_TOKENS = "equal"
DIFFERENT_TOKENS = "different"


@dataclass(frozen=True)
class PairedChatCase:
    """Two public success cases whose complete outputs share a relation."""

    pair_id: str
    first: PublicChatSuccessCase
    second: PublicChatSuccessCase
    token_relation: str


def _red_4x4() -> Image.Image:
    return Image.new(mode="RGB", size=(4, 4), color="red")


def _blue_6x4() -> Image.Image:
    return Image.new(mode="RGB", size=(6, 4), color="blue")


def _build_multiturn_text() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(content="hello"),
            UserMessage(content=[TextChunk(text="bbb"), TextChunk(text="ccc")]),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ],
    )


def _build_single_user_text() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](messages=[UserMessage(content="hello")])


def _build_empty_text() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](messages=[UserMessage(content=[TextChunk(text="")])])


def _build_system_adjacent_text() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            SystemMessage(content="You are an AI assistant"),
            UserMessage(content=[TextChunk(text="aaa"), TextChunk(text="bbb")]),
            AssistantMessage(content="aaa"),
            UserMessage(content="goodbye"),
        ]
    )


def _build_swap_image_first() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content=[ImageChunk(image=_red_4x4()), TextChunk(text="What is on this image?")])],
    )


def _build_swap_text_first() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content=[TextChunk(text="What is on this image?"), ImageChunk(image=_red_4x4())])],
    )


def _build_swap_appended_image_first() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(
                content=[
                    ImageChunk(image=_red_4x4()),
                    TextChunk(text="What is on this image?"),
                    TextChunk(text="more"),
                ]
            )
        ],
    )


def _build_swap_appended_text_first() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="What is on this image?"),
                    ImageChunk(image=_red_4x4()),
                    TextChunk(text="more"),
                ]
            )
        ],
    )


def _build_image_user_text_first() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content=[TextChunk(text="a"), ImageChunk(image=_red_4x4())])],
    )


def _build_image_system_text_first() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[TextChunk(text="C"), ImageChunk(image=_red_4x4())]),
        ],
    )


def _build_image_system_image_first() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[ImageChunk(image=_red_4x4()), TextChunk(text="C")]),
        ],
    )


def _build_image_multiturn_four() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            SystemMessage(content="A B"),
            UserMessage(content=[ImageChunk(image=_red_4x4()), ImageChunk(image=_red_4x4()), TextChunk(text="C")]),
            AssistantMessage(content="D"),
            UserMessage(content=[ImageChunk(image=_red_4x4()), TextChunk(text="E"), ImageChunk(image=_red_4x4())]),
        ]
    )


def _build_image_interleaved_two() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="A"),
                    ImageChunk(image=_red_4x4()),
                    TextChunk(text="B"),
                    TextChunk(text="C"),
                    ImageChunk(image=_red_4x4()),
                    TextChunk(text="D"),
                    TextChunk(text="E"),
                ]
            )
        ],
    )


def _build_order_empty_text_two_images() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(content=[TextChunk(text=""), ImageChunk(image=_red_4x4()), ImageChunk(image=_blue_6x4())])
        ],
    )


def _build_order_text_two_images() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(content=[TextChunk(text="x"), ImageChunk(image=_red_4x4()), ImageChunk(image=_blue_6x4())])
        ],
    )


def _build_order_two_images() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content=[ImageChunk(image=_red_4x4()), ImageChunk(image=_blue_6x4())])],
    )


def _build_order_trailing_image() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content=[TextChunk(text="x"), ImageChunk(image=_red_4x4())])],
    )


def _build_order_leading_image() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content=[ImageChunk(image=_red_4x4()), TextChunk(text="x")])],
    )


_MULTITURN_TEXT = ChatRecipe(recipe_id="v3-agree-multiturn-text", build=_build_multiturn_text)
_SINGLE_USER_TEXT = ChatRecipe(recipe_id="v3-agree-single-text", build=_build_single_user_text)
_EMPTY_TEXT = ChatRecipe(recipe_id="v3-agree-empty-text", build=_build_empty_text)
_SYSTEM_ADJACENT_TEXT = ChatRecipe(recipe_id="v3-agree-system-text", build=_build_system_adjacent_text)

_T3 = BUNDLED_TEKKEN_V3_TEXT_TEST
_M3 = BUNDLED_TEKKEN_V3_MM_TEST
_M3P = BUNDLED_TEKKEN_V3_MM_PATCH2_TEST

AGREEMENT_PAIRS: tuple[PairedChatCase, ...] = (
    PairedChatCase(
        pair_id="multiturn",
        first=PublicChatSuccessCase(case_id="chat-mm3-agree-multiturn-text", recipe=_MULTITURN_TEXT, configuration=_T3),
        second=PublicChatSuccessCase(
            case_id="chat-mm3-agree-multiturn-image", recipe=_MULTITURN_TEXT, configuration=_M3
        ),
        token_relation=EQUAL_TOKENS,
    ),
    PairedChatCase(
        pair_id="single-user",
        first=PublicChatSuccessCase(case_id="chat-mm3-agree-single-text", recipe=_SINGLE_USER_TEXT, configuration=_T3),
        second=PublicChatSuccessCase(
            case_id="chat-mm3-agree-single-image", recipe=_SINGLE_USER_TEXT, configuration=_M3
        ),
        token_relation=EQUAL_TOKENS,
    ),
    PairedChatCase(
        pair_id="empty-text",
        first=PublicChatSuccessCase(case_id="chat-mm3-agree-empty-text", recipe=_EMPTY_TEXT, configuration=_T3),
        second=PublicChatSuccessCase(case_id="chat-mm3-agree-empty-image", recipe=_EMPTY_TEXT, configuration=_M3),
        token_relation=EQUAL_TOKENS,
    ),
    PairedChatCase(
        pair_id="system-adjacent",
        first=PublicChatSuccessCase(
            case_id="chat-mm3-agree-system-text", recipe=_SYSTEM_ADJACENT_TEXT, configuration=_T3
        ),
        second=PublicChatSuccessCase(
            case_id="chat-mm3-agree-system-image", recipe=_SYSTEM_ADJACENT_TEXT, configuration=_M3
        ),
        token_relation=EQUAL_TOKENS,
    ),
)

SWAP_PAIRS: tuple[PairedChatCase, ...] = (
    PairedChatCase(
        pair_id="swap-initial",
        first=PublicChatSuccessCase(
            case_id="chat-mm3-swap-initial-image-first",
            recipe=ChatRecipe(recipe_id="v3-swap-image-first", build=_build_swap_image_first),
            configuration=_M3,
        ),
        second=PublicChatSuccessCase(
            case_id="chat-mm3-swap-initial-text-first",
            recipe=ChatRecipe(recipe_id="v3-swap-text-first", build=_build_swap_text_first),
            configuration=_M3,
        ),
        token_relation=EQUAL_TOKENS,
    ),
    PairedChatCase(
        pair_id="swap-appended",
        first=PublicChatSuccessCase(
            case_id="chat-mm3-swap-appended-image-first",
            recipe=ChatRecipe(recipe_id="v3-swap-appended-image-first", build=_build_swap_appended_image_first),
            configuration=_M3,
        ),
        second=PublicChatSuccessCase(
            case_id="chat-mm3-swap-appended-text-first",
            recipe=ChatRecipe(recipe_id="v3-swap-appended-text-first", build=_build_swap_appended_text_first),
            configuration=_M3,
        ),
        token_relation=DIFFERENT_TOKENS,
    ),
)

IMAGE_CASES: tuple[PublicChatSuccessCase, ...] = (
    PublicChatSuccessCase(
        case_id="chat-mm3-image-user-text-first",
        recipe=ChatRecipe(recipe_id="v3-image-user-text-first", build=_build_image_user_text_first),
        configuration=_M3P,
    ),
    PublicChatSuccessCase(
        case_id="chat-mm3-image-system-text-first",
        recipe=ChatRecipe(recipe_id="v3-image-system-text-first", build=_build_image_system_text_first),
        configuration=_M3P,
    ),
    PublicChatSuccessCase(
        case_id="chat-mm3-image-system-image-first",
        recipe=ChatRecipe(recipe_id="v3-image-system-image-first", build=_build_image_system_image_first),
        configuration=_M3P,
    ),
    PublicChatSuccessCase(
        case_id="chat-mm3-image-multiturn-four",
        recipe=ChatRecipe(recipe_id="v3-image-multiturn-four", build=_build_image_multiturn_four),
        configuration=_M3P,
    ),
    PublicChatSuccessCase(
        case_id="chat-mm3-image-interleaved-two",
        recipe=ChatRecipe(recipe_id="v3-image-interleaved-two", build=_build_image_interleaved_two),
        configuration=_M3P,
    ),
)

MULTI_IMAGE_ORDER_CASES: tuple[PublicChatSuccessCase, ...] = (
    PublicChatSuccessCase(
        case_id="chat-mm3-order-empty-text-two",
        recipe=ChatRecipe(recipe_id="v3-order-empty-text-two", build=_build_order_empty_text_two_images),
        configuration=_M3P,
    ),
    PublicChatSuccessCase(
        case_id="chat-mm3-order-text-two",
        recipe=ChatRecipe(recipe_id="v3-order-text-two", build=_build_order_text_two_images),
        configuration=_M3P,
    ),
    PublicChatSuccessCase(
        case_id="chat-mm3-order-two",
        recipe=ChatRecipe(recipe_id="v3-order-two", build=_build_order_two_images),
        configuration=_M3P,
    ),
)

TRAILING_IMAGE_CASE = PublicChatSuccessCase(
    case_id="chat-mm3-order-trailing-moves",
    recipe=ChatRecipe(recipe_id="v3-order-trailing-moves", build=_build_order_trailing_image),
    configuration=_M3P,
)
LEADING_IMAGE_CASE = PublicChatSuccessCase(
    case_id="chat-mm3-order-leading-stays",
    recipe=ChatRecipe(recipe_id="v3-order-leading-stays", build=_build_order_leading_image),
    configuration=_M3P,
)

ORDERING_CASES: tuple[PublicChatSuccessCase, ...] = (*MULTI_IMAGE_ORDER_CASES, TRAILING_IMAGE_CASE, LEADING_IMAGE_CASE)

V3_SUCCESS_CASES: tuple[PublicChatSuccessCase, ...] = tuple(
    [case for pair in AGREEMENT_PAIRS for case in (pair.first, pair.second)]
    + [case for pair in SWAP_PAIRS for case in (pair.first, pair.second)]
    + list(IMAGE_CASES)
    + list(ORDERING_CASES)
)

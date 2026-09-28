r"""Public chat cases that close behavior cells on the six pinned profiles."""

from PIL import Image

from mistral_common.exceptions import InvalidMessageStructureException, InvalidSystemPromptException, TokenizerException
from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from tests.fixtures.audio import get_dummy_audio_chunk, get_dummy_audio_url_chunk
from tests.integration.chat_cases import PublicChatErrorCase, PublicChatSuccessCase
from tests.integration.chat_recipes import WEATHER_FULL, ChatRecipe
from tests.integration.tokenizer_configurations import (
    PINNED_V7_AUDIO_TEST,
    PINNED_V7_IMAGE_FINETUNING,
    PINNED_V7_IMAGE_TEST,
    PINNED_V11_IMAGE_FINETUNING,
    PINNED_V11_IMAGE_TEST,
    PINNED_V13_IMAGE_TEST,
    PINNED_V13_TEXT_FINETUNING,
    PINNED_V13_TEXT_TEST,
    PINNED_V15_IMAGE_SETTINGS_FINETUNING,
    PINNED_V15_IMAGE_SETTINGS_TEST,
)


def _red_image() -> Image.Image:
    return Image.new(mode="RGB", size=(4, 4), color="red")


def _blue_image() -> Image.Image:
    return Image.new(mode="RGB", size=(30, 4), color="blue")


def _build_user_image() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content=[TextChunk(text="Describe this image."), ImageChunk(image=_red_image())])]
    )


def _build_user_audio() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content=[TextChunk(text="Transcribe this audio."), get_dummy_audio_chunk()])]
    )


def _build_user_audio_url() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content=[TextChunk(text="Transcribe this audio."), get_dummy_audio_url_chunk()])]
    )


def _build_two_user_images() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="Compare these images."),
                    ImageChunk(image=_red_image()),
                    ImageChunk(image=_blue_image()),
                ]
            )
        ]
    )


def _build_system_think() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[SystemMessage(content=[ThinkChunk(thinking="Hi")]), UserMessage(content="Hello")]
    )


def _build_prefixed_final() -> ChatCompletionRequest[ChatMessage]:
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)]
    )


_USER_IMAGE = ChatRecipe(recipe_id="released-user-image", build=_build_user_image)
_USER_AUDIO = ChatRecipe(recipe_id="released-user-audio", build=_build_user_audio)
_USER_AUDIO_URL = ChatRecipe(recipe_id="released-user-audio-url", build=_build_user_audio_url)
_TWO_USER_IMAGES = ChatRecipe(recipe_id="released-two-user-images", build=_build_two_user_images)
_SYSTEM_THINK = ChatRecipe(recipe_id="released-system-think", build=_build_system_think)
_PREFIXED_FINAL = ChatRecipe(recipe_id="released-prefixed-final", build=_build_prefixed_final)


RELEASED_SUCCESS_CASES: tuple[PublicChatSuccessCase, ...] = (
    PublicChatSuccessCase(
        case_id="chat-sample-weather-full-released-p7i", recipe=WEATHER_FULL, configuration=PINNED_V7_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-full-released-p7a", recipe=WEATHER_FULL, configuration=PINNED_V7_AUDIO_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-full-released-p11i", recipe=WEATHER_FULL, configuration=PINNED_V11_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-full-released-p13t", recipe=WEATHER_FULL, configuration=PINNED_V13_TEXT_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-full-released-p13i", recipe=WEATHER_FULL, configuration=PINNED_V13_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-sample-weather-full-released-p15i",
        recipe=WEATHER_FULL,
        configuration=PINNED_V15_IMAGE_SETTINGS_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v7-released-user-image", recipe=_USER_IMAGE, configuration=PINNED_V7_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v7-released-user-audio", recipe=_USER_AUDIO, configuration=PINNED_V7_AUDIO_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v7-released-user-audio-url", recipe=_USER_AUDIO_URL, configuration=PINNED_V7_AUDIO_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v11-released-image-order", recipe=_TWO_USER_IMAGES, configuration=PINNED_V11_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v11-released-prefixed-final", recipe=_PREFIXED_FINAL, configuration=PINNED_V11_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v13-released-user-image", recipe=_USER_IMAGE, configuration=PINNED_V13_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v13-released-system-think", recipe=_SYSTEM_THINK, configuration=PINNED_V13_TEXT_TEST
    ),
)

RELEASED_ERROR_CASES: tuple[PublicChatErrorCase, ...] = (
    PublicChatErrorCase(
        case_id="chat-v7-released-system-think-rejected",
        recipe=_SYSTEM_THINK,
        configuration=PINNED_V7_IMAGE_TEST,
        expected_exception=TokenizerException,
        message_pattern=r"Think not implemented for tokenizer < V13\.",
    ),
    PublicChatErrorCase(
        case_id="chat-v11-released-system-think-rejected",
        recipe=_SYSTEM_THINK,
        configuration=PINNED_V11_IMAGE_TEST,
        expected_exception=TokenizerException,
        message_pattern=r"Think not implemented for tokenizer < V13\.",
    ),
    PublicChatErrorCase(
        case_id="chat-v15-released-system-think-rejected",
        recipe=_SYSTEM_THINK,
        configuration=PINNED_V15_IMAGE_SETTINGS_TEST,
        expected_exception=InvalidSystemPromptException,
        message_pattern=r"Unexpected content chunk types in system message: \['ThinkChunk'\]",
    ),
    PublicChatErrorCase(
        case_id="chat-v7-released-finetuning-terminal-rejected",
        recipe=WEATHER_FULL,
        configuration=PINNED_V7_IMAGE_FINETUNING,
        expected_exception=InvalidMessageStructureException,
        message_pattern=r"Expected last role Assistant for finetuning but got tool",
    ),
    PublicChatErrorCase(
        case_id="chat-v11-released-finetuning-terminal-rejected",
        recipe=WEATHER_FULL,
        configuration=PINNED_V11_IMAGE_FINETUNING,
        expected_exception=InvalidMessageStructureException,
        message_pattern=r"Expected last role Assistant for finetuning but got tool",
    ),
    PublicChatErrorCase(
        case_id="chat-v13-released-finetuning-terminal-rejected",
        recipe=WEATHER_FULL,
        configuration=PINNED_V13_TEXT_FINETUNING,
        expected_exception=InvalidMessageStructureException,
        message_pattern=r"Expected last role Assistant for finetuning but got tool",
    ),
    PublicChatErrorCase(
        case_id="chat-v15-released-finetuning-terminal-rejected",
        recipe=WEATHER_FULL,
        configuration=PINNED_V15_IMAGE_SETTINGS_FINETUNING,
        expected_exception=InvalidMessageStructureException,
        message_pattern=r"Expected last role Assistant for finetuning but got tool",
    ),
)

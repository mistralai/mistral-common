r"""Public chat cases that close behavior cells on the seven pinned profiles."""

from PIL import Image

from mistral_common.exceptions import (
    InvalidMessageStructureException,
    InvalidRequestException,
    InvalidSystemPromptException,
    TokenizerException,
)
from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessage, UserMessage
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    JsonSchema,
    ReasoningEffort,
    ResponseFormat,
    ResponseFormats,
)
from tests.fixtures.audio import get_dummy_audio_chunk, get_dummy_audio_url_chunk
from tests.integration.chat_cases import PublicChatErrorCase, PublicChatSuccessCase
from tests.integration.chat_recipes import (
    PREFIXED_FINAL,
    SYSTEM_THEN_AUDIO_USER,
    SYSTEM_THINK,
    WEATHER_FULL,
    ChatRecipe,
    build_red_image,
    build_user_content_request,
    build_user_media_request,
)
from tests.integration.tokenizer_configurations import (
    PINNED_V7_AUDIO_TEST,
    PINNED_V7_IMAGE_FINETUNING,
    PINNED_V7_IMAGE_TEST,
    PINNED_V11_IMAGE_FINETUNING,
    PINNED_V11_IMAGE_TEST,
    PINNED_V13_IMAGE_TEST,
    PINNED_V13_TEXT_FINETUNING,
    PINNED_V13_TEXT_TEST,
    PINNED_V15_IMAGE_JSON_SCHEMA_TEST,
    PINNED_V15_IMAGE_SETTINGS_FINETUNING,
    PINNED_V15_IMAGE_SETTINGS_TEST,
)


def _blue_image() -> Image.Image:
    return Image.new(mode="RGB", size=(30, 4), color="blue")


def _build_user_image() -> ChatCompletionRequest[ChatMessage]:
    return build_user_content_request(
        content=[TextChunk(text="Describe this image."), ImageChunk(image=build_red_image())]
    )


def _build_user_audio() -> ChatCompletionRequest[ChatMessage]:
    return build_user_media_request(
        prompt="Transcribe this audio.",
        content_chunk=get_dummy_audio_chunk(),
    )


def _build_user_audio_url() -> ChatCompletionRequest[ChatMessage]:
    return build_user_media_request(
        prompt="Transcribe this audio.",
        content_chunk=get_dummy_audio_url_chunk(),
    )


def _build_two_user_images() -> ChatCompletionRequest[ChatMessage]:
    return build_user_content_request(
        content=[
            TextChunk(text="Compare these images."),
            ImageChunk(image=build_red_image()),
            ImageChunk(image=_blue_image()),
        ]
    )


def _build_settings_request(
    *, reasoning_effort: ReasoningEffort, response_format: ResponseFormat
) -> ChatCompletionRequest[ChatMessage]:
    r"""Build the prefixed-final conversation with explicit model settings inputs.

    Args:
        reasoning_effort: Requested reasoning effort.
        response_format: Requested response format.

    Returns:
        A fresh public chat request.
    """
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)],
        reasoning_effort=reasoning_effort,
        response_format=response_format,
    )


def _build_json_schema_low_effort() -> ChatCompletionRequest[ChatMessage]:
    return _build_settings_request(
        reasoning_effort=ReasoningEffort.low,
        response_format=ResponseFormat(
            type=ResponseFormats.json_schema,
            json_schema=JsonSchema(
                name="weather",
                schema={"type": "object", "properties": {"temperature": {"type": "number"}}},
            ),
        ),
    )


def _build_json_object() -> ChatCompletionRequest[ChatMessage]:
    r"""Build a prefixed-final request asking for any valid JSON object."""
    return _build_settings_request(
        reasoning_effort=ReasoningEffort.high,
        response_format=ResponseFormat(type=ResponseFormats.json_object),
    )


def _build_invalid_json_schema() -> ChatCompletionRequest[ChatMessage]:
    r"""Build a request whose JSON schema is invalid under Draft 7.

    `JsonSchema.model_construct` bypasses the request-model validation so the
    invalid schema reaches the tokenizer's model-settings validation, which is
    the project-owned rejection under test.
    """
    invalid_schema = JsonSchema.model_construct(name="bad", custom_schema={"type": 123}, strict=True)
    return _build_settings_request(
        reasoning_effort=ReasoningEffort.high,
        response_format=ResponseFormat(type=ResponseFormats.json_schema, json_schema=invalid_schema),
    )


def _build_effort_medium() -> ChatCompletionRequest[ChatMessage]:
    return _build_settings_request(reasoning_effort=ReasoningEffort.medium, response_format=ResponseFormat())


def _build_effort_none() -> ChatCompletionRequest[ChatMessage]:
    return _build_settings_request(reasoning_effort=ReasoningEffort.none, response_format=ResponseFormat())


_USER_IMAGE = ChatRecipe(recipe_id="released-user-image", build=_build_user_image)
_USER_AUDIO = ChatRecipe(recipe_id="released-user-audio", build=_build_user_audio)
_USER_AUDIO_URL = ChatRecipe(recipe_id="released-user-audio-url", build=_build_user_audio_url)
_TWO_USER_IMAGES = ChatRecipe(recipe_id="released-two-user-images", build=_build_two_user_images)
_JSON_SCHEMA_LOW_EFFORT = ChatRecipe(recipe_id="released-json-schema-low-effort", build=_build_json_schema_low_effort)
_JSON_OBJECT = ChatRecipe(recipe_id="released-json-object", build=_build_json_object)
_INVALID_JSON_SCHEMA = ChatRecipe(recipe_id="released-invalid-json-schema", build=_build_invalid_json_schema)
_EFFORT_MEDIUM = ChatRecipe(recipe_id="released-effort-medium", build=_build_effort_medium)
_EFFORT_NONE = ChatRecipe(recipe_id="released-effort-none", build=_build_effort_none)

RELEASED_USER_AUDIO_URL_CASE = PublicChatSuccessCase(
    case_id="chat-v7-released-user-audio-url",
    recipe=_USER_AUDIO_URL,
    configuration=PINNED_V7_AUDIO_TEST,
)


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
        case_id="chat-sample-weather-full-released-p15j",
        recipe=WEATHER_FULL,
        configuration=PINNED_V15_IMAGE_JSON_SCHEMA_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v7-released-user-image", recipe=_USER_IMAGE, configuration=PINNED_V7_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v7-released-user-audio", recipe=_USER_AUDIO, configuration=PINNED_V7_AUDIO_TEST
    ),
    RELEASED_USER_AUDIO_URL_CASE,
    PublicChatSuccessCase(
        case_id="chat-v11-released-image-order", recipe=_TWO_USER_IMAGES, configuration=PINNED_V11_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v11-released-prefixed-final", recipe=PREFIXED_FINAL, configuration=PINNED_V11_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v13-released-user-image", recipe=_USER_IMAGE, configuration=PINNED_V13_IMAGE_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v13-released-system-think", recipe=SYSTEM_THINK, configuration=PINNED_V13_TEXT_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-released-user-image-p15j",
        recipe=_USER_IMAGE,
        configuration=PINNED_V15_IMAGE_JSON_SCHEMA_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-released-default-high-p15j",
        recipe=PREFIXED_FINAL,
        configuration=PINNED_V15_IMAGE_JSON_SCHEMA_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-released-effort-medium-p15j",
        recipe=_EFFORT_MEDIUM,
        configuration=PINNED_V15_IMAGE_JSON_SCHEMA_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-released-json-schema-low-p15j",
        recipe=_JSON_SCHEMA_LOW_EFFORT,
        configuration=PINNED_V15_IMAGE_JSON_SCHEMA_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-released-json-object-p15j",
        recipe=_JSON_OBJECT,
        configuration=PINNED_V15_IMAGE_JSON_SCHEMA_TEST,
    ),
)

RELEASED_ERROR_CASES: tuple[PublicChatErrorCase, ...] = (
    PublicChatErrorCase(
        case_id="chat-v7-released-system-think-rejected",
        recipe=SYSTEM_THINK,
        configuration=PINNED_V7_IMAGE_TEST,
        expected_exception=TokenizerException,
        message_pattern=r"Think not implemented for tokenizer < V13\.",
    ),
    PublicChatErrorCase(
        case_id="chat-v11-released-system-think-rejected",
        recipe=SYSTEM_THINK,
        configuration=PINNED_V11_IMAGE_TEST,
        expected_exception=TokenizerException,
        message_pattern=r"Think not implemented for tokenizer < V13\.",
    ),
    PublicChatErrorCase(
        case_id="chat-v15-released-system-think-rejected",
        recipe=SYSTEM_THINK,
        configuration=PINNED_V15_IMAGE_SETTINGS_TEST,
        expected_exception=InvalidSystemPromptException,
        message_pattern=r"Unexpected content chunk types in system message: \['ThinkChunk'\]",
    ),
    PublicChatErrorCase(
        case_id="chat-v15-released-effort-none-rejected-p15j",
        recipe=_EFFORT_NONE,
        configuration=PINNED_V15_IMAGE_JSON_SCHEMA_TEST,
        expected_exception=InvalidRequestException,
        message_pattern=r"reasoning_effort should be one of \['low', 'medium', 'high'\], got none\.",
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
    PublicChatErrorCase(
        case_id="chat-v7-released-system-then-audio-user-rejected",
        recipe=SYSTEM_THEN_AUDIO_USER,
        configuration=PINNED_V7_AUDIO_TEST,
        expected_exception=ValueError,
        message_pattern=r"Found system messages at indexes .* and audio chunks in messages at indexes",
    ),
    PublicChatErrorCase(
        case_id="chat-v11-released-system-then-audio-user-rejected",
        recipe=SYSTEM_THEN_AUDIO_USER,
        configuration=PINNED_V11_IMAGE_TEST,
        expected_exception=ValueError,
        message_pattern=r"System messages are not yet allowed when audio is present",
    ),
    PublicChatErrorCase(
        case_id="chat-v15-released-invalid-json-schema-rejected-p15j",
        recipe=_INVALID_JSON_SCHEMA,
        configuration=PINNED_V15_IMAGE_JSON_SCHEMA_TEST,
        expected_exception=InvalidRequestException,
        message_pattern=r"json_schema is not a valid JSON schema",
    ),
)

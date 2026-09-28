r"""Public v15 chat recipes and cases from the legacy v15 selectors."""

import base64
from collections.abc import Callable
from io import BytesIO

from PIL import Image

from mistral_common.exceptions import InvalidRequestException
from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURLChunk, ImageURLChunk, TextChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from tests.fixtures.audio import get_dummy_audio_chunk, get_dummy_audio_url_chunk
from tests.integration.chat_cases import PublicChatErrorCase, PublicChatSuccessCase
from tests.integration.chat_recipes import ChatRecipe
from tests.integration.tokenizer_configurations import (
    PINNED_V15_IMAGE_SETTINGS_TEST,
    SYNTHETIC_V15_AUDIO_TEST,
    SYNTHETIC_V15_NO_DEFAULT_TEST,
    SYNTHETIC_V15_NO_SETTINGS_TEST,
    SYNTHETIC_V15_REASONING_EMPTY_TEST,
    SYNTHETIC_V15_REASONING_NONE_ONLY_TEST,
)


def _available_tools() -> list[Tool]:
    r"""Build fresh tools matching the v15 public tool selectors."""
    return [
        Tool(
            function=Function(
                name="math_interpreter",
                description="Get the value of an arithmetic expression.",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression.",
                        }
                    },
                },
            )
        )
    ]


def _messages() -> list[ChatMessage]:
    r"""Build fresh ordered tool-call messages for v15 settings selectors."""
    return [
        SystemMessage(content="S"),
        UserMessage(content="U1"),
        AssistantMessage(
            content="A1",
            tool_calls=[
                ToolCall(id="123456789", function=FunctionCall(name="F1", arguments="{}")),
                ToolCall(id="999999999", function=FunctionCall(name="F2", arguments="{}")),
            ],
        ),
        ToolMessage(content="R1", tool_call_id="123456789"),
        ToolMessage(content="R2", tool_call_id="999999999"),
        AssistantMessage(content="A2"),
        UserMessage(content="U2"),
    ]


def _build_call_id_request(*, tool_call_id: str) -> ChatCompletionRequest[ChatMessage]:
    r"""Build a request whose tool call and result share an arbitrary id.

    Args:
        tool_call_id: Identifier paired between the call and result.

    Returns:
        A fresh request with one call and its result.
    """
    return ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(content="a"),
            AssistantMessage(tool_calls=[ToolCall(id=tool_call_id, function=FunctionCall(name="f", arguments="{}"))]),
            ToolMessage(content="b", tool_call_id=tool_call_id),
        ]
    )


def _call_id_recipe(*, recipe_id: str, tool_call_id: str) -> ChatRecipe:
    r"""Bind an arbitrary tool-call id to a fresh-request recipe.

    Args:
        recipe_id: Stable identity for the recipe.
        tool_call_id: Identifier paired between the call and result.

    Returns:
        A recipe that builds a fresh request each time.
    """

    def build() -> ChatCompletionRequest[ChatMessage]:
        return _build_call_id_request(tool_call_id=tool_call_id)

    return ChatRecipe(recipe_id=recipe_id, build=build)


def _build_settings_request(
    *, reasoning_effort: ReasoningEffort | None, include_tools: bool
) -> ChatCompletionRequest[ChatMessage]:
    r"""Build fresh settings messages with optional available tools.

    Args:
        reasoning_effort: Requested reasoning effort, or `None` when absent.
        include_tools: Whether the request carries the v15 math tool.

    Returns:
        A fresh public chat request.
    """
    return ChatCompletionRequest[ChatMessage](
        messages=_messages(),
        tools=_available_tools() if include_tools else None,
        reasoning_effort=reasoning_effort,
    )


def _settings_recipe(*, recipe_id: str, reasoning_effort: ReasoningEffort | None, include_tools: bool) -> ChatRecipe:
    r"""Bind v15 settings inputs to a recipe that constructs fresh requests.

    Args:
        recipe_id: Stable identity for the recipe.
        reasoning_effort: Requested reasoning effort, or `None` when absent.
        include_tools: Whether the request carries the v15 math tool.

    Returns:
        A recipe that builds a fresh request each time.
    """

    def build() -> ChatCompletionRequest[ChatMessage]:
        return _build_settings_request(reasoning_effort=reasoning_effort, include_tools=include_tools)

    return ChatRecipe(recipe_id=recipe_id, build=build)


def _build_prefixed_final() -> ChatCompletionRequest[ChatMessage]:
    r"""Build the legacy user/assistant-prefix request with fresh messages."""
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content="a"), AssistantMessage(content="b", prefix=True)]
    )


def _dummy_image_url_chunk() -> ImageURLChunk:
    r"""Build a fresh 4x4 red PNG data URL for an image request."""
    image = Image.new(mode="RGB", size=(4, 4), color="red")
    buffer = BytesIO()
    image.save(fp=buffer, format="PNG")
    image_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    return ImageURLChunk(image_url=image_url)


def _build_tool_multimodal_request(
    *, content_chunk: AudioChunk | AudioURLChunk | ImageURLChunk
) -> ChatCompletionRequest[ChatMessage]:
    r"""Build the v15 tool-result request with the supplied fresh media chunk.

    Args:
        content_chunk: Audio or image content appended to the tool result.

    Returns:
        A fresh public request with one tool call and result.
    """
    return ChatCompletionRequest[ChatMessage](
        messages=[
            UserMessage(content="Use the tool"),
            AssistantMessage(tool_calls=[ToolCall(id="test12345", function=FunctionCall(name="fn", arguments="{}"))]),
            ToolMessage(
                content=[TextChunk(text="result"), content_chunk],
                tool_call_id="test12345",
            ),
        ],
        tools=[Tool(function=Function(name="fn", description="test", parameters={}))],
    )


def _tool_media_recipe(
    *, recipe_id: str, content_factory: Callable[[], AudioChunk | AudioURLChunk | ImageURLChunk]
) -> ChatRecipe:
    r"""Create a tool recipe that obtains a new media chunk per request.

    Args:
        recipe_id: Stable identity for the recipe.
        content_factory: Function returning a fresh media chunk.

    Returns:
        A recipe that builds a fresh multimodal request each time.
    """

    def build() -> ChatCompletionRequest[ChatMessage]:
        return _build_tool_multimodal_request(content_chunk=content_factory())

    return ChatRecipe(recipe_id=recipe_id, build=build)


def _build_system_audio() -> ChatCompletionRequest[ChatMessage]:
    r"""Build the legacy system-audio request with a fresh waveform chunk."""
    return ChatCompletionRequest[ChatMessage](
        messages=[
            SystemMessage(content=[TextChunk(text="System with content"), get_dummy_audio_chunk()]),
            UserMessage(content="Hello"),
        ]
    )


def _build_user_multimodal_request(
    *, content_chunk: AudioChunk | AudioURLChunk | ImageURLChunk
) -> ChatCompletionRequest[ChatMessage]:
    r"""Build the legacy user-media request with the supplied fresh chunk.

    Args:
        content_chunk: Audio or image content appended to the user message.

    Returns:
        A fresh public request containing the media chunk.
    """
    return ChatCompletionRequest[ChatMessage](
        messages=[UserMessage(content=[TextChunk(text="Here is content"), content_chunk])]
    )


def _user_media_recipe(
    *, recipe_id: str, content_factory: Callable[[], AudioChunk | AudioURLChunk | ImageURLChunk]
) -> ChatRecipe:
    r"""Create a user recipe that obtains a new media chunk per request.

    Args:
        recipe_id: Stable identity for the recipe.
        content_factory: Function returning a fresh media chunk.

    Returns:
        A recipe that builds a fresh multimodal request each time.
    """

    def build() -> ChatCompletionRequest[ChatMessage]:
        return _build_user_multimodal_request(content_chunk=content_factory())

    return ChatRecipe(recipe_id=recipe_id, build=build)


_CALL_ID_X = _call_id_recipe(recipe_id="v15-call-id-x", tool_call_id="x")
_CALL_ID_SLASH = _call_id_recipe(recipe_id="v15-call-id-slash", tool_call_id="call/id-1")
_POLICY_ABSENT = _settings_recipe(recipe_id="v15-policy-absent", reasoning_effort=None, include_tools=True)
_POLICY_NONE = _settings_recipe(recipe_id="v15-policy-none", reasoning_effort=ReasoningEffort.none, include_tools=True)
_POLICY_HIGH = _settings_recipe(recipe_id="v15-policy-high", reasoning_effort=ReasoningEffort.high, include_tools=True)
_IGNORE_ABSENT = _settings_recipe(recipe_id="v15-ignore-absent", reasoning_effort=None, include_tools=False)
_IGNORE_NONE = _settings_recipe(recipe_id="v15-ignore-none", reasoning_effort=ReasoningEffort.none, include_tools=False)
_DEFAULT_ABSENT = _settings_recipe(recipe_id="v15-default-absent", reasoning_effort=None, include_tools=False)
_DEFAULT_NONE = _settings_recipe(
    recipe_id="v15-default-none", reasoning_effort=ReasoningEffort.none, include_tools=False
)
_DEFAULT_HIGH = _settings_recipe(
    recipe_id="v15-default-high", reasoning_effort=ReasoningEffort.high, include_tools=False
)
_NO_DEFAULT_ABSENT = _settings_recipe(recipe_id="v15-no-default-absent", reasoning_effort=None, include_tools=False)
_NO_DEFAULT_NONE = _settings_recipe(
    recipe_id="v15-no-default-none", reasoning_effort=ReasoningEffort.none, include_tools=False
)
_NO_DEFAULT_HIGH = _settings_recipe(
    recipe_id="v15-no-default-high", reasoning_effort=ReasoningEffort.high, include_tools=False
)
_TOOL_AUDIO = _tool_media_recipe(recipe_id="v15-tool-audio", content_factory=get_dummy_audio_chunk)
_TOOL_AUDIO_URL = _tool_media_recipe(recipe_id="v15-tool-audio-url", content_factory=get_dummy_audio_url_chunk)
_TOOL_IMAGE_URL = _tool_media_recipe(recipe_id="v15-tool-image-url", content_factory=_dummy_image_url_chunk)
_SYSTEM_AUDIO = ChatRecipe(recipe_id="v15-system-audio", build=_build_system_audio)
_USER_AUDIO = _user_media_recipe(recipe_id="v15-user-audio", content_factory=get_dummy_audio_chunk)
_USER_AUDIO_URL = _user_media_recipe(recipe_id="v15-user-audio-url", content_factory=get_dummy_audio_url_chunk)
_USER_IMAGE_URL = _user_media_recipe(recipe_id="v15-user-image-url", content_factory=_dummy_image_url_chunk)
_PREFIXED_FINAL = ChatRecipe(recipe_id="v15-prefixed-final", build=_build_prefixed_final)

V15_SUCCESS_CASES: tuple[PublicChatSuccessCase, ...] = (
    PublicChatSuccessCase(
        case_id="chat-v15-call-id-x", recipe=_CALL_ID_X, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-call-id-slash", recipe=_CALL_ID_SLASH, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-policy-absent-both", recipe=_POLICY_ABSENT, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-policy-none-both", recipe=_POLICY_NONE, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-policy-high-both", recipe=_POLICY_HIGH, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-policy-none-only", recipe=_POLICY_NONE, configuration=SYNTHETIC_V15_REASONING_NONE_ONLY_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-ignore-absent-empty",
        recipe=_IGNORE_ABSENT,
        configuration=SYNTHETIC_V15_REASONING_EMPTY_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-ignore-none-no-builder",
        recipe=_IGNORE_NONE,
        configuration=SYNTHETIC_V15_NO_SETTINGS_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-default-absent", recipe=_DEFAULT_ABSENT, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-default-none", recipe=_DEFAULT_NONE, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-default-high", recipe=_DEFAULT_HIGH, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-no-default-absent",
        recipe=_NO_DEFAULT_ABSENT,
        configuration=SYNTHETIC_V15_NO_DEFAULT_TEST,
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-no-default-none", recipe=_NO_DEFAULT_NONE, configuration=SYNTHETIC_V15_NO_DEFAULT_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-no-default-high", recipe=_NO_DEFAULT_HIGH, configuration=SYNTHETIC_V15_NO_DEFAULT_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-prefixed-final", recipe=_PREFIXED_FINAL, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
    PublicChatSuccessCase(case_id="chat-v15-tool-audio", recipe=_TOOL_AUDIO, configuration=SYNTHETIC_V15_AUDIO_TEST),
    PublicChatSuccessCase(
        case_id="chat-v15-tool-audio-url", recipe=_TOOL_AUDIO_URL, configuration=SYNTHETIC_V15_AUDIO_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-tool-image-url", recipe=_TOOL_IMAGE_URL, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-system-audio", recipe=_SYSTEM_AUDIO, configuration=SYNTHETIC_V15_AUDIO_TEST
    ),
    PublicChatSuccessCase(case_id="chat-v15-user-audio", recipe=_USER_AUDIO, configuration=SYNTHETIC_V15_AUDIO_TEST),
    PublicChatSuccessCase(
        case_id="chat-v15-user-audio-url", recipe=_USER_AUDIO_URL, configuration=SYNTHETIC_V15_AUDIO_TEST
    ),
    PublicChatSuccessCase(
        case_id="chat-v15-user-image-url", recipe=_USER_IMAGE_URL, configuration=PINNED_V15_IMAGE_SETTINGS_TEST
    ),
)

V15_ERROR_CASES: tuple[PublicChatErrorCase, ...] = (
    PublicChatErrorCase(
        case_id="chat-v15-policy-high-forbidden",
        recipe=_POLICY_HIGH,
        configuration=SYNTHETIC_V15_REASONING_NONE_ONLY_TEST,
        expected_exception=InvalidRequestException,
        message_pattern=r"should be one of",
    ),
    PublicChatErrorCase(
        case_id="chat-v15-policy-none-unsupported",
        recipe=_POLICY_NONE,
        configuration=SYNTHETIC_V15_REASONING_EMPTY_TEST,
        expected_exception=InvalidRequestException,
        message_pattern=r"not supported for this model",
    ),
)

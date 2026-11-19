r"""Unit tests for `mistral_common.protocol.instruct.validator`.

Migrated from the legacy `tests/validation/` modules and parametrized across
every supported tokenizer version: each behavior is proven through
`get_validator(version, mode)` on all of v1, v2, v3, v7, v11, v13 and v15, with
the per-version expectation derived from the documented validator dispatch
(base for v1/v2, V3 for v3, V5 for v7, V11 for v11, V13 for v13, V15 for v15).

Every candidate message is constructed before the validator is called, so a
rejection always comes from the validator rule under test and never from
message/request construction.
"""

import re
from typing import Any

import pytest

from mistral_common.exceptions import (
    InvalidAssistantMessageException,
    InvalidFunctionCallException,
    InvalidMessageStructureException,
    InvalidRequestException,
    InvalidSystemPromptException,
    InvalidToolException,
    InvalidToolMessageException,
    InvalidToolSchemaException,
    InvalidUserMessageException,
)
from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURLChunk,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    FinetuningAssistantMessage,
    Roles,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    ReasoningEffort,
)
from mistral_common.protocol.instruct.tool_calls import Function, FunctionCall, Tool, ToolCall
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidator,
    MistralRequestValidatorV3,
    MistralRequestValidatorV5,
    MistralRequestValidatorV11,
    MistralRequestValidatorV13,
    MistralRequestValidatorV15,
    ValidationMode,
    get_validator,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.fixtures.audio import get_dummy_audio_chunk, get_dummy_audio_url_chunk
from tests.fixtures.chunks import get_content_chunks
from tests.integration.chat_recipes import build_tool_call_messages, build_user_content_request

_Messages = list[UserMessage | AssistantMessage | SystemMessage | ToolMessage]

_ALL_VERSIONS = list(TokenizerVersion)
_PRE_V15_VERSIONS = [version for version in _ALL_VERSIONS if version < TokenizerVersion.v15]


def _user_convo(content: "str | list[Any]") -> _Messages:
    return [UserMessage(content=content)]


def _assistant_convo(content: "str | list[Any]") -> _Messages:
    return [UserMessage(content="hi"), AssistantMessage(content=content), UserMessage(content="next")]


def _system_convo(content: "str | list[Any]") -> _Messages:
    return [SystemMessage(content=content), UserMessage(content="hi")]


def _tool_convo(content: "str | list[Any]") -> _Messages:
    return [
        UserMessage(content="hi"),
        AssistantMessage(tool_calls=[ToolCall(function=FunctionCall(name="fn", arguments="{}"), id="test12345")]),
        ToolMessage(content=content, tool_call_id="test12345"),
    ]


# Per-version chunk allow-lists, mirroring the validator dispatch: base allows
# text only; v3 adds image chunks to user messages; v7 (V5) adds audio to user
# and audio+think to system; v11 adds think to assistant; v15 drops the tool
# content check and removes think from system.
def _user_chunk_allowed(version: TokenizerVersion, chunk_name: str) -> bool:
    if chunk_name == "text":
        return True
    if chunk_name in ("image", "image_url"):
        return version >= TokenizerVersion.v3
    return version >= TokenizerVersion.v7


def _system_chunk_allowed(version: TokenizerVersion, chunk_name: str) -> bool:
    if chunk_name == "text":
        return True
    if chunk_name == "audio":
        return version >= TokenizerVersion.v7
    return TokenizerVersion.v7 <= version < TokenizerVersion.v15


def _assistant_chunk_allowed(version: TokenizerVersion, chunk_name: str) -> bool:
    if chunk_name == "text":
        return True
    return version >= TokenizerVersion.v11


def _tool_chunk_allowed(version: TokenizerVersion, chunk_name: str) -> bool:
    if chunk_name == "text":
        return True
    return version == TokenizerVersion.v15


_ROLE_CHUNKS: dict[str, tuple[str, ...]] = {
    "user": ("text", "image", "image_url", "audio", "audio_url"),
    "system": ("text", "audio", "think"),
    "assistant": ("text", "think"),
    "tool": ("text", "image", "image_url", "audio", "audio_url"),
}


def _chunk_name(chunk: Any) -> str:
    return {
        "TextChunk": "text",
        "ImageChunk": "image",
        "ImageURLChunk": "image_url",
        "AudioChunk": "audio",
        "AudioURLChunk": "audio_url",
        "ThinkChunk": "think",
    }[type(chunk).__name__]


@pytest.fixture(params=_ALL_VERSIONS, ids=lambda version: version.value)
def version(request: pytest.FixtureRequest) -> TokenizerVersion:
    param = request.param
    assert isinstance(param, TokenizerVersion)
    return param


@pytest.fixture
def validator_serving(version: TokenizerVersion) -> MistralRequestValidator:
    return get_validator(version, ValidationMode.serving)


@pytest.fixture
def validator_finetuning(version: TokenizerVersion) -> MistralRequestValidator:
    return get_validator(version, ValidationMode.finetuning)


@pytest.fixture
def validator_agnostic(version: TokenizerVersion) -> MistralRequestValidator:
    return get_validator(version, ValidationMode.agnostic)


@pytest.fixture
def validator_v15() -> MistralRequestValidatorV15:
    return MistralRequestValidatorV15(ValidationMode.serving)


@pytest.fixture(
    params=[get_dummy_audio_chunk, get_dummy_audio_url_chunk],
    ids=["audio_chunk", "audio_url_chunk"],
)
def audio_content_chunk(request: pytest.FixtureRequest) -> AudioChunk | AudioURLChunk:
    chunk = request.param()
    assert isinstance(chunk, (AudioChunk, AudioURLChunk))
    return chunk


class TestChatStructureAcrossVersions:
    r"""Structure, order and terminal-role rules proven on every version in serving mode."""

    def test_multiple_system_messages_OK(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving.validate_messages(
            messages=[
                SystemMessage(content="foo"),
                SystemMessage(content="foo"),
                SystemMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ],
        )

    def test_system_user_messages_OK(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving.validate_messages(
            messages=[
                SystemMessage(content="foo"),
                UserMessage(content="foo"),
            ],
        )

    def test_user_system_messages_OK(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving.validate_messages(
            messages=[
                UserMessage(content="foo"),
                SystemMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ],
        )

    def test_user_user_messages_OK(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving.validate_messages(
            messages=[
                UserMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ],
        )

    def test_empty_messages(self, validator_serving: MistralRequestValidator) -> None:
        with pytest.raises(InvalidMessageStructureException, match=r"Conversation must have at least one message"):
            validator_serving.validate_messages(
                messages=[],
            )

    def test_starts_with_system_or_user(self, validator_serving: MistralRequestValidator) -> None:
        """
        This is allowed in validation, we will add a blank message in normalization
        """
        validator_serving.validate_messages(
            messages=[
                AssistantMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ],
        )

    def test_ends_with_assistant(self, validator_serving: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=(r"Expected last role User or Tool \(or Assistant with prefix\) for serving but got assistant"),
        ):
            validator_serving.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo"),
                ],
            )

    def test_assistant_prefix(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(content="foo", prefix=True),
            ],
        )
        with pytest.raises(
            InvalidAssistantMessageException,
            match=r"Assistant message with prefix True must be last message",
        ):
            validator_serving.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo", prefix=True),
                    UserMessage(content="foo"),
                ],
            )

    def test_unprefixed_assistant_remains_invalid(self, validator_serving: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=(r"Expected last role User or Tool \(or Assistant with prefix\) for serving but got assistant"),
        ):
            validator_serving.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo"),
                ],
            )

    def test_user_tool_user(self, validator_serving: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=(
                r"^Unexpected role 'tool' after role 'user': tool messages must come right after the assistant "
                r"message that made the tool calls \(or after another tool message\)\. Send all tool results first, "
                r"then any user message\.$"
            ),
        ):
            validator_serving.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    ToolMessage(content="foo"),
                    UserMessage(content="foo"),
                ],
            )

    def test_serving_requires_model_across_versions(self, version: TokenizerVersion) -> None:
        validator = get_validator(version, ValidationMode.serving)
        with pytest.raises(InvalidRequestException, match=r"Model name parameter is required for serving mode"):
            validator.validate_request(ChatCompletionRequest(messages=[UserMessage(content="foo")]))

    def test_test_mode_allows_missing_model_across_versions(self, version: TokenizerVersion) -> None:
        validator = get_validator(version, ValidationMode.test)
        validator.validate_request(ChatCompletionRequest(messages=[UserMessage(content="foo")]))

    def test_tool_calls_OK(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving.validate_messages(
            messages=build_tool_call_messages(call_ids=("123456789",), result_ids=("123456789",))
        )

    def test_not_enough_tool_messages(self, validator_serving: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException, match=r"Not the same number of function calls and responses"
        ):
            validator_serving.validate_messages(
                messages=build_tool_call_messages(call_ids=("123456789", "999999999"), result_ids=("123456789",)),
            )

    def test_too_many_tool_messages(self, validator_serving: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=r"Not the same number of function calls and responses|Unexpected tool call id",
        ):
            validator_serving.validate_messages(
                messages=build_tool_call_messages(call_ids=("123456789",), result_ids=("123456789", "999999999")),
            )

    def test_tool_then_user_ok(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving.validate_messages(
            messages=[
                *build_tool_call_messages(call_ids=("123456789",), result_ids=("123456789",)),
                UserMessage(content="continue with this context"),
            ],
        )

    def test_tool_then_user_full_loop(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving.validate_messages(
            messages=[
                UserMessage(content="search for info"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="aaaaaaaaa", function=FunctionCall(name="search", arguments="{}")),
                        ToolCall(id="bbbbbbbbb", function=FunctionCall(name="search", arguments="{}")),
                    ]
                ),
                ToolMessage(name="search", content="result1", tool_call_id="aaaaaaaaa"),
                ToolMessage(name="search", content="result2", tool_call_id="bbbbbbbbb"),
                UserMessage(content="now refine"),
                AssistantMessage(
                    tool_calls=[ToolCall(id="ccccccccc", function=FunctionCall(name="search", arguments="{}"))]
                ),
                ToolMessage(name="search", content="result3", tool_call_id="ccccccccc"),
                AssistantMessage(content="here is the final answer"),
                UserMessage(content="thanks"),
            ],
        )

    def test_partial_tool_results_then_user_fails(self, validator_serving: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException, match=r"Not the same number of function calls and responses"
        ):
            validator_serving.validate_messages(
                messages=[
                    *build_tool_call_messages(call_ids=("123456789", "999999999"), result_ids=("123456789",)),
                    UserMessage(content="stop, only need the first result"),
                ],
            )

    def test_allow_tool_results_wrong_order(self, validator_serving: MistralRequestValidator) -> None:
        # v11+ pairing compares the sets of expected and observed ids, and the count-based
        # base validator is order-agnostic too, so reversed results are accepted on every
        # version; normalizer reordering is owned by the normalization tests.
        validator_serving.validate_messages(
            messages=[
                *build_tool_call_messages(call_ids=("123456789", "999999999"), result_ids=("999999999", "123456789")),
                UserMessage(content="next"),
            ],
        )

    def test_tool_call_duplicate_ids_same_assistant_messages(
        self, version: TokenizerVersion, validator_serving: MistralRequestValidator
    ) -> None:
        messages = build_tool_call_messages(call_ids=("123456789", "123456789"), result_ids=("123456789",))
        if version >= TokenizerVersion.v11:
            match = r"Duplicate tool call id 123456789 in assistant message"
        else:
            match = r"Not the same number of function calls and responses"
        with pytest.raises(InvalidMessageStructureException, match=match):
            validator_serving.validate_messages(messages=messages)

    def test_tool_call_duplicate_ids_different_assistant_messages(
        self, validator_serving: MistralRequestValidator
    ) -> None:
        validator_serving.validate_messages(
            messages=[
                *build_tool_call_messages(call_ids=("123456789",), result_ids=("123456789",)),
                *build_tool_call_messages(call_ids=("123456789",), result_ids=("123456789",))[1:],
            ],
        )

    def test_multiple_assistant_messages_some_with_tool_calls(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving.validate_messages(
            messages=[
                *build_tool_call_messages(call_ids=("123456789",), result_ids=("123456789",)),
                AssistantMessage(content="h"),
                UserMessage(content="foo"),
                *build_tool_call_messages(call_ids=("123456789",), result_ids=("123456789",))[1:],
                AssistantMessage(content="h"),
                UserMessage(content="foo"),
            ],
        )

    def test_reasoning_effort_policy_across_versions(
        self, version: TokenizerVersion, validator_serving: MistralRequestValidator
    ) -> None:
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=ReasoningEffort.none)
        if version >= TokenizerVersion.v15:
            # v15 delegates model-settings validation to its override, which accepts it.
            validator_serving._validate_model_settings(request)
        else:
            with pytest.raises(
                InvalidRequestException, match="reasoning_effort='none' is not supported for this model"
            ):
                validator_serving._validate_model_settings(request)

    @pytest.mark.parametrize("reasoning_effort", [*list(ReasoningEffort), None])
    def test_v15_accepts_every_reasoning_effort(
        self, reasoning_effort: ReasoningEffort | None, validator_v15: MistralRequestValidatorV15
    ) -> None:
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=reasoning_effort)
        validator_v15._validate_model_settings(request)


class TestChunkPolicyAcrossVersions:
    r"""Per-role content-chunk allow-lists proven on every version."""

    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    @pytest.mark.parametrize("chunk_name", _ROLE_CHUNKS["user"])
    def test_user_chunk_policy(self, version: TokenizerVersion, chunk_name: str) -> None:
        validator = get_validator(version, ValidationMode.serving)
        request = build_user_content_request(content=get_content_chunks((chunk_name,)))
        if _user_chunk_allowed(version, chunk_name):
            validator.validate_messages(request.messages)
        else:
            with pytest.raises(InvalidUserMessageException, match="Unexpected content chunk types in user message"):
                validator.validate_messages(request.messages)

    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    @pytest.mark.parametrize("chunk_name", _ROLE_CHUNKS["system"])
    def test_system_chunk_policy(self, version: TokenizerVersion, chunk_name: str) -> None:
        validator = get_validator(version, ValidationMode.serving)
        if _system_chunk_allowed(version, chunk_name):
            validator.validate_messages(_system_convo(get_content_chunks((chunk_name,))))
        else:
            with pytest.raises(InvalidSystemPromptException, match="Unexpected content chunk types in system message"):
                validator.validate_messages(_system_convo(get_content_chunks((chunk_name,))))

    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    @pytest.mark.parametrize("chunk_name", _ROLE_CHUNKS["assistant"])
    def test_assistant_chunk_policy(self, version: TokenizerVersion, chunk_name: str) -> None:
        validator = get_validator(version, ValidationMode.serving)
        if _assistant_chunk_allowed(version, chunk_name):
            validator.validate_messages(_assistant_convo(get_content_chunks((chunk_name,))))
        else:
            with pytest.raises(
                InvalidAssistantMessageException, match="Unexpected content chunk types in assistant message"
            ):
                validator.validate_messages(_assistant_convo(get_content_chunks((chunk_name,))))

    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    @pytest.mark.parametrize("chunk_name", _ROLE_CHUNKS["tool"])
    def test_tool_chunk_policy(self, version: TokenizerVersion, chunk_name: str) -> None:
        # "think" is rejected at the Pydantic model level (ToolMessage construction), not by the validator.
        validator = get_validator(version, ValidationMode.serving)
        if _tool_chunk_allowed(version, chunk_name):
            validator.validate_messages(_tool_convo(get_content_chunks((chunk_name,))))
        else:
            with pytest.raises(InvalidToolMessageException, match="Unexpected content chunk types in tool message"):
                validator.validate_messages(_tool_convo(get_content_chunks((chunk_name,))))

    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    def test_reports_sorted_unique_invalid_chunk_types(self, version: TokenizerVersion) -> None:
        validator = get_validator(version, ValidationMode.serving)
        chunks = get_content_chunks(("audio", "image_url"))
        rejected = sorted(
            type(chunk).__name__ for chunk in chunks if not _user_chunk_allowed(version, _chunk_name(chunk))
        )
        if not rejected:
            validator.validate_messages(_user_convo(chunks))
        else:
            with pytest.raises(
                InvalidUserMessageException,
                match=rf"Unexpected content chunk types in user message: {re.escape(str(rejected))}",
            ):
                validator.validate_messages(_user_convo(chunks))

    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    def test_system_prompt_with_audio_across_versions(
        self, version: TokenizerVersion, audio_content_chunk: AudioChunk | AudioURLChunk
    ) -> None:
        validator = get_validator(version, ValidationMode.serving)
        messages = [
            SystemMessage(content="This is a system prompt"),
            UserMessage(content=[audio_content_chunk]),
        ]
        if version == TokenizerVersion.v7:
            # Only the V5 validator (selected by v7) forbids system+audio.
            with pytest.raises(
                ValueError,
                match=(
                    r"Found system messages at indexes \[0\] and audio chunks in messages at indexes \[1\]\. This is "
                    r"not allowed prior to the tokenizer version 13\."
                ),
            ):
                validator.validate_messages(messages=messages)
        elif version >= TokenizerVersion.v11:
            validator.validate_messages(messages=messages)
        else:
            # Pre-v7 user messages reject audio chunks before any system/audio rule.
            with pytest.raises(InvalidUserMessageException, match="Unexpected content chunk types in user message"):
                validator.validate_messages(messages=messages)

    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    def test_audio_without_system_prompt_ok(
        self, version: TokenizerVersion, audio_content_chunk: AudioChunk | AudioURLChunk
    ) -> None:
        validator = get_validator(version, ValidationMode.serving)
        if version >= TokenizerVersion.v7:
            validator.validate_messages(
                messages=[
                    UserMessage(content=[audio_content_chunk]),
                    UserMessage(content="User message after audio"),
                ],
            )
        else:
            with pytest.raises(InvalidUserMessageException, match="Unexpected content chunk types in user message"):
                validator.validate_messages(
                    messages=[
                        UserMessage(content=[audio_content_chunk]),
                        UserMessage(content="User message after audio"),
                    ],
                )

    def test_system_prompt_without_audio_ok(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving.validate_messages(
            messages=[
                SystemMessage(content="This is a system prompt"),
                UserMessage(content="User message after system"),
            ],
        )


class TestToolCallIdPolicyAcrossVersionsAndModes:
    r"""Tool-call-id rules proven on every version in every validation mode."""

    _VARIANTS = ("missing", "empty", "null", "bad_format", "valid")

    @staticmethod
    def _make_call(variant: str) -> ToolCall:
        function = FunctionCall(name="foo", arguments="{}")
        if variant == "missing":
            return ToolCall(function=function)
        if variant == "empty":
            return ToolCall(id="", function=function)
        if variant == "null":
            return ToolCall(id="null", function=function)
        if variant == "bad_format":
            return ToolCall(id="bad_id", function=function)
        return ToolCall(id="123456789", function=function)

    @staticmethod
    def _expected_rejection(version: TokenizerVersion, mode: ValidationMode, variant: str) -> str | None:
        r"""Expected `pytest.raises` match, or `None` when the id is accepted.

        Derived from the validator dispatch: base has no id rule; v3-v11 require
        a 9-char alphanumeric id and allow "null" only on the last assistant
        message in finetuning/agnostic mode; v13+ require a non-empty non-"null"
        id in every mode and dropped the format check.
        """
        if variant == "valid":
            return None
        if version < TokenizerVersion.v3:
            return None
        if version >= TokenizerVersion.v13:
            if variant in ("missing", "empty", "null"):
                return r"Tool call id must be a non-empty string other than 'null'"
            return None
        if variant in ("empty", "bad_format"):
            return r"must be a-z, A-Z, 0-9, with a length of 9"
        # missing / null
        if mode in (ValidationMode.finetuning, ValidationMode.agnostic):
            return None
        if mode == ValidationMode.serving:
            return r"Tool call id has to be defined in serving mode"
        return r"only allowed for the last assistant message in finetuning mode"

    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    @pytest.mark.parametrize("mode", list(ValidationMode))
    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_tool_call_id_policy(self, version: TokenizerVersion, mode: ValidationMode, variant: str) -> None:
        tool_call = self._make_call(variant)
        messages: list[ChatMessage] = [UserMessage(content="foo"), AssistantMessage(tool_calls=[tool_call])]
        if mode == ValidationMode.serving:
            messages.append(ToolMessage(content="result", tool_call_id=tool_call.id))
        elif mode == ValidationMode.test:
            messages.append(UserMessage(content="continue"))

        validator = get_validator(version, mode)
        match = self._expected_rejection(version, mode, variant)
        if match is None:
            validator.validate_request(ChatCompletionRequest(messages=messages, model="test"))
        else:
            with pytest.raises(InvalidFunctionCallException, match=match):
                validator.validate_request(ChatCompletionRequest(messages=messages, model="test"))

    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    @pytest.mark.parametrize("variant", ("missing", "bad_format", "null", "valid"))
    def test_tool_message_id_policy(self, version: TokenizerVersion, variant: str) -> None:
        if variant == "missing":
            message = ToolMessage(name="foo", content="bar")
        elif variant == "bad_format":
            message = ToolMessage(name="foo", content="bar", tool_call_id="bad_id")
        elif variant == "null":
            message = ToolMessage(name="foo", content="bar", tool_call_id="null")
        else:
            message = ToolMessage(name="foo", content="bar", tool_call_id="123456789")

        validator = get_validator(version, ValidationMode.serving)
        if variant == "valid":
            validator._validate_tool_message(message)
        elif version < TokenizerVersion.v3:
            validator._validate_tool_message(message)
        elif version >= TokenizerVersion.v13:
            if variant in ("missing", "null"):
                with pytest.raises(
                    InvalidToolMessageException,
                    match=r"Tool call id must be a non-empty string other than 'null'",
                ):
                    validator._validate_tool_message(message)
            else:
                # v13+ dropped the 9-char format check; any non-empty non-"null" id is accepted.
                validator._validate_tool_message(message)
        elif variant == "missing":
            with pytest.raises(InvalidRequestException, match=r"Tool call id has to be defined"):
                validator._validate_tool_message(message)
        else:
            with pytest.raises(
                InvalidToolMessageException,
                match=r"Tool call id was .* but must be a-z, A-Z, 0-9, with a length of 9",
            ):
                validator._validate_tool_message(message)


class TestServingMessagePolicyAcrossVersions:
    r"""Assistant/tool message-level rules proven on every version in serving mode."""

    def test_content_and_tool_calls(self, version: TokenizerVersion) -> None:
        validator = get_validator(version, ValidationMode.serving)
        message = AssistantMessage(
            role=Roles.assistant,
            content="a",
            tool_calls=[ToolCall(id="AbCd56789", function=FunctionCall(name="a", arguments='{"a": 1}'))],
        )
        if version >= TokenizerVersion.v7:
            # V5+ allows content and tool calls together.
            validator._validate_assistant_message(message)
        else:
            with pytest.raises(
                InvalidAssistantMessageException,
                match=r"Assistant message must have either content or tool_calls, but not both.",
            ):
                validator._validate_assistant_message(message)

    @pytest.mark.parametrize("content,tool_calls", [("", None), (None, None), ("", []), (None, [])])
    def test_no_content_and_no_tool_calls(
        self, version: TokenizerVersion, content: str | None, tool_calls: list | None
    ) -> None:
        validator = get_validator(version, ValidationMode.serving)
        message = AssistantMessage(role=Roles.assistant, content=content, tool_calls=tool_calls)
        if version >= TokenizerVersion.v7:
            # V5+ only enforces the XOR rule when content and tool calls are both set.
            validator._validate_assistant_message(message)
        else:
            with pytest.raises(
                InvalidAssistantMessageException,
                match=r"Assistant message must have either content or tool_calls, but not both.",
            ):
                validator._validate_assistant_message(message)

    def test_no_content_and_tool_calls_OK(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving._validate_assistant_message(
            AssistantMessage(
                role=Roles.assistant,
                content="",
                tool_calls=[
                    ToolCall(
                        id="AbCd56789",
                        function=FunctionCall(name="function_call", arguments='{"a": 1}'),
                    )
                ],
            ),
        )

    def test_content_and_no_tool_calls_OK(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving._validate_assistant_message(
            AssistantMessage(
                role=Roles.assistant,
                content="content",
                tool_calls=[],
            ),
        )

    def test_bad_function_name(self, validator_serving: MistralRequestValidator) -> None:
        with pytest.raises(InvalidFunctionCallException, match=r"must be a-z, A-Z, 0-9"):
            validator_serving._validate_assistant_message(
                AssistantMessage(
                    role=Roles.assistant,
                    tool_calls=[
                        ToolCall(
                            id="AbCd56789",
                            function=FunctionCall(name=')_(_)~@:}>""', arguments='{"a": 1}'),
                        )
                    ],
                ),
            )

    def test_accepts_opaque_function_arguments(self, validator_serving: MistralRequestValidator) -> None:
        # Malformed JSON in function arguments is treated as opaque text, not as a name/JSON
        # error; the valid id lets id-aware validators reach the argument check.
        validator_serving._validate_assistant_message(
            AssistantMessage(
                role=Roles.assistant,
                tool_calls=[
                    ToolCall(id="AbCd56789", function=FunctionCall(name="function_name", arguments='{"a":aewfawef 1}'))
                ],
            )
        )

    def test_user_message_blank_content_OK(self, validator_serving: MistralRequestValidator) -> None:
        validator_serving._validate_user_message(UserMessage(content=""))

    def test_tool_message_bad_name(self, validator_serving: MistralRequestValidator) -> None:
        with pytest.raises(InvalidToolMessageException, match=r"must be a-z, A-Z, 0-9"):
            validator_serving._validate_tool_message(ToolMessage(name="~@}£Q_+", content="{}"))

    def test_accepts_opaque_tool_result(self, validator_serving: MistralRequestValidator) -> None:
        # Malformed JSON in a tool result is treated as opaque text, not as a name/JSON error;
        # the valid id lets id-aware validators reach the content check.
        validator_serving._validate_tool_message(
            ToolMessage(name="function_name", content='{"afewfaef}', tool_call_id="AbCd56789")
        )


class TestFinetuningPolicyAcrossVersions:
    r"""Finetuning-mode rules proven on every version."""

    def test_ends_with_assistant(self, validator_finetuning: MistralRequestValidator) -> None:
        validator_finetuning.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(content="foo"),
            ],
        )

    def test_rejects_prefixed_final_assistant(self, validator_finetuning: MistralRequestValidator) -> None:
        with pytest.raises(InvalidMessageStructureException, match=r"Cannot continue final message in finetuning mode"):
            validator_finetuning.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    FinetuningAssistantMessage(content="foo", prefix=True),
                ],
            )

    @pytest.mark.parametrize("weight", [0, 1])
    def test_valid_weight_accepted(self, weight: float, validator_finetuning: MistralRequestValidator) -> None:
        validator_finetuning.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(content="foo", weight=weight),
            ],
        )

    def test_invalid_weight_rejected(self, validator_finetuning: MistralRequestValidator) -> None:
        with pytest.raises(InvalidAssistantMessageException, match=r"Assistant message weight must be either 0 or 1"):
            validator_finetuning.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    FinetuningAssistantMessage(content="foo", weight=0.5),
                ],
            )

    def test_should_allow_multiple_weights(self, validator_finetuning: MistralRequestValidator) -> None:
        validator_finetuning.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(content="foo", weight=0),
                FinetuningAssistantMessage(content="foo", weight=1),
                FinetuningAssistantMessage(content="foo", weight=1),
            ],
        )

    def test_one_message_with_user_is_not_valid(self, validator_finetuning: MistralRequestValidator) -> None:
        with pytest.raises(InvalidMessageStructureException) as exc:
            validator_finetuning.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                ],
            )
        assert str(exc.value) == "Expected last role Assistant for finetuning but got user"

    def test_parallel_tool_calls_unanswered_final_calls_accepted(
        self, validator_finetuning: MistralRequestValidator
    ) -> None:
        validator_finetuning.validate_messages(
            messages=build_tool_call_messages(call_ids=("123456789", "912345678")),
        )

    def test_parallel_tool_results_wrong_order_accepted(self, validator_finetuning: MistralRequestValidator) -> None:
        validator_finetuning.validate_messages(
            messages=[
                *build_tool_call_messages(call_ids=("123456789", "912345678"), result_ids=("912345678", "123456789")),
                FinetuningAssistantMessage(content="foo"),
            ],
        )

    def test_parallel_tool_extra_result_rejected(
        self, version: TokenizerVersion, validator_finetuning: MistralRequestValidator
    ) -> None:
        messages = [
            *build_tool_call_messages(
                call_ids=("123456789", "912345678"), result_ids=("912345678", "123456789", "891234567")
            ),
            FinetuningAssistantMessage(content="foo"),
        ]
        if version >= TokenizerVersion.v11:
            # The unknown third result id is rejected when it is consumed.
            match = r"Unexpected tool call id"
        else:
            # The count-based validator rejects at the final assistant message.
            match = r"Not the same number of function calls and responses"
        with pytest.raises(InvalidMessageStructureException, match=match):
            validator_finetuning.validate_messages(messages=messages)

    def test_ends_with_tool_call_null(
        self, version: TokenizerVersion, validator_finetuning: MistralRequestValidator
    ) -> None:
        function = FunctionCall(name="foo", arguments='{"a": 1}')
        messages = [
            UserMessage(content="foo"),
            FinetuningAssistantMessage(tool_calls=[ToolCall(id="123456789", function=function)]),
            ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
            # tool_call id left "null" as final message
            FinetuningAssistantMessage(tool_calls=[ToolCall(function=function)]),
        ]
        if version >= TokenizerVersion.v13:
            with pytest.raises(
                InvalidFunctionCallException,
                match=r"Tool call id must be a non-empty string other than 'null'",
            ):
                validator_finetuning.validate_messages(messages=messages)
        else:
            validator_finetuning.validate_messages(messages=messages)

    def test_ends_with_no_tool_call(
        self, version: TokenizerVersion, validator_finetuning: MistralRequestValidator
    ) -> None:
        function = FunctionCall(name="foo", arguments='{"a": 1}')
        messages = [
            UserMessage(content="foo"),
            FinetuningAssistantMessage(tool_calls=[ToolCall(id="123456789", function=function)]),
            ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
            FinetuningAssistantMessage(content="foo"),
        ]
        validator_finetuning.validate_messages(messages=messages)

    def test_middle_with_tool_call_null_raises(
        self, version: TokenizerVersion, validator_finetuning: MistralRequestValidator
    ) -> None:
        function = FunctionCall(name="foo", arguments='{"a": 1}')
        messages = [
            UserMessage(content="foo"),
            # tool_call id left "null" as non-final message
            FinetuningAssistantMessage(tool_calls=[ToolCall(function=function)]),
            ToolMessage(name="foo", content="bar", tool_call_id="null"),
            FinetuningAssistantMessage(tool_calls=[ToolCall(function=function)]),
        ]
        if version < TokenizerVersion.v3:
            # The base validator has no tool-call-id rule.
            validator_finetuning.validate_messages(messages=messages)
        elif version >= TokenizerVersion.v13:
            with pytest.raises(
                InvalidFunctionCallException,
                match=r"Tool call id must be a non-empty string other than 'null'",
            ):
                validator_finetuning.validate_messages(messages=messages)
        else:
            with pytest.raises(
                InvalidFunctionCallException,
                match=r"Tool call id of assistant message that is not last has to be defined in finetuning mode",
            ):
                validator_finetuning.validate_messages(messages=messages)


class TestAgnosticPolicyAcrossVersions:
    r"""Agnostic-mode rules proven on every version."""

    @pytest.mark.parametrize(
        "messages",
        [
            [UserMessage(content="foo"), AssistantMessage(content="answer")],
            [UserMessage(content="foo"), UserMessage(content="follow-up")],
            [UserMessage(content="foo"), SystemMessage(content="context")],
            [
                UserMessage(content="foo"),
                AssistantMessage(
                    tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}"))]
                ),
                ToolMessage(content="result", tool_call_id="123456789"),
            ],
        ],
        ids=["assistant", "user", "system", "tool"],
    )
    def test_accepts_any_terminal_role(self, messages: _Messages, validator_agnostic: MistralRequestValidator) -> None:
        validator_agnostic.validate_messages(messages=messages)

    @pytest.mark.parametrize(
        "messages",
        [
            [AssistantMessage(content="answer")],
            [ToolMessage(content="result")],
        ],
        ids=["assistant", "tool"],
    )
    def test_retains_single_message_start_rule(
        self, messages: _Messages, validator_agnostic: MistralRequestValidator
    ) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=r"Conversation must start with a user message or system message",
        ):
            validator_agnostic.validate_messages(messages=messages)

    def test_allows_interrupted_tool_sequence_at_terminal_user(
        self, validator_agnostic: MistralRequestValidator
    ) -> None:
        validator_agnostic.validate_messages(
            messages=[
                *build_tool_call_messages(call_ids=("123456789",)),
                UserMessage(content="stop here"),
            ],
        )

    def test_allows_terminal_tool_call_without_response(self, validator_agnostic: MistralRequestValidator) -> None:
        validator_agnostic.validate_messages(messages=build_tool_call_messages(call_ids=("123456789",)))

    def test_rejects_unresolved_tool_call_before_later_assistant(
        self, validator_agnostic: MistralRequestValidator
    ) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=r"Not the same number of function calls and responses",
        ):
            validator_agnostic.validate_messages(
                messages=[
                    *build_tool_call_messages(call_ids=("123456789",)),
                    AssistantMessage(content="answer"),
                ],
            )

    def test_extra_tool_response(self, version: TokenizerVersion, validator_agnostic: MistralRequestValidator) -> None:
        messages = build_tool_call_messages(call_ids=("123456789",), result_ids=("123456789", "999999999"))
        if version >= TokenizerVersion.v11:
            with pytest.raises(InvalidMessageStructureException, match=r"Unexpected tool call id"):
                validator_agnostic.validate_messages(messages=messages)
        else:
            validator_agnostic.validate_messages(messages=messages)

    def test_duplicate_tool_response(
        self, version: TokenizerVersion, validator_agnostic: MistralRequestValidator
    ) -> None:
        messages = build_tool_call_messages(call_ids=("123456789",), result_ids=("123456789", "123456789"))
        if version >= TokenizerVersion.v11:
            with pytest.raises(
                InvalidMessageStructureException,
                match=r"Duplicate tool call id 123456789 in tool results",
            ):
                validator_agnostic.validate_messages(messages=messages)
        else:
            validator_agnostic.validate_messages(messages=messages)

    def test_final_null_tool_call_id(
        self, version: TokenizerVersion, validator_agnostic: MistralRequestValidator
    ) -> None:
        messages = build_tool_call_messages(call_ids=(None,))
        if version >= TokenizerVersion.v13:
            with pytest.raises(
                InvalidFunctionCallException,
                match=r"Tool call id must be a non-empty string other than 'null'",
            ):
                validator_agnostic.validate_messages(messages=messages)
        else:
            validator_agnostic.validate_messages(messages=messages)

    def test_non_final_null_tool_call_id(
        self, version: TokenizerVersion, validator_agnostic: MistralRequestValidator
    ) -> None:
        messages = [
            *build_tool_call_messages(call_ids=(None,)),
            UserMessage(content="continue"),
        ]
        if version < TokenizerVersion.v3:
            validator_agnostic.validate_messages(messages=messages)
        elif version >= TokenizerVersion.v13:
            with pytest.raises(
                InvalidFunctionCallException,
                match=r"Tool call id must be a non-empty string other than 'null'",
            ):
                validator_agnostic.validate_messages(messages=messages)
        else:
            with pytest.raises(InvalidFunctionCallException, match=r"Tool call id"):
                validator_agnostic.validate_messages(messages=messages)

    @pytest.mark.parametrize("weight", [None, 0, 1])
    def test_accepts_valid_finetuning_weight(
        self, weight: float | None, validator_agnostic: MistralRequestValidator
    ) -> None:
        validator_agnostic.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(content="answer", weight=weight),
            ],
        )

    def test_rejects_invalid_finetuning_weight(self, validator_agnostic: MistralRequestValidator) -> None:
        with pytest.raises(InvalidAssistantMessageException, match=r"Assistant message weight must be either 0 or 1"):
            validator_agnostic.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    FinetuningAssistantMessage(content="answer", weight=0.5),
                ],
            )


class TestValidateTools:
    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    def test_tool_function_bad_name(self, version: TokenizerVersion) -> None:
        validator = get_validator(version, ValidationMode.test)
        with pytest.raises(InvalidToolException, match=r"must be a-z, A-Z, 0-9"):
            validator._validate_tools(
                tools=[
                    Tool(
                        function=Function(
                            name=')_(_)~@:}>""',
                            parameters={
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                },
                            },
                        )
                    )
                ]
            )

    @pytest.mark.parametrize("version", _ALL_VERSIONS, ids=lambda v: v.value)
    def test_tool_function_invalid_schema(self, version: TokenizerVersion) -> None:
        validator = get_validator(version, ValidationMode.test)
        with pytest.raises(
            InvalidToolSchemaException, match=r"^Invalid tool schema: 32 is not valid under any of the given schemas"
        ):
            validator._validate_tools(tools=[Tool(function=Function(name="function_name", parameters={"type": 32}))])


@pytest.mark.parametrize(
    "version,expected_class",
    [
        (TokenizerVersion.v1, MistralRequestValidator),
        (TokenizerVersion.v2, MistralRequestValidator),
        (TokenizerVersion.v3, MistralRequestValidatorV3),
        (TokenizerVersion.v7, MistralRequestValidatorV5),
        (TokenizerVersion.v11, MistralRequestValidatorV11),
        (TokenizerVersion.v13, MistralRequestValidatorV13),
        (TokenizerVersion.v15, MistralRequestValidatorV15),
    ],
)
def test_get_validator_version_mapping(version: TokenizerVersion, expected_class: type) -> None:
    validator = get_validator(version, ValidationMode.test)
    assert isinstance(validator, expected_class)
    assert validator._mode == ValidationMode.test


@pytest.mark.parametrize("mode", list(ValidationMode))
def test_get_validator_mode_passed_correctly(mode: ValidationMode) -> None:
    validator = get_validator(TokenizerVersion.v2, mode)
    assert validator._mode == validator.mode == mode

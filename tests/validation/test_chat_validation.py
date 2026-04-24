import pytest

from mistral_common.exceptions import (
    InvalidAssistantMessageException,
    InvalidMessageStructureException,
    InvalidRequestException,
)
from mistral_common.protocol.instruct.chunk import AudioChunk, AudioURLChunk
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import (
    ChatCompletionRequest,
    ReasoningEffort,
)
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidator,
    MistralRequestValidatorV3,
    MistralRequestValidatorV5,
    MistralRequestValidatorV13,
    MistralRequestValidatorV15,
    ValidationMode,
)
from tests.fixtures.audio import get_dummy_audio_chunk, get_dummy_audio_url_chunk


@pytest.fixture(scope="module")
def audio_chunk() -> AudioChunk:
    return get_dummy_audio_chunk()


@pytest.fixture(scope="module")
def audio_url_chunk() -> AudioURLChunk:
    return get_dummy_audio_url_chunk()


@pytest.fixture(
    params=[
        MistralRequestValidator(ValidationMode.serving),
        MistralRequestValidatorV3(ValidationMode.serving),
        MistralRequestValidatorV13(ValidationMode.serving),
    ]
)
def validator(request: pytest.FixtureRequest) -> MistralRequestValidator:
    return request.param  # type: ignore


@pytest.fixture
def validator_v5() -> MistralRequestValidatorV5:
    return MistralRequestValidatorV5(ValidationMode.serving)


@pytest.fixture(
    params=[
        MistralRequestValidatorV13(ValidationMode.serving),
    ]
)
def validator_v13(request: pytest.FixtureRequest) -> MistralRequestValidator:
    return request.param  # type: ignore


@pytest.fixture(
    params=[
        MistralRequestValidatorV15(ValidationMode.serving),
    ]
)
def validator_v15(request: pytest.FixtureRequest) -> MistralRequestValidator:
    return request.param  # type: ignore


@pytest.fixture(
    params=[
        MistralRequestValidator(ValidationMode.agentic),
        MistralRequestValidatorV3(ValidationMode.agentic),
        MistralRequestValidatorV13(ValidationMode.agentic),
    ]
)
def agentic_validator(request: pytest.FixtureRequest) -> MistralRequestValidator:
    return request.param  # type: ignore


@pytest.fixture(
    params=[
        MistralRequestValidatorV13(ValidationMode.agentic),
    ]
)
def agentic_validator_v13(request: pytest.FixtureRequest) -> MistralRequestValidator:
    return request.param  # type: ignore


@pytest.fixture(
    params=[
        MistralRequestValidator(ValidationMode.serving),
        MistralRequestValidatorV3(ValidationMode.serving),
        MistralRequestValidatorV13(ValidationMode.serving),
        MistralRequestValidator(ValidationMode.agentic),
        MistralRequestValidatorV3(ValidationMode.agentic),
        MistralRequestValidatorV13(ValidationMode.agentic),
    ]
)
def serving_or_agentic_validator(request: pytest.FixtureRequest) -> MistralRequestValidator:
    return request.param  # type: ignore


class TestChatValidation:
    def test_multiple_system_messages_OK(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                SystemMessage(content="foo"),
                SystemMessage(content="foo"),
                SystemMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ],
            continue_final_message=False,
        )

    def test_system_user_messages_OK(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                SystemMessage(content="foo"),
                UserMessage(content="foo"),
            ],
            continue_final_message=False,
        )

    def test_user_system_messages_OK(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                SystemMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ],
            continue_final_message=False,
        )

    def test_user_user_messages_OK(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ],
            continue_final_message=False,
        )

    def test_empty_messages(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(InvalidMessageStructureException, match=r"Conversation must have at least one message"):
            validator.validate_messages(
                messages=[],
                continue_final_message=False,
            )

    def test_starts_with_system_or_user(self, validator: MistralRequestValidator) -> None:
        """
        This is allowed in validation, we will add a blank message in normalization
        """
        validator.validate_messages(
            messages=[
                AssistantMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ],
            continue_final_message=False,
        )

    def test_ends_with_assistant(self, serving_or_agentic_validator: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=(
                r"Expected last role User or Tool \(or Assistant with prefix or continue_final_message set to "
                r"True\) for serving but got assistant"
            ),
        ):
            serving_or_agentic_validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo"),
                ],
                continue_final_message=False,
            )

    def test_assistant_prefix(self, serving_or_agentic_validator: MistralRequestValidator) -> None:
        serving_or_agentic_validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(content="foo", prefix=True),
            ],
            continue_final_message=False,
        )
        with pytest.raises(
            InvalidAssistantMessageException,
            match=r"Assistant message with prefix True must be last message",
        ):
            serving_or_agentic_validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo", prefix=True),
                    UserMessage(content="foo"),
                ],
                continue_final_message=False,
            )

    def test_continue_final_message(self, serving_or_agentic_validator: MistralRequestValidator) -> None:
        serving_or_agentic_validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(content="foo"),
            ],
            continue_final_message=True,
        )
        with pytest.raises(
            InvalidMessageStructureException,
            match=(
                r"Expected last role Assistant with prefix False for serving with continue_final_message set to True "
                r"but got user"
            ),
        ):
            serving_or_agentic_validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo", prefix=True),
                    UserMessage(content="foo"),
                ],
                continue_final_message=True,
            )
        with pytest.raises(
            InvalidMessageStructureException,
            match=(
                r"Expected last role User or Tool \(or Assistant with prefix or continue_final_message set to True\)"
                r" for serving but got assistant"
            ),
        ):
            serving_or_agentic_validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo"),
                ],
                continue_final_message=False,
            )
        with pytest.raises(
            InvalidMessageStructureException,
            match=(
                r"Expected last role Assistant with prefix False for serving with continue_final_message set to True "
                r"but got assistant"
            ),
        ):
            serving_or_agentic_validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo", prefix=True),
                ],
                continue_final_message=True,
            )

    def test_user_tool_user(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(InvalidMessageStructureException, match=r"Unexpected role 'tool' after role 'user'"):
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    ToolMessage(content="foo"),
                    UserMessage(content="foo"),
                ],
                continue_final_message=False,
            )

    def test_model_none_serving_mode(self) -> None:
        validator = MistralRequestValidator[UserMessage, AssistantMessage, ToolMessage, SystemMessage](
            ValidationMode.serving
        )
        with pytest.raises(InvalidRequestException, match=r"Model name parameter is required for serving mode"):
            validator.validate_request(ChatCompletionRequest(messages=[UserMessage(content="foo")]))

    def test_model_none_test_mode(self) -> None:
        validator = MistralRequestValidator[UserMessage, AssistantMessage, ToolMessage, SystemMessage](
            ValidationMode.test
        )
        validator.validate_request(ChatCompletionRequest(messages=[UserMessage(content="foo")]))

    def test_tool_calls_OK(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(
                    tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}"))]
                ),
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
            ],
            continue_final_message=False,
        )

    def test_not_enough_tool_messages(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException, match=r"Not the same number of function calls and responses"
        ):
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(
                        tool_calls=[
                            ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}")),
                            ToolCall(id="999999999", function=FunctionCall(name="foo", arguments="{}")),
                        ]
                    ),
                    ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                    # missing ToolMessage
                ],
                continue_final_message=False,
            )

    def test_too_many_tool_messages(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=r"Not the same number of function calls and responses|Unexpected tool call id",
        ):
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(
                        tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}"))]
                    ),
                    ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                    ToolMessage(name="foo", content="bar", tool_call_id="999999999"),  # too many tool messages
                ],
                continue_final_message=False,
            )

    def test_build_settings_raises_error(self, validator: MistralRequestValidator) -> None:
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=ReasoningEffort.none)

        with pytest.raises(InvalidRequestException, match="reasoning_effort='none' is not supported for this model"):
            validator._validate_model_settings(request)

    def test_agentic_tool_then_user_ok(self, agentic_validator: MistralRequestValidator) -> None:
        agentic_validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(
                    tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}"))]
                ),
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                UserMessage(content="continue with this context"),
            ],
            continue_final_message=False,
        )

    def test_agentic_full_loop(self, agentic_validator: MistralRequestValidator) -> None:
        agentic_validator.validate_messages(
            messages=[
                UserMessage(content="search for info"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="aaaaaaaaa", function=FunctionCall(name="search", arguments="{}")),
                        ToolCall(id="ccccccccc", function=FunctionCall(name="search", arguments="{}")),
                    ]
                ),
                ToolMessage(name="search", content="result1", tool_call_id="aaaaaaaaa"),
                UserMessage(content="now refine the search"),
                AssistantMessage(
                    tool_calls=[ToolCall(id="bbbbbbbbb", function=FunctionCall(name="search", arguments="{}"))]
                ),
                ToolMessage(name="search", content="result2", tool_call_id="bbbbbbbbb"),
                AssistantMessage(content="here is the final answer"),
                UserMessage(content="thanks"),
            ],
            continue_final_message=False,
        )


class TestChatValidationV5:
    @pytest.mark.parametrize("audio_fixture", ["audio_chunk", "audio_url_chunk"])
    def test_audio_with_system_prompt_raises_error(
        self, validator_v5: MistralRequestValidatorV5, audio_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        audio_chunk: AudioChunk | AudioURLChunk = request.getfixturevalue(audio_fixture)
        with pytest.raises(
            ValueError,
            match=(
                r"Found system messages at indexes \[0\] and audio chunks in messages at indexes \[1\]\. This is not "
                r"allowed prior to the tokenizer version 13\."
            ),
        ):
            validator_v5.validate_messages(
                messages=[
                    SystemMessage(content="This is a system prompt"),
                    UserMessage(content=[audio_chunk]),
                ],
                continue_final_message=False,
            )

    @pytest.mark.parametrize("audio_fixture", ["audio_chunk", "audio_url_chunk"])
    def test_audio_without_system_prompt_ok(
        self, validator_v5: MistralRequestValidatorV5, audio_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        audio_chunk: AudioChunk | AudioURLChunk = request.getfixturevalue(audio_fixture)
        validator_v5.validate_messages(
            messages=[
                UserMessage(content=[audio_chunk]),
                UserMessage(content="User message after audio"),
            ],
            continue_final_message=False,
        )

    def test_system_prompt_without_audio_ok(self, validator_v5: MistralRequestValidatorV5) -> None:
        validator_v5.validate_messages(
            messages=[
                SystemMessage(content="This is a system prompt"),
                UserMessage(content="User message after system"),
            ],
            continue_final_message=False,
        )

    def test_build_settings_raises_error(self, validator: MistralRequestValidator) -> None:
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=ReasoningEffort.none)

        with pytest.raises(InvalidRequestException, match="reasoning_effort='none' is not supported for this model"):
            validator._validate_model_settings(request)


class TestChatValidationV13:
    def test_right_number_results_invalid_id(self, validator_v13: MistralRequestValidatorV13) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=r"Unexpected tool call id",
        ):
            validator_v13.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(
                        tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}"))]
                    ),
                    ToolMessage(name="foo", content="bar", tool_call_id="999999999"),  # invalid id
                ],
                continue_final_message=False,
            )

    def test_extra_results(self, validator_v13: MistralRequestValidatorV13) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=r"Unexpected tool call id",
        ):
            validator_v13.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(
                        tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}"))]
                    ),
                    ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                    ToolMessage(name="foo", content="bar", tool_call_id="999999999"),  # extra results
                ],
                continue_final_message=False,
            )

    def test_build_settings_raises_error(self, validator: MistralRequestValidator) -> None:
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=ReasoningEffort.none)

        with pytest.raises(InvalidRequestException, match="reasoning_effort='none' is not supported for this model"):
            validator._validate_model_settings(request)

    def test_parallel_call_missing_results(self, validator_v13: MistralRequestValidatorV13) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=r"Not the same number of function calls and responses",
        ):
            validator_v13.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(
                        tool_calls=[
                            ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}")),
                            ToolCall(id="999999999", function=FunctionCall(name="foo", arguments="{}")),
                        ]
                    ),
                    ToolMessage(name="foo", content="bar", tool_call_id="123456789"),  # missing ToolMessage
                ],
                continue_final_message=False,
            )

    def allow_tool_results_wrong_order(self, validator_v13: MistralRequestValidatorV13) -> None:
        validator_v13.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="999999999", function=FunctionCall(name="foo", arguments="{}")),
                    ]
                ),
                ToolMessage(name="foo", content="bar", tool_call_id="999999999"),  # invalid order
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
            ],
            continue_final_message=False,
        )

    def test_tool_call_duplicate_ids_same_assistant_messages(self, validator_v13: MistralRequestValidatorV13) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=r"Duplicate",
        ):
            validator_v13.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(
                        tool_calls=[
                            ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}")),
                            ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}")),
                        ]
                    ),
                    ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                ],
                continue_final_message=False,
            )

    def test_tool_call_duplicate_ids_different_assistant_messages(
        self, validator_v13: MistralRequestValidatorV13
    ) -> None:
        validator_v13.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}")),
                    ]
                ),
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}")),
                    ]
                ),
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
            ],
            continue_final_message=False,
        )

    def test_multiple_assistant_messages_some_with_tool_calls(self, validator_v13: MistralRequestValidatorV13) -> None:
        validator_v13.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}")),
                    ]
                ),
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                AssistantMessage(content="h"),
                UserMessage(content="foo"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}")),
                    ]
                ),
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                AssistantMessage(content="h"),
                UserMessage(content="foo"),
            ],
            continue_final_message=False,
        )

    @pytest.mark.parametrize("audio_fixture", ["audio_chunk", "audio_url_chunk"])
    def test_audio_with_system_prompt_raises_ok(
        self, validator_v13: MistralRequestValidatorV13, audio_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        audio_chunk: AudioChunk | AudioURLChunk = request.getfixturevalue(audio_fixture)
        validator_v13.validate_messages(
            messages=[
                SystemMessage(content="This is a system prompt"),
                UserMessage(content=[audio_chunk]),
            ],
            continue_final_message=False,
        )

    def test_agentic_partial_tool_results_then_user_ok(self, agentic_validator_v13: MistralRequestValidator) -> None:
        agentic_validator_v13.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="aaaaaaaaa", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="bbbbbbbbb", function=FunctionCall(name="bar", arguments="{}")),
                        ToolCall(id="ccccccccc", function=FunctionCall(name="baz", arguments="{}")),
                    ]
                ),
                ToolMessage(name="foo", content="result1", tool_call_id="aaaaaaaaa"),
                UserMessage(content="stop, only need the first result"),
            ],
            continue_final_message=False,
        )

    def test_agentic_user_after_partial_results_then_new_tool_calls(
        self, agentic_validator_v13: MistralRequestValidator
    ) -> None:
        agentic_validator_v13.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="aaaaaaaaa", function=FunctionCall(name="foo", arguments="{}")),
                        ToolCall(id="bbbbbbbbb", function=FunctionCall(name="bar", arguments="{}")),
                    ]
                ),
                ToolMessage(name="foo", content="result1", tool_call_id="aaaaaaaaa"),
                UserMessage(content="skip bar, try something else"),
                AssistantMessage(
                    tool_calls=[ToolCall(id="ccccccccc", function=FunctionCall(name="baz", arguments="{}"))]
                ),
                ToolMessage(name="baz", content="result3", tool_call_id="ccccccccc"),
                UserMessage(content="done"),
            ],
            continue_final_message=False,
        )


class TestChatValidationV15:
    @pytest.mark.parametrize("reasoning_effort", [*list(ReasoningEffort), None])
    def test_build_settings_v15_reasoning_effort(
        self, reasoning_effort: ReasoningEffort | None, validator_v15: MistralRequestValidatorV15
    ) -> None:
        request = ChatCompletionRequest(messages=[UserMessage(content="Hello")], reasoning_effort=reasoning_effort)
        validator_v15._validate_model_settings(request)

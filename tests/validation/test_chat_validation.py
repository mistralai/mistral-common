import pytest

from mistral_common.exceptions import (
    InvalidAssistantMessageException,
    InvalidMessageStructureException,
    InvalidRequestException,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidator,
    MistralRequestValidatorV3,
    MistralRequestValidatorV13,
    ValidationMode,
)


@pytest.fixture(
    params=[
        MistralRequestValidator(ValidationMode.serving),
        MistralRequestValidatorV3(ValidationMode.serving),
        MistralRequestValidatorV13(ValidationMode.serving),
    ]
)
def validator(request: pytest.FixtureRequest) -> MistralRequestValidator:
    return request.param  # type: ignore


@pytest.fixture(
    params=[
        MistralRequestValidatorV13(ValidationMode.serving),
    ]
)
def validator_v13(request: pytest.FixtureRequest) -> MistralRequestValidator:
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

    def test_ends_with_assistant(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=(
                r"Expected last role User or Tool \(or Assistant with prefix or continue_final_message set to "
                r"True\) for serving but got assistant"
            ),
        ):
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo"),
                ],
                continue_final_message=False,
            )

    def test_assistant_prefix(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
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
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo", prefix=True),
                    UserMessage(content="foo"),
                ],
                continue_final_message=False,
            )

    def test_continue_final_message(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
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
            validator.validate_messages(
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
            validator.validate_messages(
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
            validator.validate_messages(
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


class TestChatValidationV6:
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

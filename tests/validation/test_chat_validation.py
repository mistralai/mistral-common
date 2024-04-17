import pytest
from mistral_common.exceptions import (
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
    ValidationMode,
)


@pytest.fixture(
    params=[MistralRequestValidator(ValidationMode.serving), MistralRequestValidatorV3(ValidationMode.serving)]
)
def validator(request: pytest.FixtureRequest) -> MistralRequestValidator:
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
        )

    def test_system_user_messages_OK(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                SystemMessage(content="foo"),
                UserMessage(content="foo"),
            ]
        )

    def test_user_system_messages_OK(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                SystemMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ]
        )

    def test_user_user_messages_OK(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ]
        )

    def test_empty_messages(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(InvalidMessageStructureException, match=r"Conversation must have at least one message"):
            validator.validate_messages(messages=[])

    def test_starts_with_system_or_user(self, validator: MistralRequestValidator) -> None:
        """
        This is allowed in validation, we will add a blank message in normalization
        """
        validator.validate_messages(
            messages=[
                AssistantMessage(content="foo"),
                UserMessage(content="foo"),  # so we don't get an error for ending with a system message
            ]
        )

    def test_ends_with_assistant(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException,
            match=r"Expected last role to be one of: \[(tool|user), (tool|user)\] but got assistant",
        ):
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(content="foo"),
                ],
            )

    def test_user_tool_user(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(InvalidMessageStructureException, match=r"Unexpected role 'tool' after role 'user'"):
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    ToolMessage(content="foo"),
                    UserMessage(content="foo"),
                ],
            )

    def test_model_none_chat_mode(self) -> None:
        validator = MistralRequestValidator(ValidationMode.serving)
        with pytest.raises(InvalidRequestException, match=r"Model name parameter is required for serving mode"):
            validator.validate_request(ChatCompletionRequest(messages=[UserMessage(content="foo")]))

    def test_model_none_test_mode(self, validator: MistralRequestValidator) -> None:
        validator = MistralRequestValidator(ValidationMode.test)
        validator.validate_request(ChatCompletionRequest(messages=[UserMessage(content="foo")]))

    def test_tool_calls_OK(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                AssistantMessage(
                    tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}"))]
                ),
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
            ]
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
                ]
            )

    def test_too_many_tool_messages(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidMessageStructureException, match=r"Not the same number of function calls and responses"
        ):
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    AssistantMessage(
                        tool_calls=[ToolCall(id="123456789", function=FunctionCall(name="foo", arguments="{}"))]
                    ),
                    ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                    ToolMessage(name="foo", content="bar", tool_call_id="999999999"),  # too many tool messages
                ]
            )

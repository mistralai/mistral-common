import pytest

from mistral_common.exceptions import (
    InvalidAssistantMessageException,
    InvalidFunctionCallException,
    InvalidMessageStructureException,
)
from mistral_common.protocol.instruct.messages import (
    FinetuningAssistantMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidator,
    MistralRequestValidatorV3,
    ValidationMode,
)


@pytest.fixture(
    params=[MistralRequestValidator(ValidationMode.finetuning), MistralRequestValidatorV3(ValidationMode.finetuning)]
)
def validator(request: pytest.FixtureRequest) -> MistralRequestValidator:
    return request.param  # type: ignore


class TestFineTuningValidation:
    # This fails in chat validation
    def test_ends_with_assistant(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(content="foo"),
            ],
            continue_final_message=False,
        )

    def test_has_weight_in_message_assistant(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(content="foo", weight=1),
            ],
            continue_final_message=False,
        )

    def test_has_invalid_weight_in_message_assistant(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(InvalidAssistantMessageException, match=r"Assistant message weight must be either 0 or 1"):
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    FinetuningAssistantMessage(content="foo", weight=0.5),
                ],
                continue_final_message=False,
            )

    def test_should_allow_multiple_weights(self, validator: MistralRequestValidator) -> None:
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(content="foo", weight=0),
                FinetuningAssistantMessage(content="foo", weight=1),
                FinetuningAssistantMessage(content="foo", weight=1),
            ],
            continue_final_message=False,
        )

    def test_ends_with_tool_call_null(self, validator: MistralRequestValidator) -> None:
        if not isinstance(validator, MistralRequestValidatorV3):
            pytest.skip("MistralRequestValidator v1 does not validate tool call ids")

        function = FunctionCall(name="foo", arguments='{"a": 1}')
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(tool_calls=[ToolCall(id="123456789", function=function)]),
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                # tool_call id left "null" as final message => OK!
                FinetuningAssistantMessage(tool_calls=[ToolCall(function=function)]),
            ],
            continue_final_message=False,
        )

    def test_ends_with_no_tool_call(self, validator: MistralRequestValidator) -> None:
        if not isinstance(validator, MistralRequestValidatorV3):
            pytest.skip("MistralRequestValidator v1 does not validate tool call ids")

        function = FunctionCall(name="foo", arguments='{"a": 1}')
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(tool_calls=[ToolCall(id="123456789", function=function)]),
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                FinetuningAssistantMessage(content="foo"),
            ],
            continue_final_message=False,
        )

    def test_middle_with_tool_call_null_raises(self, validator: MistralRequestValidator) -> None:
        if not isinstance(validator, MistralRequestValidatorV3):
            pytest.skip("MistralRequestValidator v1 does not validate tool call ids")

        function = FunctionCall(name="foo", arguments='{"a": 1}')
        expected_err = r"Tool call id of assistant message that is not last has to be defined in finetuning mode."
        with pytest.raises(InvalidFunctionCallException, match=expected_err):
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    # tool_call id left "null" as non-final message => Raise!
                    FinetuningAssistantMessage(tool_calls=[ToolCall(function=function)]),
                    ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                    FinetuningAssistantMessage(tool_calls=[ToolCall(function=function)]),
                ],
                continue_final_message=False,
            )

    def test_one_message_with_user_is_not_valid(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(InvalidMessageStructureException) as exc:
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                ],
                continue_final_message=False,
            )
        assert str(exc.value) == "Expected last role Assistant for finetuning but got user"

    def test_parallel_tool_call(self, validator: MistralRequestValidator) -> None:
        function = FunctionCall(name="foo", arguments='{"a": 1}')
        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(
                    tool_calls=[
                        ToolCall(id="123456789", function=function),
                        ToolCall(id="912345678", function=function),
                    ]
                ),
            ],
            continue_final_message=False,
        )

        validator.validate_messages(
            messages=[
                UserMessage(content="foo"),
                FinetuningAssistantMessage(
                    tool_calls=[
                        ToolCall(id="123456789", function=function),
                        ToolCall(id="912345678", function=function),
                    ]
                ),
                ToolMessage(name="foo", content="bar", tool_call_id="912345678"),
                ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                FinetuningAssistantMessage(content="foo"),
            ],
            continue_final_message=False,
        )

        with pytest.raises(InvalidMessageStructureException):
            validator.validate_messages(
                messages=[
                    UserMessage(content="foo"),
                    FinetuningAssistantMessage(
                        tool_calls=[
                            ToolCall(id="123456789", function=function),
                            ToolCall(id="912345678", function=function),
                        ]
                    ),
                    ToolMessage(name="foo", content="bar", tool_call_id="912345678"),
                    ToolMessage(name="foo", content="bar", tool_call_id="123456789"),
                    ToolMessage(name="foo", content="bar", tool_call_id="891234567"),
                    FinetuningAssistantMessage(content="foo"),
                ],
                continue_final_message=False,
            )

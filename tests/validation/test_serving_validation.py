from typing import Optional

import pytest
from mistral_common.exceptions import (
    InvalidAssistantMessageException,
    InvalidFunctionCallException,
    InvalidToolMessageException,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    Roles,
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
    params=[
        MistralRequestValidator(ValidationMode.serving),
        MistralRequestValidatorV3(ValidationMode.serving),
    ]
)
def validator(request: pytest.FixtureRequest) -> MistralRequestValidator:
    return request.param  # type: ignore


class TestValidateAssistantMessage:
    def test_content_and_tool_calls(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidAssistantMessageException,
            match=r"Assistant message must have either content or tool_calls, but not both.",
        ):
            validator._validate_assistant_message(
                AssistantMessage(
                    role=Roles.assistant,
                    content="a",
                    tool_calls=[
                        ToolCall(function=FunctionCall(name="a", arguments='{"a": 1}'))
                    ],
                ),
            )

    @pytest.mark.parametrize("content,tool_calls", [("", None), (None, None), ("", []), (None, [])])
    def test_no_content_and_no_tool_calls(self,
                                          content: Optional[str],
                                          tool_calls: Optional[list],
                                          validator: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidAssistantMessageException,
            match=r"Assistant message must have either content or tool_calls, but not both.",
        ):
            validator._validate_assistant_message(
                AssistantMessage(
                    role=Roles.assistant,
                    content=content,
                    tool_calls=tool_calls
                ),
            )


    def test_no_content_and_tool_calls(self, validator: MistralRequestValidator) -> (
        None
    ):
        validator._validate_assistant_message(
            AssistantMessage(
                role=Roles.assistant,
                content="",
                tool_calls=[
                    ToolCall(
                        id="AbCd56789",
                        function=FunctionCall(
                            name="function_call", arguments='{"a": 1}'
                        ),
                    )
                ],
            ),
        )

    def test_content_and_no_tool_calls(self, validator: MistralRequestValidator) -> None:
        validator._validate_assistant_message(
            AssistantMessage(
                role=Roles.assistant,
                content="content",
                tool_calls=[],
            ),
        )

    def test_bad_function_name(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidFunctionCallException, match=r"must be a-z, A-Z, 0-9"
        ):
            validator._validate_assistant_message(
                AssistantMessage(
                    role=Roles.assistant,
                    tool_calls=[
                        ToolCall(
                            id="AbCd56789",
                            function=FunctionCall(
                                name=')_(_)~@:}>""', arguments='{"a": 1}'
                            ),
                        )
                    ],
                ),
            )

    @pytest.mark.skip(
        reason="If json parse fails we assume that its plain text like '22' or 'a'"
    )
    def test_invalid_function_json(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(
            InvalidFunctionCallException, match=r"must be a-z, A-Z, 0-9"
        ):
            validator._validate_assistant_message(
                AssistantMessage(
                    role=Roles.assistant,
                    tool_calls=[
                        ToolCall(
                            function=FunctionCall(
                                name="function_name", arguments='{"a":aewfawef 1}'
                            )
                        )
                    ],
                )
            )

    def test_tool_id_defined(self, validator: MistralRequestValidator) -> None:
        if not isinstance(validator, MistralRequestValidatorV3):
            pytest.skip("MistralRequestValidator v1 does not validate tool call ids")
        with pytest.raises(
            InvalidFunctionCallException, match=r"Tool call id has to be defined"
        ):
            validator._validate_assistant_message(
                AssistantMessage(
                    role=Roles.assistant,
                    tool_calls=[
                        ToolCall(
                            function=FunctionCall(name="foo", arguments='{"a": 1}')
                        )
                    ],
                )
            )

    def test_tool_calls_bad_id(self, validator: MistralRequestValidator) -> None:
        if not isinstance(validator, MistralRequestValidatorV3):
            pytest.skip("MistralRequestValidator v1 does not validate tool call ids")
        with pytest.raises(
            InvalidFunctionCallException,
            match=r"Tool call id was bad_id but must be a-z, A-Z, 0-9, with a length of 9.",
        ):
            validator._validate_assistant_message(
                AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="bad_id",
                            function=FunctionCall(name="foo", arguments="{}"),
                        )
                    ]
                )
            )


class TestValidateUserMessage:
    def test_user_message_blank_content_OK(
        self, validator: MistralRequestValidator
    ) -> None:
        validator._validate_user_message(UserMessage(content=""))


class TestValidateToolMessage:
    def test_tool_message_bad_name(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(InvalidToolMessageException, match=r"must be a-z, A-Z, 0-9"):
            validator._validate_tool_message(ToolMessage(name="~@}£Q_+", content="{}"))

    @pytest.mark.skip(
        reason="If json parse fails we assume that its plain text like '22' or 'a'"
    )
    def test_tool_message_invalid_json(self, validator: MistralRequestValidator) -> (
        None
    ):
        with pytest.raises(InvalidToolMessageException, match=r"must be a-z, A-Z, 0-9"):
            validator._validate_tool_message(
                ToolMessage(name="function_name", content='{"afewfaef}')
            )

    def test_tool_message_bad_id(self, validator: MistralRequestValidator) -> None:
        if not isinstance(validator, MistralRequestValidatorV3):
            pytest.skip("MistralRequestValidator v1 does not validate tool call ids")
        with pytest.raises(
            InvalidToolMessageException,
            match=r"Tool call id was bad_id but must be a-z, A-Z, 0-9, with a length of 9.",
        ):
            validator._validate_tool_message(
                ToolMessage(name="foo", content="bar", tool_call_id="bad_id"),
            )

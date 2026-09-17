import pytest

from mistral_common.exceptions import InvalidMessageStructureException
from mistral_common.protocol.instruct.messages import AssistantMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.base import TokenizerVersion


@pytest.mark.parametrize("version", [TokenizerVersion.v3, TokenizerVersion.v7])
def test_rejects_tool_result_with_mismatched_call_id(version: TokenizerVersion) -> None:
    validator = get_validator(version=version, mode=ValidationMode.serving)

    with pytest.raises(InvalidMessageStructureException, match="Unexpected tool call id 999999999"):
        validator.validate_messages(
            messages=[
                UserMessage(content="Run the tool"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="123456789",
                            function=FunctionCall(name="example", arguments="{}"),
                        )
                    ]
                ),
                ToolMessage(content="result", tool_call_id="999999999"),
            ]
        )


@pytest.mark.parametrize("version", [TokenizerVersion.v3, TokenizerVersion.v7])
def test_allows_tool_result_with_matching_call_id(version: TokenizerVersion) -> None:
    validator = get_validator(version=version, mode=ValidationMode.serving)

    validator.validate_messages(
        messages=[
            UserMessage(content="Run the tool"),
            AssistantMessage(
                tool_calls=[
                    ToolCall(
                        id="123456789",
                        function=FunctionCall(name="example", arguments="{}"),
                    )
                ]
            ),
            ToolMessage(content="result", tool_call_id="123456789"),
        ]
    )

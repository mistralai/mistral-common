import pytest

from mistral_common.exceptions import (
    InvalidToolException,
    InvalidToolSchemaException,
)
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.protocol.instruct.validator import MistralRequestValidator


class TestValidateTools:
    @pytest.fixture
    def validator(self) -> MistralRequestValidator:
        return MistralRequestValidator()

    def test_tool_function_bad_name(self, validator: MistralRequestValidator) -> None:
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

    def test_tool_function_invalid_schema(self, validator: MistralRequestValidator) -> None:
        with pytest.raises(InvalidToolSchemaException, match=r"32 is not valid under any of the given schemas"):
            validator._validate_tools(tools=[Tool(function=Function(name="function_name", parameters={"type": 32}))])

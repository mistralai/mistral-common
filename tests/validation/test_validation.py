import pytest

from mistral_common.exceptions import (
    InvalidToolException,
    InvalidToolSchemaException,
)
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.protocol.instruct.validator import (
    MistralRequestValidator,
    MistralRequestValidatorV3,
    MistralRequestValidatorV5,
    MistralRequestValidatorV13,
    ValidationMode,
    get_validator,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion


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


@pytest.mark.parametrize(
    "version,expected_class",
    [
        (TokenizerVersion.v1, MistralRequestValidator),
        (TokenizerVersion.v2, MistralRequestValidator),
        (TokenizerVersion.v3, MistralRequestValidatorV3),
        (TokenizerVersion.v7, MistralRequestValidatorV5),
        (TokenizerVersion.v13, MistralRequestValidatorV13),
    ],
)
def test_get_validator_version_mapping(version: TokenizerVersion, expected_class: type) -> None:
    validator = get_validator(version, ValidationMode.test)
    assert isinstance(validator, expected_class)
    assert validator._mode == ValidationMode.test


@pytest.mark.parametrize("mode", [ValidationMode.serving, ValidationMode.finetuning, ValidationMode.test])
def test_get_validator_mode_passed_correctly(mode: ValidationMode) -> None:
    validator = get_validator(TokenizerVersion.v2, mode)
    assert validator._mode == validator.mode == mode

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
    MistralCommonException,
    TokenizerException,
    UnsupportedTokenizerFeatureException,
)


@pytest.mark.parametrize(
    "message, expected",
    [
        pytest.param(None, "Internal server error", id="default"),
        pytest.param("", "Internal server error", id="empty"),
        pytest.param("failure", "failure", id="custom"),
    ],
)
def test_mistral_common_exception_message(message: str | None, expected: str) -> None:
    assert MistralCommonException(message).message == expected


@pytest.mark.parametrize(
    "exception_type",
    [
        pytest.param(TokenizerException, id="TokenizerException"),
        pytest.param(UnsupportedTokenizerFeatureException, id="UnsupportedTokenizerFeatureException"),
        pytest.param(InvalidRequestException, id="InvalidRequestException"),
        pytest.param(InvalidSystemPromptException, id="InvalidSystemPromptException"),
        pytest.param(InvalidMessageStructureException, id="InvalidMessageStructureException"),
        pytest.param(InvalidAssistantMessageException, id="InvalidAssistantMessageException"),
        pytest.param(InvalidToolMessageException, id="InvalidToolMessageException"),
        pytest.param(InvalidToolSchemaException, id="InvalidToolSchemaException"),
        pytest.param(InvalidUserMessageException, id="InvalidUserMessageException"),
        pytest.param(InvalidFunctionCallException, id="InvalidFunctionCallException"),
        pytest.param(InvalidToolException, id="InvalidToolException"),
    ],
)
def test_specialized_exceptions_preserve_message(exception_type: type[MistralCommonException]) -> None:
    assert exception_type("bad request").message == "bad request"

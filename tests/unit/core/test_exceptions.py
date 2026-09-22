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
    "exception_type",
    [
        TokenizerException,
        UnsupportedTokenizerFeatureException,
        InvalidRequestException,
        InvalidSystemPromptException,
        InvalidMessageStructureException,
        InvalidAssistantMessageException,
        InvalidToolMessageException,
        InvalidToolSchemaException,
        InvalidUserMessageException,
        InvalidFunctionCallException,
        InvalidToolException,
    ],
)
def test_specialized_exceptions_preserve_message(exception_type: type[MistralCommonException]) -> None:
    assert exception_type("bad request").message == "bad request"


def test_base_exception_uses_default_and_ignores_empty_override() -> None:
    assert MistralCommonException().message == "Internal server error"
    assert MistralCommonException("").message == "Internal server error"
    assert MistralCommonException("failure").message == "failure"

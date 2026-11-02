from mistral_common.exceptions import InvalidRequestException, MistralCommonException


def test_mistral_common_exception_uses_default_message() -> None:
    error = MistralCommonException()

    assert error.message == "Internal server error"


def test_mistral_common_exception_uses_custom_message() -> None:
    error = MistralCommonException(message="custom error")

    assert error.message == "custom error"


def test_subclass_uses_custom_message() -> None:
    error = InvalidRequestException(message="invalid request")

    assert error.message == "invalid request"

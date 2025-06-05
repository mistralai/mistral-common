from typing import Optional


class MistralCommonException(Exception):
    r"""Base class for all Mistral exceptions.

    Attributes:
        message: A human-readable message describing the error.
    """

    message: str = "Internal server error"

    def __init__(
        self,
        message: Optional[str] = None,
    ) -> None:
        r"""Initialize the `MistralCommonException` with an optional message.

        If no message is provided, the default message is used.

        Args:
           message: A human-readable message describing the error.
        """
        if message:
            self.message = message


class TokenizerException(MistralCommonException):
    r"""Exception raised for errors in the tokenizer."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `TokenizerException` with a message.

        Args:
          message: A human-readable message describing the error.
        """
        super().__init__(message)


class UnsupportedTokenizerFeatureException(MistralCommonException):
    r"""Exception raised for unsupported features in the tokenizer."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `UnsupportedTokenizerFeatureException` with a message.

        Args:
           message: A human-readable message describing the error.
        """
        super().__init__(message)


class InvalidRequestException(MistralCommonException):
    r"""Exception raised for invalid requests."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `InvalidRequestException` with a message.

        Args:
           message: A human-readable message describing the error.
        """
        super().__init__(message)


class InvalidSystemPromptException(MistralCommonException):
    r"""Exception raised for invalid system prompts."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `InvalidSystemPromptException` with a message.

        Args:
           message: A human-readable message describing the error.
        """
        super().__init__(message)


class InvalidMessageStructureException(MistralCommonException):
    r"""Exception raised for invalid message structures."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `InvalidMessageStructureException` with a message.

        Args:
           message: A human-readable message describing the error.
        """
        super().__init__(message)


class InvalidAssistantMessageException(MistralCommonException):
    r"""Exception raised for invalid assistant messages."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `InvalidAssistantMessageException` with a message.

        Args:
           message: A human-readable message describing the error.
        """
        super().__init__(message)


class InvalidToolMessageException(MistralCommonException):
    r"""Exception raised for invalid tool messages."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `InvalidToolMessageException` with a message.

        Args:
           message: A human-readable message describing the error.
        """
        super().__init__(message)


class InvalidToolSchemaException(MistralCommonException):
    r"""Exception raised for invalid tool schemas."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `InvalidToolSchemaException` with a message.

        Args:
           message: A human-readable message describing the error.
        """
        super().__init__(message)


class InvalidUserMessageException(MistralCommonException):
    r"""Exception raised for invalid user messages."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `InvalidUserMessageException` with a message.

        Args:
           message: A human-readable message describing the error.
        """
        super().__init__(message)


class InvalidFunctionCallException(MistralCommonException):
    r"""Exception raised for invalid function calls."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `InvalidFunctionCallException` with a message.

        Args:
           message: A human-readable message describing the error.
        """
        super().__init__(message)


class InvalidToolException(MistralCommonException):
    r"""Exception raised for invalid tools."""

    def __init__(self, message: str) -> None:
        r"""Initialize the `InvalidToolException` with a message.

        Args:
           message: A human-readable message describing the error.
        """
        super().__init__(message)

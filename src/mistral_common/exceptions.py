from typing import Optional


class MistralCommonException(Exception):
    message: str = "Internal server error"

    def __init__(
        self,
        message: Optional[str] = None,
    ) -> None:
        if message:
            self.message = message


class TokenizerException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnsupportedTokenizerFeatureException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidRequestException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidSystemPromptException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidMessageStructureException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidAssistantMessageException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidToolMessageException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidToolSchemaException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidUserMessageException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidFunctionCallException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidToolException(MistralCommonException):
    def __init__(self, message: str) -> None:
        super().__init__(message)

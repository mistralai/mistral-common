import importlib.metadata
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_settings import BaseSettings

from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


class OpenAIChatCompletionRequest(BaseModel):
    r"""OpenAI chat completion request.

    This class is used to parse the request body for the OpenAI chat completion endpoint.

    Attributes:
        messages: The messages to use for the chat completion.
        tools: The tools to use for the chat completion.

    Note:
        This class accepts extra fields that are not validated.
    """

    messages: List[dict[str, Union[str, List[dict[str, Union[str, dict[str, Any]]]]]]]
    tools: Optional[List[dict[str, Any]]] = None

    # Allow extra fields as the `from_openai` method will handle them.
    # We never validate the input, so we don't need to worry about the extra fields.
    model_config = ConfigDict(extra="allow")

    def drop_extra_fields(self) -> dict[str, Any]:
        r"""Drop extra fields from the model.

        This method is used to drop extra fields from the model, which are not defined in the model fields.

        Returns:
            The extra fields that were dropped from the model.
        """
        extra_fields = {
            field: value for field, value in self.model_dump().items() if field not in type(self).model_fields
        }

        self.__dict__ = {k: v for k, v in self.__dict__.items() if k not in extra_fields}
        return extra_fields


class EngineBackend(str, Enum):
    r"""The engine backend to use.

    Attributes:
        llama_cpp: The llama.cpp backend.
    """

    llama_cpp = "llama_cpp"


class Settings(BaseSettings):
    r"""Settings for the Mistral-common API.

    Attributes:
        app_name: The name of the application.
        app_version: The version of the application.
        engine_url: The URL of the engine.
        engine_backend: The backend to use for the engine.
        timeout: The timeout to use for the engine API.
    """

    app_name: str = "Mistral-common API"
    app_version: str = importlib.metadata.version("mistral-common")
    engine_url: str = "127.0.0.1"
    engine_backend: EngineBackend = EngineBackend.llama_cpp
    api_key: str = ""
    timeout: int = 60

    @field_validator("engine_url", mode="before")
    @classmethod
    def _validate_engine_url(cls, value: str) -> str:
        if isinstance(value, str) and value.endswith("/"):
            value = value[:-1]
        return value

    @field_validator("engine_backend", mode="before")
    @classmethod
    def _validate_backend(cls, value: Union[str, EngineBackend]) -> EngineBackend:
        if isinstance(value, str):
            value = EngineBackend(value)
        return value

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._tokenizer: Optional[MistralTokenizer] = None

    def _load_tokenizer(self, tokenizer_path: Union[str, Path], validation_mode: ValidationMode) -> None:
        if tokenizer_path == "":
            raise ValueError("Tokenizer path must be set via the environment variable `TOKENIZER_PATH`.")
        elif self._tokenizer is not None:
            raise ValueError("Tokenizer has already been initialized.")

        if isinstance(tokenizer_path, str):
            candidate_tokenizer_path = Path(tokenizer_path)
            if candidate_tokenizer_path.exists():
                tokenizer_path = candidate_tokenizer_path

        if isinstance(tokenizer_path, Path) and tokenizer_path.exists():
            self._tokenizer = MistralTokenizer.from_file(tokenizer_path, mode=validation_mode)
        else:
            self._tokenizer = MistralTokenizer.from_hf_hub(str(tokenizer_path), mode=validation_mode)

    @property
    def tokenizer(self) -> MistralTokenizer:
        if self._tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: MistralTokenizer) -> None:
        if not isinstance(value, MistralTokenizer):
            raise ValueError("Tokenizer must be an instance of MistralTokenizer.")
        self._tokenizer = value


def get_settings() -> Settings:
    r"""Get the settings for the Mistral-common API."""
    return Settings()

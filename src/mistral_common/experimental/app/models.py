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

    Attributes:
        messages: The messages to use for the chat completion.
        tools: The tools to use for the chat completion.

    Note:
        This class accepts extra fields, as the
        [from_openai][mistral_common.protocol.instruct.request.ChatCompletionRequest.from_openai] method will handle
        them.
    """

    messages: List[dict[str, Union[str, List[dict[str, Union[str, dict[str, Any]]]]]]]
    tools: Optional[List[dict[str, Any]]] = None

    # Allow extra fields as the `from_openai` method will handle them.
    # We never validate the input, so we don't need to worry about the extra fields.
    model_config = ConfigDict(extra="allow")


class GenerationBackend(str, Enum):
    r"""The generation backend to use.

    Attributes:
        llama_cpp: The llama.cpp backend.
    """

    llama_cpp = "llama_cpp"


class Settings(BaseSettings):
    r"""Settings for the Mistral-common API.

    Attributes:
        app_name: The name of the application.
        app_version: The version of the application.
        generation_host: The host to use for the generation API.
        generation_port: The port to use for the generation API.
        api_key: The API key to use for the generation API.
        timeout: The timeout to use for the generation API.
    """

    app_name: str = "Mistral-common API"
    app_version: str = importlib.metadata.version("mistral-common")
    generation_host: str = "127.0.0.1"
    generation_port: int = 8080
    generation_backend: GenerationBackend = GenerationBackend.llama_cpp
    api_key: str = ""
    timeout: int = 60

    @field_validator("generation_backend", mode="before")
    @classmethod
    def _validate_backend(cls, value: Union[str, GenerationBackend]) -> GenerationBackend:
        if isinstance(value, str):
            value = GenerationBackend(value)
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

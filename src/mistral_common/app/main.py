import importlib.metadata
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Union

import click
import uvicorn
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import BaseSettings

from mistral_common.app.utils import InvalidtoolCallError, decode_tool_call, find_content_tool_calls
from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessageType
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenized
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


class Settings(BaseSettings):
    r"""Settings for the Mistral-common API.

    Attributes:
        app_name: The name of the application.
        app_version: The version of the application.
    """

    app_name: str = "Mistral-common API"
    app_version: str = importlib.metadata.version("mistral-common")

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._tokenizer: Optional[MistralTokenizer] = None

    def _initialize_tokenizer(self, tokenizer_path: Union[str, Path], validation_mode: ValidationMode) -> None:
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


main_router = APIRouter(tags=["app"])
tokenize_router = APIRouter(prefix="/tokenize", tags=["tokenizer", "tokenize"])
decode_router = APIRouter(prefix="/detokenize", tags=["tokenizer", "detokenize"])


def get_settings() -> Settings:
    r"""Get the settings for the Mistral-common API."""
    return Settings()


@main_router.get("/")
def redirect_to_docs() -> RedirectResponse:
    r"""Redirect to the documentation."""
    return RedirectResponse(url="docs")


@main_router.get("/info")
def get_info(settings: Annotated[Settings, Depends(get_settings)]) -> Dict[str, str]:
    r"""Get the information about the Mistral-common API."""
    return {"app_name": settings.app_name, "app_version": settings.app_version}


def _tokenize_request(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest],
    settings: Annotated[Settings, Depends(get_settings)],
) -> List[int]:
    if isinstance(request, OpenAIChatCompletionRequest):
        try:
            request = ChatCompletionRequest.from_openai(**request.model_dump(exclude_none=True))
        except (ValidationError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    if request.messages == []:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    tokenized = settings.tokenizer.encode_chat_completion(request)
    assert isinstance(tokenized, Tokenized), type(tokenized)
    return tokenized.tokens


def _tokenize_messages(
    messages: list[ChatMessageType], settings: Annotated[Settings, Depends(get_settings)]
) -> List[int]:
    if len(messages) == 0:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
    request = ChatCompletionRequest(messages=messages)

    return _tokenize_request(request, settings)


@tokenize_router.post("/messages")
def tokenize_messages(
    settings: Annotated[Settings, Depends(get_settings)], messages: list[ChatMessageType] = Body(default_factory=list)
) -> list[int]:
    r"""Tokenize a list of messages."""
    return _tokenize_messages(messages, settings=settings)


@tokenize_router.post("/prompt")
def tokenize_prompt(
    settings: Annotated[Settings, Depends(get_settings)],
    prompt: str = Body(default_factory=str),
    add_special: bool = Body(default=True),
) -> list[int]:
    r"""Tokenize a prompt."""
    if prompt == "":
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    return settings.tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=add_special, eos=add_special)


@tokenize_router.post("/request")
def tokenize_request(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest],
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[int]:
    r"""Tokenize a chat completion request."""
    return _tokenize_request(request, settings=settings)


@decode_router.post("/")
def detokenize_tokens(
    settings: Annotated[Settings, Depends(get_settings)],
    tokens: list[int] = Body(default_factory=list),
    special_token_policy: SpecialTokenPolicy = Body(default=SpecialTokenPolicy.IGNORE),
) -> str:
    r"""Detokenize a list of tokens."""
    if len(tokens) == 0:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")

    try:
        return settings.tokenizer.decode(tokens, special_token_policy=special_token_policy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@decode_router.post("/assistant")
def detokenize_to_assistant_message(
    settings: Annotated[Settings, Depends(get_settings)],
    tokens: list[int] = Body(default_factory=list),
) -> AssistantMessage:
    r"""Detokenize a list of tokens to an assistant message.

    Parse tool calls from the tokens and extract content before the first tool call.
    """
    if len(tokens) == 0:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")

    content_tokens, tool_calls_tokens = find_content_tool_calls(
        tokens, settings.tokenizer.instruct_tokenizer.tokenizer.get_control_token("[TOOL_CALLS]")
    )

    if content_tokens:
        content = settings.tokenizer.decode(content_tokens, special_token_policy=SpecialTokenPolicy.IGNORE)
    else:
        content = None

    if tool_calls_tokens:
        try:
            tool_calls = decode_tool_call(tool_calls_tokens, settings.tokenizer.instruct_tokenizer.tokenizer)
        except InvalidtoolCallError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        tool_calls = None

    return AssistantMessage(content=content, tool_calls=tool_calls)


def create_app(tokenizer_path: Union[str, Path], validation_mode: ValidationMode) -> FastAPI:
    r"""Create a Mistral-common FastAPI app with the given tokenizer path and validation mode."""
    app = FastAPI()
    app.include_router(tokenize_router)
    app.include_router(decode_router)
    app.include_router(main_router)

    @lru_cache
    def get_settings_override() -> Settings:
        settings = Settings()
        settings._initialize_tokenizer(tokenizer_path, validation_mode)
        return settings

    app.dependency_overrides[get_settings] = get_settings_override

    return app


@click.command(context_settings={"auto_envvar_prefix": "UVICORN"})
@click.argument("tokenizer_path", type=str)
@click.argument(
    "validation_mode",
    type=click.Choice([mode.value for mode in ValidationMode], case_sensitive=False),
    default=ValidationMode.test.value,
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Mistral-common API host",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default=0,
    help="Mistral-common API port",
    show_default=True,
)
def serve_app(
    tokenizer_path: Union[str, Path], validation_mode: Union[ValidationMode, str], host: str, port: int
) -> None:
    r"""Serve the Mistral-common API with the given tokenizer path and validation mode."""
    app = create_app(tokenizer_path, ValidationMode(validation_mode))
    uvicorn.run(app, host=host, port=port)

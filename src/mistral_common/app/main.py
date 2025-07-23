import importlib.metadata
import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Union

import click
import uvicorn
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import BaseSettings

from mistral_common.protocol.instruct.messages import AssistantMessage, ChatMessageType, TextChunk, ThinkChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.think import split_content_and_think_chunks
from mistral_common.tokens.tokenizers.tools import decode_tool_calls, split_content_and_tool_calls


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
    as_message: bool = Body(default=False),
    special_token_policy: SpecialTokenPolicy = Body(default=SpecialTokenPolicy.IGNORE),
) -> Union[str, AssistantMessage]:
    r"""Detokenize a list of tokens.

    If `as_message` is `True`, the tokens are detokenized to an assistant message. It will parse tool calls from the
    tokens and extract content before the first tool call.
    Otherwise, the tokens are detokenized to a string.

    Args:
        tokens: The tokens to detokenize.
        as_message: Whether to detokenize to an assistant message.
        special_token_policy: The policy to use for special tokens.

    Returns:
        The detokenized string or assistant message.
    """
    if len(tokens) == 0:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")

    if as_message:
        if special_token_policy != SpecialTokenPolicy.IGNORE:
            raise HTTPException(
                status_code=400, detail="Special token policy must be IGNORE when detokenizing to message."
            )
        return detokenize_to_assistant_message(settings, tokens)

    try:
        return settings.tokenizer.decode(tokens, special_token_policy=special_token_policy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def detokenize_to_assistant_message(
    settings: Annotated[Settings, Depends(get_settings)],
    tokens: list[int] = Body(default_factory=list),
) -> AssistantMessage:
    r"""Detokenize a list of tokens to an assistant message.

    Parse tool calls from the tokens and extract content before the first tool call.

    Args:
        tokens: The tokens to detokenize.

    Returns:
        The detokenized assistant message.
    """
    if len(tokens) == 0:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")

    if settings.tokenizer.instruct_tokenizer.tokenizer.version > TokenizerVersion.v1:
        content_tokens, tool_calls_tokens = split_content_and_tool_calls(
            tokens, settings.tokenizer.instruct_tokenizer.tokenizer.get_control_token("[TOOL_CALLS]")
        )
    else:
        content_tokens, tool_calls_tokens = tokens, ()

    content: Optional[Union[str, List[Union[TextChunk, ThinkChunk]]]] = None

    if settings.tokenizer.instruct_tokenizer.tokenizer.version >= TokenizerVersion.v13:
        assert isinstance(settings.tokenizer.instruct_tokenizer, InstructTokenizerV13)

        begin_think = settings.tokenizer.instruct_tokenizer.BEGIN_THINK
        end_think = settings.tokenizer.instruct_tokenizer.END_THINK
    else:
        begin_think = end_think = None

    if begin_think is not None and end_think is not None:
        try:
            content_or_think_tokens = split_content_and_think_chunks(content_tokens, begin_think, end_think)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        eos = settings.tokenizer.instruct_tokenizer.tokenizer.eos_id

        if content_or_think_tokens:
            content = [
                TextChunk(text=settings.tokenizer.decode(chunk, special_token_policy=SpecialTokenPolicy.IGNORE))
                if not is_think
                else ThinkChunk(
                    thinking=settings.tokenizer.decode(chunk, special_token_policy=SpecialTokenPolicy.IGNORE),
                    closed=chunk[-1] == end_think,
                )
                for chunk, is_think in content_or_think_tokens
                if chunk != [eos]
            ]

    elif content_tokens:
        content = settings.tokenizer.decode(content_tokens, special_token_policy=SpecialTokenPolicy.IGNORE)

    if tool_calls_tokens:
        try:
            tool_calls = decode_tool_calls(tool_calls_tokens, settings.tokenizer.instruct_tokenizer.tokenizer)
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        tool_calls = None
    has_eos = tokens[-1] == settings.tokenizer.instruct_tokenizer.tokenizer.eos_id

    return AssistantMessage(content=content, tool_calls=tool_calls, prefix=not has_eos)


def create_app(
    tokenizer: Union[str, Path, MistralTokenizer], validation_mode: ValidationMode = ValidationMode.test
) -> FastAPI:
    r"""Create a Mistral-common FastAPI app with the given tokenizer and validation mode.

    Args:
        tokenizer: The tokenizer path or a MistralTokenizer instance.
        validation_mode: The validation mode to use.

    Returns:
        The Mistral-common FastAPI app.
    """
    if not isinstance(tokenizer, (MistralTokenizer, str, Path)):
        raise ValueError("Tokenizer must be a path or a MistralTokenizer instance.")

    app = FastAPI()
    app.include_router(tokenize_router)
    app.include_router(decode_router)
    app.include_router(main_router)

    @lru_cache
    def get_settings_override() -> Settings:
        settings = Settings()
        if isinstance(tokenizer, MistralTokenizer):
            settings.tokenizer = tokenizer
        else:
            settings._load_tokenizer(tokenizer, validation_mode)
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

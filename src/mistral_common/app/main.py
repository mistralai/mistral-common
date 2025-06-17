import importlib.metadata
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, overload

import click
import uvicorn
from fastapi import APIRouter, Body, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import BaseSettings

from mistral_common.protocol.instruct.messages import ChatMessageType
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


class OpenAIChatCompletionRequest(BaseModel):
    messages: List[dict[str, Union[str, List[dict[str, Union[str, dict[str, Any]]]]]]]
    tools: Optional[List[dict[str, Any]]] = None

    # Allow extra fields as the `from_openai` method will handle them.
    # We never validate the input, so we don't need to worry about the extra fields.
    model_config = ConfigDict(extra="allow")


class Settings(BaseSettings):
    app_name: str = "Mistral-common API"
    app_version: str = importlib.metadata.version("mistral-common")

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._tokenizer: Optional[MistralTokenizer] = None

    def _initialize_tokenizer(self, tokenizer_path: Union[str, Path]) -> None:
        if tokenizer_path == "":
            raise ValueError("Tokenizer path must be set via the environment variable `TOKENIZER_PATH`.")
        elif self._tokenizer is not None:
            raise ValueError("Tokenizer has already been initialized.")

        if isinstance(tokenizer_path, str):
            candidate_tokenizer_path = Path(tokenizer_path)
            if candidate_tokenizer_path.exists():
                tokenizer_path = candidate_tokenizer_path

        if isinstance(tokenizer_path, Path) and tokenizer_path.exists():
            self._tokenizer = MistralTokenizer.from_file(tokenizer_path)
        else:
            self._tokenizer = MistralTokenizer.from_hf_hub(str(tokenizer_path))

    @property
    def tokenizer(self) -> MistralTokenizer:
        if self._tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        return self._tokenizer


main_router = APIRouter(tags=["app"])
tokenize_router = APIRouter(prefix="/tokenizer/tokenize", tags=["tokenizer", "tokenize"])
template_router = APIRouter(prefix="/tokenizer/apply-template", tags=["tokenizer", "template"])
decode_router = APIRouter(prefix="/tokenizer/detokenize", tags=["tokenizer", "detokenize"])

settings = Settings()


@main_router.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="docs")


@main_router.get("/info")
def get_info() -> Dict[str, str]:
    return {"app_name": settings.app_name, "app_version": settings.app_version}


@overload
def _tokenize_request(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest], as_int: Literal[True]
) -> List[int]: ...
@overload
def _tokenize_request(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest], as_int: Literal[False]
) -> str: ...
@overload
def _tokenize_request(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest], as_int: bool
) -> Union[List[int], str]: ...
def _tokenize_request(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest], as_int: bool
) -> Union[List[int], str]:
    if isinstance(request, OpenAIChatCompletionRequest):
        try:
            request = ChatCompletionRequest.from_openai(**request.model_dump(exclude_none=True))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

    tokenized = settings.tokenizer.encode_chat_completion(request)
    assert isinstance(tokenized, Tokenized)
    if as_int:
        return tokenized.tokens
    else:
        assert isinstance(tokenized.text, str)
        return tokenized.text


@overload
def _tokenize_messages(messages: list[ChatMessageType], as_int: Literal[True]) -> List[int]: ...
@overload
def _tokenize_messages(messages: list[ChatMessageType], as_int: Literal[False]) -> str: ...
@overload
def _tokenize_messages(messages: list[ChatMessageType], as_int: bool) -> Union[List[int], str]: ...
def _tokenize_messages(messages: list[ChatMessageType], as_int: bool) -> Union[List[int], str]:
    if len(messages) == 0:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
    request = ChatCompletionRequest(messages=messages)

    return _tokenize_request(request, as_int)


@tokenize_router.post("/messages")
def tokenize_messages(messages: list[ChatMessageType] = Body(default_factory=list)) -> list[int]:
    return _tokenize_messages(messages, as_int=True)


@template_router.post("/messages")
def tokenize_messages_text(messages: list[ChatMessageType] = Body(default_factory=list)) -> str:
    return _tokenize_messages(messages, as_int=False)


@tokenize_router.post("/request")
def tokenize(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest],
) -> list[int]:
    return _tokenize_request(request, as_int=True)


@template_router.post("/request")
def tokenize_request_text(request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest]) -> str:
    return _tokenize_request(request, as_int=False)


@decode_router.post("/")
def detokenize_tokens(
    tokens: list[int] = Body(default_factory=list),
    special_token_policy: SpecialTokenPolicy = Body(default=SpecialTokenPolicy.IGNORE),
) -> str:
    if len(tokens) == 0:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")

    try:
        return settings.tokenizer.decode(tokens, special_token_policy=special_token_policy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@click.command(context_settings={"auto_envvar_prefix": "UVICORN"})
@click.argument("TOKENIZER_PATH", type=str)
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
def serve_app(tokenizer_path: Union[str, Path], host: str, port: int) -> None:
    settings._initialize_tokenizer(tokenizer_path)
    app = FastAPI()
    app.include_router(tokenize_router)
    app.include_router(decode_router)
    app.include_router(main_router)

    uvicorn.run(app, host=host, port=port)

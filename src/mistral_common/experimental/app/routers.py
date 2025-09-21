import json
from typing import Annotated, List, Optional, Union

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException 
from fastapi.responses import RedirectResponse
from pydantic import ValidationError

from mistral_common.experimental.app.models import (
    EngineBackend,
    OpenAIChatCompletionRequest,
    Settings,
    get_settings,
)
from mistral_common.experimental.think import _split_content_and_think_chunks
from mistral_common.experimental.tools import _decode_tool_calls, _split_content_and_tool_calls
from mistral_common.protocol.instruct.chunk import TextChunk, ThinkChunk
from mistral_common.protocol.instruct.messages import AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import InstructTokenizerV13

main_router = APIRouter(tags=["app"])
tokenize_router = APIRouter(prefix="/v1/tokenize", tags=["tokenizer", "tokenize"])
decode_router = APIRouter(prefix="/v1/detokenize", tags=["tokenizer", "detokenize"])


@main_router.get("/")
async def redirect_to_docs() -> RedirectResponse:
    """Redirect to the documentation."""
    return RedirectResponse(url="docs")


@tokenize_router.post("/")
async def tokenize_request(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest],
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[int]:
    """Tokenize a chat completion request."""
    if isinstance(request, OpenAIChatCompletionRequest):
        try:
            request.drop_extra_fields()
            request = ChatCompletionRequest.from_openai(**request.model_dump())
        except (ValidationError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    tokenized = settings.tokenizer.encode_chat_completion(request)
    assert isinstance(tokenized, Tokenized), type(tokenized)
    return tokenized.tokens


@decode_router.post("/string")
async def detokenize_to_string(
    settings: Annotated[Settings, Depends(get_settings)],
    tokens: list[int] = Body(default_factory=list),
    special_token_policy: SpecialTokenPolicy = Body(default=SpecialTokenPolicy.IGNORE),
) -> str:
    """Detokenize a list of tokens to a string."""
    if not tokens:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")
    try:
        return settings.tokenizer.decode(tokens, special_token_policy=special_token_policy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@decode_router.post("/")
async def detokenize_to_assistant_message(
    settings: Annotated[Settings, Depends(get_settings)],
    tokens: list[int] = Body(default_factory=list),
) -> AssistantMessage:
    """Detokenize a list of tokens to an assistant message."""
    if not tokens:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")

    if settings.tokenizer.instruct_tokenizer.tokenizer.version > TokenizerVersion.v1:
        content_tokens, tool_calls_tokens = _split_content_and_tool_calls(
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

    if begin_think and end_think:
        try:
            content_or_think_tokens = _split_content_and_think_chunks(content_tokens, begin_think, end_think)
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
            if len(content) == 1 and isinstance(content[0], TextChunk):
                content = content[0].text

    elif content_tokens:
        content = settings.tokenizer.decode(content_tokens, special_token_policy=SpecialTokenPolicy.IGNORE)

    if tool_calls_tokens:
        try:
            tool_calls = _decode_tool_calls(tool_calls_tokens, settings.tokenizer.instruct_tokenizer.tokenizer)
        except (ValueError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        tool_calls = None

    has_eos = tokens[-1] == settings.tokenizer.instruct_tokenizer.tokenizer.eos_id

    return AssistantMessage(content=content, tool_calls=tool_calls, prefix=not has_eos)


@main_router.post("/v1/chat/completions", tags=["chat", "completions"])
async def generate(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AssistantMessage:
    """Generate a chat completion."""
    if isinstance(request, OpenAIChatCompletionRequest):
        extra_fields = request.drop_extra_fields()
        request = ChatCompletionRequest.from_openai(**request.model_dump())
    else:
        extra_fields = {}

    tokens_ids = await tokenize_request(request, settings)

    exclude_fields = {"messages", "tools"}
    request_json = {k: v for k, v in {**request.model_dump(), **extra_fields}.items() if k not in exclude_fields}

    if request_json.get("stream", False):
        raise HTTPException(status_code=400, detail="Streaming is not supported.")

    if settings.engine_backend != EngineBackend.llama_cpp:
        raise HTTPException(status_code=400, detail=f"Unsupported engine backend: {settings.engine_backend}")

    try:
        async with httpx.AsyncClient(timeout=settings.timeout) as client:
            response = await client.post(
                f"{settings.engine_url}/completions",
                json={
                    "prompt": tokens_ids,
                    "return_tokens": True,
                    **request_json,
                },
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Engine request error: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    response_json = response.json()
    return await detokenize_to_assistant_message(settings, response_json["tokens"])

@main_router.get("/v1/models", tags=["models"])
async def get_models(
        settings: Annotated[Settings, Depends(get_settings)]
) -> dict:
    """
    Get list of models from the engine.
    """
    if settings.engine_backend != EngineBackend.llama_cpp:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported engine backend: {settings.engine_backend}"
        )

    try:
        async with httpx.AsyncClient(timeout=settings.timeout) as client:
            response = await client.get(f"{settings.engine_url}/models")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout from engine")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Engine request error: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()
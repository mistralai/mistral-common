import json
from typing import Annotated, List, Optional, Union

import httpx
import requests
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


async def _handle_llama_cpp_request(
    request_json: dict, 
    tokens_ids: list[int], 
    settings: Settings
) -> dict:
    """Handle Llama.cpp specific request formatting and execution."""
    try:
        response = requests.post(
            f"{settings.engine_url}/completion",
            json={
                "prompt": tokens_ids,
                "n_predict": request_json.get("max_tokens", -1),
                "temperature": request_json.get("temperature", 0.7),
                "top_p": request_json.get("top_p", 1.0),
                "top_k": request_json.get("top_k", 40),
                "repeat_penalty": request_json.get("frequency_penalty", 1.1),
                "seed": request_json.get("seed", -1),
                "stop": request_json.get("stop", []),
                "stream": False,
                "cache_prompt": True,
                "return_tokens": True,
                **{k: v for k, v in request_json.items() 
                   if k not in ["max_tokens", "temperature", "top_p", "top_k", 
                               "frequency_penalty", "seed", "stop"]}
            },
            timeout=settings.timeout,
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Timeout from Llama.cpp server")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Llama.cpp request error: {str(e)}")
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Llama.cpp error: {response.text}"
        )
    
    return response.json()


async def _handle_vllm_request(
    request_json: dict, 
    tokens_ids: list[int], 
    settings: Settings
) -> dict:
    """Handle vLLM specific request formatting and execution."""
    # Convert tokens back to text for vLLM (it expects text input)
    prompt_text = settings.tokenizer.decode(tokens_ids, special_token_policy=SpecialTokenPolicy.IGNORE)
    
    vllm_request = {
        "model": request_json.get("model", "default"),
        "prompt": prompt_text,
        "max_tokens": request_json.get("max_tokens", 512),
        "temperature": request_json.get("temperature", 0.7),
        "top_p": request_json.get("top_p", 1.0),
        "top_k": request_json.get("top_k", -1),
        "frequency_penalty": request_json.get("frequency_penalty", 0.0),
        "presence_penalty": request_json.get("presence_penalty", 0.0),
        "seed": request_json.get("seed"),
        "stop": request_json.get("stop", []),
        "stream": False,
        "echo": False,
        "logprobs": request_json.get("logprobs"),
        "top_logprobs": request_json.get("top_logprobs"),
        **{k: v for k, v in request_json.items() 
           if k not in ["model", "prompt", "max_tokens", "temperature", "top_p", 
                       "top_k", "frequency_penalty", "presence_penalty", "seed", 
                       "stop", "stream", "echo", "logprobs", "top_logprobs"]}
    }
    
    # Remove None values
    vllm_request = {k: v for k, v in vllm_request.items() if v is not None}
    
    try:
        async with httpx.AsyncClient(timeout=settings.timeout) as client:
            response = await client.post(
                f"{settings.engine_url}/v1/completions",
                json=vllm_request
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout from vLLM server")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"vLLM request error: {str(e)}")
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"vLLM error: {response.text}"
        )
    
    response_json = response.json()
    
    # Extract the generated text and convert to tokens for consistent processing
    if "choices" in response_json and len(response_json["choices"]) > 0:
        generated_text = response_json["choices"][0]["text"]
        # Tokenize the generated text to get tokens
        generated_tokens = settings.tokenizer.encode(generated_text).tokens
        return {"tokens": generated_tokens}
    else:
        raise HTTPException(status_code=500, detail="Invalid vLLM response format")


@main_router.post("/v1/chat/completions", tags=["chat", "completions"])
async def generate(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AssistantMessage:
    """Generate a chat completion with support for multiple backends.

    Args:
        request: The chat completion request.
        settings: The settings for the API.

    Returns:
        The generated chat completion.
        
    Supported backends:
        - llama_cpp: Uses Llama.cpp server
        - vllm: Uses vLLM server
    """
    if isinstance(request, OpenAIChatCompletionRequest):
        extra_fields = request.drop_extra_fields()
        request = ChatCompletionRequest.from_openai(**request.model_dump())
    else:
        extra_fields = {}
    
    tokens_ids = await tokenize_request(request, settings)

    exclude_fields = {"messages", "tools"}
    request_json = request.model_dump()
    request_json.update(extra_fields)
    request_json = {k: v for k, v in request_json.items() if k not in exclude_fields}

    if request_json.get("stream", False):
        raise HTTPException(status_code=400, detail="Streaming is not supported.")

    # Handle different engine backends
    if settings.engine_backend == EngineBackend.llama_cpp:
        response_json = await _handle_llama_cpp_request(request_json, tokens_ids, settings)
    elif settings.engine_backend == EngineBackend.vllm:
        response_json = await _handle_vllm_request(request_json, tokens_ids, settings)
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported engine backend: {settings.engine_backend}"
        )

    return await detokenize_to_assistant_message(settings, response_json["tokens"])


@main_router.get("/v1/models", tags=["models"])
async def get_models(
    settings: Annotated[Settings, Depends(get_settings)]
) -> dict:
    """Get list of models from the engine backend.
    
    Supports both Llama.cpp and vLLM model listing.
    """
    try:
        if settings.engine_backend == EngineBackend.llama_cpp:
            async with httpx.AsyncClient(timeout=settings.timeout) as client:
                response = await client.get(f"{settings.engine_url}/models")
        elif settings.engine_backend == EngineBackend.vllm:
            async with httpx.AsyncClient(timeout=settings.timeout) as client:
                response = await client.get(f"{settings.engine_url}/v1/models")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported engine backend: {settings.engine_backend}"
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout from engine")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Engine request error: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()


@main_router.get("/v1/engine/info", tags=["engine"])
async def get_engine_info(
    settings: Annotated[Settings, Depends(get_settings)]
) -> dict:
    """Get engine backend information and status."""
    try:
        if settings.engine_backend == EngineBackend.llama_cpp:
            async with httpx.AsyncClient(timeout=settings.timeout) as client:
                # Try to get props/stats from llama.cpp
                try:
                    props_response = await client.get(f"{settings.engine_url}/props")
                    props_data = props_response.json() if props_response.status_code == 200 else {}
                except:
                    props_data = {}
                
                return {
                    "backend": "llama_cpp",
                    "engine_url": settings.engine_url,
                    "status": "active",
                    "properties": props_data
                }
                
        elif settings.engine_backend == EngineBackend.vllm:
            async with httpx.AsyncClient(timeout=settings.timeout) as client:
                # Get vLLM version info
                try:
                    version_response = await client.get(f"{settings.engine_url}/version")
                    version_data = version_response.json() if version_response.status_code == 200 else {}
                except:
                    version_data = {}
                
                return {
                    "backend": "vllm",
                    "engine_url": settings.engine_url,
                    "status": "active",
                    "version": version_data
                }
        else:
            return {
                "backend": str(settings.engine_backend),
                "engine_url": settings.engine_url,
                "status": "unknown",
                "error": f"Unsupported engine backend: {settings.engine_backend}"
            }
            
    except httpx.TimeoutException:
        return {
            "backend": str(settings.engine_backend),
            "engine_url": settings.engine_url,
            "status": "timeout",
            "error": "Engine connection timeout"
        }
    except httpx.RequestError as e:
        return {
            "backend": str(settings.engine_backend),
            "engine_url": settings.engine_url,
            "status": "error",
            "error": f"Engine connection error: {str(e)}"
        }
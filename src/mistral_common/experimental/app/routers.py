import json
import time
import uuid
from typing import Annotated, List, Optional, Union, Dict, Any, AsyncGenerator

import httpx
import requests
from fastapi import APIRouter, Body, Depends, HTTPException 
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

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

# OpenAI-compatible response models
class ChatCompletionMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None

class ChatCompletionChoiceDelta(BaseModel):
    index: int
    delta: ChatCompletionMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChoiceDelta]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None

class TokenizeResponse(BaseModel):
    tokens: List[int]

class DetokenizeResponse(BaseModel):
    text: str

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]

main_router = APIRouter(tags=["app"])
tokenize_router = APIRouter(prefix="/v1/tokenize", tags=["tokenizer", "tokenize"])
decode_router = APIRouter(prefix="/v1/detokenize", tags=["tokenizer", "detokenize"])


@main_router.get("/")
async def redirect_to_docs() -> RedirectResponse:
    """Redirect to the documentation."""
    return RedirectResponse(url="docs")


@tokenize_router.post("/", response_model=TokenizeResponse)
async def tokenize_request(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenizeResponse:
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
    return TokenizeResponse(tokens=tokenized.tokens)


# Internal function for getting tokens as list (used by generate function)
async def _get_tokens_list(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest],
    settings: Settings,
) -> List[int]:
    """Internal function to get tokens as a list."""
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


@decode_router.post("/string", response_model=DetokenizeResponse)
async def detokenize_to_string(
    settings: Annotated[Settings, Depends(get_settings)],
    tokens: List[int] = Body(default_factory=list),
    special_token_policy: SpecialTokenPolicy = Body(default=SpecialTokenPolicy.IGNORE),
) -> DetokenizeResponse:
    """Detokenize a list of tokens to a string."""
    if not tokens:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")
    try:
        text = settings.tokenizer.decode(tokens, special_token_policy=special_token_policy)
        return DetokenizeResponse(text=text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@decode_router.post("/")
async def detokenize_to_assistant_message(
    settings: Annotated[Settings, Depends(get_settings)],
    tokens: List[int] = Body(default_factory=list),
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


def _create_openai_compatible_response(
    assistant_message: AssistantMessage,
    model_name: str,
    request_id: str,
    usage_stats: Optional[Dict[str, int]] = None
) -> ChatCompletionResponse:
    """Convert AssistantMessage to OpenAI-compatible response format."""
    
    # Handle content - convert chunks to string if needed
    content = None
    if isinstance(assistant_message.content, list):
        # Convert chunks to text
        text_parts = []
        for chunk in assistant_message.content:
            if isinstance(chunk, TextChunk):
                text_parts.append(chunk.text)
            elif isinstance(chunk, ThinkChunk):
                text_parts.append(chunk.thinking)
        content = "".join(text_parts)
    elif isinstance(assistant_message.content, str):
        content = assistant_message.content

    # Convert tool calls to OpenAI format if present
    tool_calls = None
    if assistant_message.tool_calls:
        tool_calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tool_call.get("name", ""),
                    "arguments": json.dumps(tool_call.get("arguments", {}))
                }
            }
            for i, tool_call in enumerate(assistant_message.tool_calls)
        ]

    # Determine finish reason
    finish_reason = "stop"
    if assistant_message.prefix:
        finish_reason = None
    elif tool_calls:
        finish_reason = "tool_calls"

    choice = ChatCompletionChoice(
        index=0,
        message=ChatCompletionMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls
        ),
        finish_reason=finish_reason
    )

    # Default usage if not provided
    if usage_stats is None:
        usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[choice],
        usage=ChatCompletionUsage(**usage_stats)
    )


def _create_stream_chunk(
    content: Optional[str],
    model_name: str,
    request_id: str,
    finish_reason: Optional[str] = None,
    usage_stats: Optional[Dict[str, int]] = None
) -> str:
    """Create a streaming chunk in OpenAI format."""
    
    chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionChoiceDelta(
                index=0,
                delta=ChatCompletionMessage(
                    role="assistant",
                    content=content
                ),
                finish_reason=finish_reason
            )
        ],
        usage=ChatCompletionUsage(**usage_stats) if usage_stats else None
    )
    
    return f"data: {chunk.model_dump_json()}\n\n"


async def _stream_llama_cpp_response(
    tokens_ids: List[int],
    request_json: Dict[str, Any],
    settings: Settings,
    model_name: str,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Stream response from llama.cpp backend."""
    
    # Add streaming parameters
    stream_request = {
        "prompt": tokens_ids,
        "stream": True,
        **request_json,
    }
    
    try:
        async with httpx.AsyncClient(timeout=settings.timeout) as client:
            async with client.stream(
                "POST",
                f"{settings.engine_url}/completions",
                json=stream_request
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=await response.aread())
                
                completion_tokens = 0
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            # Send final chunk with usage stats
                            usage_stats = {
                                "prompt_tokens": len(tokens_ids),
                                "completion_tokens": completion_tokens,
                                "total_tokens": len(tokens_ids) + completion_tokens
                            }
                            yield _create_stream_chunk(
                                content=None,
                                model_name=model_name,
                                request_id=request_id,
                                finish_reason="stop",
                                usage_stats=usage_stats
                            )
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            content = chunk_data.get("content", "")
                            
                            if content:
                                completion_tokens += 1
                                yield _create_stream_chunk(
                                    content=content,
                                    model_name=model_name,
                                    request_id=request_id
                                )
                                
                        except json.JSONDecodeError:
                            continue
                            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout from engine")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Engine request error: {str(e)}")


async def _stream_vllm_response(
    openai_request: Dict[str, Any],
    settings: Settings,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Stream response from vLLM backend."""
    
    try:
        async with httpx.AsyncClient(timeout=settings.timeout) as client:
            async with client.stream(
                "POST",
                f"{settings.engine_url}/v1/chat/completions",
                json=openai_request
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=await response.aread())
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            # vLLM already returns OpenAI-compatible format
                            # Just need to ensure the request_id matches if needed
                            chunk_data = json.loads(data_str)
                            chunk_data["id"] = request_id  # Ensure consistent request ID
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                            
                        except json.JSONDecodeError:
                            continue
                            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout from engine")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Engine request error: {str(e)}")


@main_router.post("/v1/chat/completions", tags=["chat", "completions"])
async def generate(
    request: Union[ChatCompletionRequest, OpenAIChatCompletionRequest],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """Generate a chat completion with OpenAI-compatible response format."""
    
    if isinstance(request, OpenAIChatCompletionRequest):
        extra_fields = request.drop_extra_fields()
        original_request = request
        request = ChatCompletionRequest.from_openai(**request.model_dump())
        model_name = original_request.model or "unknown"
    else:
        extra_fields = {}
        model_name = request.model or "unknown"

    # Generate request ID
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    exclude_fields = {"messages", "tools"}
    request_json = request.model_dump()
    request_json.update(extra_fields)
    request_json = {k: v for k, v in request_json.items() if k not in exclude_fields}

    # Check if streaming is requested
    is_streaming = request_json.get("stream", False)

    try:
        if settings.engine_backend == EngineBackend.llama_cpp:
            if is_streaming:
                # For streaming llama.cpp
                tokens_ids = await _get_tokens_list(request, settings)
                
                return StreamingResponse(
                    _stream_llama_cpp_response(
                        tokens_ids=tokens_ids,
                        request_json=request_json,
                        settings=settings,
                        model_name=model_name,
                        request_id=request_id
                    ),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream"
                    }
                )
            else:
                # Non-streaming llama.cpp (existing logic)
                tokens_ids = await _get_tokens_list(request, settings)
                
                response = requests.post(
                    f"{settings.engine_url}/completions",
                    json={
                        "prompt": tokens_ids,
                        "return_tokens": True,
                        **request_json,
                    },
                    timeout=settings.timeout,
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=response.text)

                response_json = response.json()
                
                # Get the generated tokens and convert to AssistantMessage
                generated_tokens = response_json.get("tokens", [])
                assistant_message = await detokenize_to_assistant_message(settings, generated_tokens)
                
                # Extract usage stats from llama.cpp response
                usage_stats = {
                    "prompt_tokens": response_json.get("tokens_evaluated", 0),
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": response_json.get("tokens_evaluated", 0) + len(generated_tokens)
                }
                
                return _create_openai_compatible_response(
                    assistant_message=assistant_message,
                    model_name=model_name,
                    request_id=request_id,
                    usage_stats=usage_stats
                )
            
        elif settings.engine_backend == EngineBackend.vllm:
            # Convert back to OpenAI format for vLLM
            openai_request = {
                "model": model_name,
                "messages": [msg.model_dump() for msg in request.messages],
                **request_json
            }
            
            if request.tools:
                openai_request["tools"] = [tool.model_dump() for tool in request.tools]
            
            if is_streaming:
                # For streaming vLLM
                return StreamingResponse(
                    _stream_vllm_response(
                        openai_request=openai_request,
                        settings=settings,
                        request_id=request_id
                    ),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/event-stream"
                    }
                )
            else:
                # Non-streaming vLLM
                async with httpx.AsyncClient(timeout=settings.timeout) as client:
                    response = await client.post(
                        f"{settings.engine_url}/v1/chat/completions",
                        json=openai_request
                    )
                    
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=response.text)
                
                # vLLM returns OpenAI-compatible format, so we can return it directly
                # but we need to ensure it matches our response model
                response_json = response.json()
                response_json["id"] = request_id  # Ensure consistent request ID
                return ChatCompletionResponse(**response_json)
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported engine backend: {settings.engine_backend}. Supported backends: llama_cpp, vllm"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Timeout from engine")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout from engine")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Engine request error: {str(e)}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Engine request error: {str(e)}")


@main_router.get("/v1/models", response_model=ModelsResponse, tags=["models"])
async def get_models(
    settings: Annotated[Settings, Depends(get_settings)]
) -> ModelsResponse:
    """Get list of models from the engine."""
    
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
                detail=f"Unsupported engine backend: {settings.engine_backend}. Supported backends: llama_cpp, vllm"
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout from engine")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Engine request error: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    response_json = response.json()
    
    # Ensure response follows OpenAI format
    if "data" not in response_json:
        # Convert non-OpenAI format to OpenAI format
        if isinstance(response_json, list):
            response_json = {"object": "list", "data": response_json}
        else:
            response_json = {"object": "list", "data": [response_json]}
    
    return ModelsResponse(**response_json)


# Add health check endpoint
@main_router.get("/health", tags=["health"])
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)]
) -> Dict[str, Any]:
    """Check the health of the API and backend engine."""
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if settings.engine_backend == EngineBackend.llama_cpp:
                response = await client.get(f"{settings.engine_url}/models")
            elif settings.engine_backend == EngineBackend.vllm:
                response = await client.get(f"{settings.engine_url}/v1/models")
            else:
                return {
                    "status": "error",
                    "backend": settings.engine_backend.value,
                    "message": f"Unsupported backend: {settings.engine_backend}"
                }
        
        engine_healthy = response.status_code == 200
        
        return {
            "status": "healthy" if engine_healthy else "degraded",
            "backend": settings.engine_backend.value,
            "engine_url": settings.engine_url,
            "engine_healthy": engine_healthy,
            "engine_status_code": response.status_code
        }
        
    except Exception as e:
        return {
            "status": "error",
            "backend": settings.engine_backend.value,
            "engine_url": settings.engine_url,
            "engine_healthy": False,
            "error": str(e)
        }
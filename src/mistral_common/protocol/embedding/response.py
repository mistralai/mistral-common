from typing import List

from pydantic import Field

from mistral_common.base import MistralBase
from mistral_common.protocol.base import UsageInfo
from mistral_common.protocol.utils import random_uuid


class EmbeddingObject(MistralBase):
    r"""Embedding object returned by the API.

    Attributes:
       object: The type of the object returned.
       embedding: The embedding vector.
       index: The index of the embedding in the input text.

    Examples:
        >>> embedding_object = EmbeddingObject(
        ...    object="embedding",
        ...    embedding=[0.1, 0.2, 0.3],
        ...    index=0
        ... )
    """

    object: str = Field(default="embedding", description="The type of the object returned.")
    embedding: List[float] = Field(description="The type of the object returned.")
    index: int = Field(description="The index of the embedding in the input text.")


class EmbeddingResponse(MistralBase):
    r""" "Embedding response returned by the API.

    See the [EmbeddingRequest][mistral_common.protocol.embedding.request.EmbeddingRequest] for the request body.

    Attributes:
        id: The ID of the embedding.
        object: The type of the object returned.
        data: List of embeddings.
        model: The model used to generate the embeddings.

    Examples:
        >>> response = EmbeddingResponse(
        ...    id="embd-123",
        ...    object="list",
        ...    data=[],
        ...    model="text-embedding-ada-002",
        ...    usage=UsageInfo(prompt_tokens=1, total_tokens=1, completion_tokens=0)
        ... )
    """

    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = Field(default="list", description="The type of the object returned.")
    data: List[EmbeddingObject] = Field(description="List of embeddings.")
    model: str = Field(description="The model used to generate the embeddings.")
    usage: UsageInfo

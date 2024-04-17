from typing import List

from mistral_common.base import MistralBase
from mistral_common.protocol.base import UsageInfo
from mistral_common.protocol.utils import random_uuid
from pydantic import Field


class EmbeddingObject(MistralBase):
    object: str = Field(default="embedding", description="The type of the object returned.")
    embedding: List[float] = Field(description="The type of the object returned.")
    index: int = Field(description="The index of the embedding in the input text.")


class EmbeddingResponse(MistralBase):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = Field(default="list", description="The type of the object returned.")
    data: List[EmbeddingObject] = Field(description="List of embeddings.")
    model: str = Field(description="The model used to generate the embeddings.")
    usage: UsageInfo

from typing import List, Optional, Union

from pydantic import Field

from mistral_common.base import MistralBase


class EmbeddingRequest(MistralBase):
    r""" "Embedding request model used to generate embeddings for the given input.

    See [EmbeddingResponse][mistral_common.protocol.embedding.response.EmbeddingResponse] for the response model.

    Attributes:
        input: Text to embed.
        model: ID of the model to use.
        encoding_format: The format to return the embeddings in.

    Examples:
        >>> request = EmbeddingRequest(input="Hello world!", model="mistral-embed")
    """

    input: Union[str, List[str]] = Field(description="Text to embed.")
    model: str = Field(description="ID of the model to use.")
    encoding_format: Optional[str] = Field(default="float", description="The format to return the embeddings in.")

from typing import Optional

from pydantic import Field
from pydantic_extra_types.language_code import LanguageAlpha2

from mistral_common.protocol.base import BaseCompletionRequest
from mistral_common.protocol.instruct.messages import AudioChunk


class TranscriptionRequest(BaseCompletionRequest):
    id: Optional[str] = None
    model: str
    audio: AudioChunk
    language: Optional[LanguageAlpha2] = Field(
        ...,
        description=(
            "The language of the input audio. Supplying the input language "
            "in ISO-639-1 format will improve accuracy and latency."
        ),
    )

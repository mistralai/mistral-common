from mistral_common.base import MistralBase
from mistral_common.protocol.base import UsageInfo


class TranscriptionSegment(MistralBase):
    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(MistralBase):
    text: str
    language: str
    duration: float
    segments: list[TranscriptionSegment]
    usage: UsageInfo

from typing import Optional

from mistral_common.base import MistralBase


class UsageInfo(MistralBase):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

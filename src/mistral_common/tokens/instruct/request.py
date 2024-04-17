from typing import List, Optional

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.messages import ChatMessage
from mistral_common.protocol.instruct.tool_calls import Tool


class InstructRequest(MistralBase):
    """
    A valid request to be tokenized
    """

    messages: List[ChatMessage]
    system_prompt: Optional[str] = None
    available_tools: Optional[List[Tool]] = None

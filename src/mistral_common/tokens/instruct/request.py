from typing import Generic, List, Optional

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.messages import ChatMessageType
from mistral_common.protocol.instruct.tool_calls import ToolType


class FIMRequest(MistralBase):
    """
    A valid Fill in the Middle completion request to be tokenized
    """

    prompt: str
    suffix: Optional[str] = None


class InstructRequest(MistralBase, Generic[ChatMessageType, ToolType]):
    """
    A valid request to be tokenized
    """

    messages: List[ChatMessageType]
    system_prompt: Optional[str] = None
    available_tools: Optional[List[ToolType]] = None

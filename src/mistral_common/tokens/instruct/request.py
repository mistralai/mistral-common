from typing import Generic, List, Optional

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.messages import ChatMessageType
from mistral_common.protocol.instruct.tool_calls import ToolType


class InstructRequest(MistralBase, Generic[ChatMessageType, ToolType]):
    """
    A valid request to be tokenized
    """

    messages: List[ChatMessageType]
    system_prompt: Optional[str] = None
    available_tools: Optional[List[ToolType]] = None

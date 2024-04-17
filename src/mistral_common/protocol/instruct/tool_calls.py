from enum import Enum
from typing import Any, Dict

from mistral_common.base import MistralBase


class Function(MistralBase):
    name: str
    description: str = ""
    parameters: Dict[str, Any]


class ToolType(str, Enum):
    function = "function"


class ToolChoice(str, Enum):
    auto: str = "auto"
    none: str = "none"
    any: str = "any"


class Tool(MistralBase):
    type: ToolType = ToolType.function
    function: Function


class FunctionCall(MistralBase):
    name: str
    arguments: str


class ToolCall(MistralBase):
    id: str = "null"  # required for V3 tokenization
    type: ToolType = ToolType.function
    function: FunctionCall

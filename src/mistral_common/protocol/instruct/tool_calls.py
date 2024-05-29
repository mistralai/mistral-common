import json
from enum import Enum
from typing import Any, Dict, TypeVar, Union

from pydantic import field_validator

from mistral_common.base import MistralBase


class Function(MistralBase):
    name: str
    description: str = ""
    parameters: Dict[str, Any]


class ToolTypes(str, Enum):
    function = "function"


class ToolChoice(str, Enum):
    auto: str = "auto"
    none: str = "none"
    any: str = "any"


class Tool(MistralBase):
    type: ToolTypes = ToolTypes.function
    function: Function


class FunctionCall(MistralBase):
    name: str
    arguments: str

    @field_validator("arguments", mode="before")
    def validate_arguments(cls, v: Union[str, Dict[str, Any]]) -> str:
        """
        This is for backward compatibility
        """
        if isinstance(v, dict):
            return json.dumps(v)
        return v


class ToolCall(MistralBase):
    id: str = "null"  # required for V3 tokenization
    type: ToolTypes = ToolTypes.function
    function: FunctionCall


ToolType = TypeVar("ToolType", bound=Tool)

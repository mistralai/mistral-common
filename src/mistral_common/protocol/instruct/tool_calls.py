import json
from enum import Enum
from typing import Any, Dict, TypeVar, Union

from pydantic import field_validator

from mistral_common.base import MistralBase


class Function(MistralBase):
    r"""Function definition for tools.

    Attributes:
        name: The name of the function.
        description: A description of what the function does.
        parameters: The parameters the functions accepts, described as a JSON Schema object.

    Examples:
        >>> function = Function(
        ...     name="get_current_weather",
        ...     description="Get the current weather in a given location",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "location": {
        ...                 "type": "string",
        ...                 "description": "The city and state, e.g. San Francisco, CA",
        ...             },
        ...             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        ...         },
        ...         "required": ["location"],
        ...     },
        ... )
    """

    name: str
    description: str = ""
    parameters: Dict[str, Any]


class ToolTypes(str, Enum):
    r"""Enum of tool types.

    Attributes:
       function: A function tool.

    Examples:
        >>> tool_type = ToolTypes.function
    """

    function = "function"


class ToolChoice(str, Enum):
    r"""Enum of tool choice types.

    Attributes:
        auto: Automatically choose the tool.
        none: Do not use any tools.
        any: Use any tool.

    Examples:
        >>> tool_choice = ToolChoice.auto
    """

    auto = "auto"
    none = "none"
    any = "any"


class Tool(MistralBase):
    r"""Tool definition.

    Attributes:
        type: The type of the tool.
        function: The function definition.

    Examples:
        >>> tool = Tool(
        ...     function=Function(
        ...         name="get_current_weather",
        ...         description="Get the current weather in a given location",
        ...         parameters={
        ...             "type": "object",
        ...             "properties": {
        ...                 "location": {
        ...                     "type": "string",
        ...                     "description": "The city and state, e.g. San Francisco, CA",
        ...                 },
        ...                 "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        ...             },
        ...             "required": ["location"],
        ...         },
        ...     ),
        ... )
    """

    type: ToolTypes = ToolTypes.function
    function: Function

    def to_openai(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_openai(cls, openai_tool: Dict[str, Any]) -> "Tool":
        return cls.model_validate(openai_tool)


class FunctionCall(MistralBase):
    r"""Function call.

    Attributes:
        name: The name of the function to call.
        arguments: The arguments to pass to the function.

    Examples:
        >>> function_call = FunctionCall(
        ...     name="get_current_weather",
        ...     arguments={"location": "San Francisco, CA", "unit": "celsius"},
        ... )
    """

    name: str
    arguments: str

    @field_validator("arguments", mode="before")
    def validate_arguments(cls, v: Union[str, Dict[str, Any]]) -> str:
        """Convert arguments to a JSON string if they are a dictionary.

        Args:
            v: The arguments to validate.

        Returns:
            The arguments as a JSON string.
        """
        if isinstance(v, dict):
            return json.dumps(v)
        return v


class ToolCall(MistralBase):
    r"""Tool call.

    Attributes:
        id: The ID of the tool call. Required for V3+ tokenization
        type: The type of the tool call.
        function: The function call.

    Examples:
        >>> tool_call = ToolCall(
        ...     id="call_abc123",
        ...     function=FunctionCall(
        ...         name="get_current_weather",
        ...         arguments={"location": "San Francisco, CA", "unit": "celsius"},
        ...     ),
        ... )
    """

    id: str = "null"
    type: ToolTypes = ToolTypes.function
    function: FunctionCall

    def to_openai(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_openai(cls, tool_call: Dict[str, Any]) -> "ToolCall":
        return cls.model_validate(tool_call)


ToolType = TypeVar("ToolType", bound=Tool)

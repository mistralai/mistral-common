import json
from enum import Enum
from typing import Any, TypeAlias, TypeVar

from pydantic import field_validator

from mistral_common.base import MistralBase


class FunctionName(MistralBase):
    r"""A function identified by name.

    Attributes:
        name: The name of the function.

    Examples:
        >>> function_name = FunctionName(name="get_current_weather")
    """

    name: str


class Function(FunctionName):
    r"""Function definition for tools.

    Attributes:
        name: The name of the function.
        description: A description of what the function does.
        parameters: The parameters the functions accepts, described as a JSON Schema object.
        strict: Whether to enforce strict function calling.

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

    description: str = ""
    parameters: dict[str, Any]
    strict: bool = False


class ToolTypes(str, Enum):
    r"""Enum of tool types.

    Attributes:
       function: A function tool.

    Examples:
        >>> tool_type = ToolTypes.function
    """

    function = "function"


class ToolChoiceEnum(str, Enum):
    r"""Enum of tool choice types.

    Attributes:
        auto: Automatically choose the tool.
        none: Do not use any tools.
        any: Deprecated in favor of `required`.
        required: Require the model to call at least one tool.

    Examples:
        >>> tool_choice = ToolChoiceEnum.auto
        >>> isinstance(tool_choice, ToolChoice)
        True
    """

    auto = "auto"
    none = "none"
    any = "any"  # deprecated in favor of `required`
    required = "required"


class NamedToolChoice(MistralBase):
    r"""Forces the model to call a specific function.

    Attributes:
        type: The type of the tool.
        function: The function the model should call.

    Examples:
        >>> named = NamedToolChoice(function=FunctionName(name="get_weather"))
        >>> isinstance(named, ToolChoice)
        True
    """

    type: ToolTypes = ToolTypes.function
    function: FunctionName


ToolChoice: TypeAlias = ToolChoiceEnum | NamedToolChoice
r"""Tool choice are either a `ToolChoiceEnum` or a `NamedToolChoice`."""


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

    def to_openai(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_openai(cls, openai_tool: dict[str, Any]) -> "Tool":
        return cls.model_validate_ignore_extra(openai_tool)


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
    def validate_arguments(cls, v: str | dict[str, Any]) -> str:
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

    def to_openai(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_openai(cls, tool_call: dict[str, Any]) -> "ToolCall":
        return cls.model_validate_ignore_extra(tool_call)


ToolType = TypeVar("ToolType", bound=Tool)

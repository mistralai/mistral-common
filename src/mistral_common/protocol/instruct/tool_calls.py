import json
from enum import Enum
from typing import Any, TypeAlias, TypeVar

from pydantic import field_validator

from mistral_common.base import MistralBase


class FunctionName(MistralBase):
    r"""Identifier for a function by its name.

    Used to reference a function without its full definition.

    Attributes:
        name: The function name string. Must be a valid identifier.

    Examples:
        >>> function_name = FunctionName(name="get_current_weather")
    """

    name: str


class Function(FunctionName):
    r"""Complete function definition for a tool.

    Defines a callable function with its schema and metadata. Used within Tool
    definitions to specify what functions are available.

    Attributes:
        name: Function name. Used by the model to identify which function to call.
        description: Human-readable description of the function's purpose.
            Displayed to help users understand when to use this function.
        parameters: JSON Schema object defining the function's parameters.
            Used for validation and to generate the function call arguments.
            Must have "type": "object" and define "properties" and optionally "required".
        strict: If `True`, the function's parameters schema is enforced: the model's
            arguments must conform to it. If `False` (default), the parameters schema
            is not enforced and the model may emit any JSON object as arguments.

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

    @classmethod
    def from_openai(cls, openai_function: dict[str, Any]) -> "Function":
        r"""Convert an OpenAI function definition to a Mistral Function.

        Handles conversion from OpenAI's format, filtering out unknown fields
        and providing defaults for missing required fields.

        Args:
            openai_function: Dictionary matching OpenAI's function schema.
                Must contain at least a "name" key.

        Returns:
            Function instance with parameters and description defaulted to
            empty dict and empty string respectively if not provided.
        """
        filtered = cls._filter_cls_fields(openai_function)
        if filtered.get("parameters") is None:
            filtered["parameters"] = {}
        if filtered.get("description") is None:
            filtered["description"] = ""
        return cls.model_validate(filtered)


class ToolTypes(str, Enum):
    r"""Enum of supported tool types.

    Attributes:
        function: A function tool that can be called with arguments.

    Examples:
        >>> tool_type = ToolTypes.function
    """

    function = "function"


class ToolChoiceEnum(str, Enum):
    r"""Enum controlling how the model selects tools.

    Attributes:
        auto: Model automatically decides whether to call tools.
        none: Model will not call any tools.
        any: Deprecated; use `required` instead.
        required: Model must call at least one available tool.

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

    Use this to constrain the model to use a particular function rather than
    letting it choose from available tools.

    Attributes:
        type: The tool type. Must be ToolTypes.function.
        function: The FunctionName identifying which specific function to call.
            The model will only use this function, regardless of what tools are available.

    Examples:
        >>> named = NamedToolChoice(function=FunctionName(name="get_weather"))
        >>> isinstance(named, ToolChoice)
        True
    """

    type: ToolTypes = ToolTypes.function
    function: FunctionName


ToolChoice: TypeAlias = ToolChoiceEnum | NamedToolChoice
r"""Tool choice can be either a ToolChoiceEnum value or a NamedToolChoice instance."""


class Tool(MistralBase):
    r"""Definition of a tool available to the model.

    A tool combines a function definition with metadata about how it should
    be presented and used.

    Attributes:
        type: The tool type. Must be ToolTypes.function.
        function: The Function definition specifying the callable function,
            its parameters, and description.

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
        r"""Convert this tool to OpenAI format.

        Returns:
            Dictionary matching OpenAI's tool schema.
        """
        return self.model_dump()

    @classmethod
    def from_openai(cls, openai_tool: dict[str, Any]) -> "Tool":
        r"""Create a Tool from an OpenAI tool definition.

        Converts OpenAI's tool format to Mistral's format, delegating function
        parsing to Function.`from_openai`.

        Args:
            openai_tool: Dictionary matching OpenAI's tool schema. Must have
                "type" and optionally "function" keys.

        Returns:
            Tool instance with the function converted from OpenAI format.
        """
        openai_tool = openai_tool.copy()
        if function := openai_tool.get("function"):
            openai_tool["function"] = Function.from_openai(function)
        return cls.model_validate_ignore_extra(openai_tool)


class FunctionCall(MistralBase):
    r"""Represents a function call made by the model.

    Contains the function name and its arguments as they would be passed
    to the actual function.

    Attributes:
        name: The name of the function being called.
        arguments: JSON string of the arguments to pass to the function.
            Stored as a string but parsed from/to dict for convenience.

    Examples:
        >>> function_call = FunctionCall(
        ...     name="get_current_weather",
        ...     arguments={"location": "San Francisco, CA", "unit": "celsius"},
        ... )
    """

    name: str
    arguments: str

    @field_validator("arguments", mode="before")
    def validate_arguments(cls, v: str | dict[str, Any] | None) -> str:
        r"""Convert arguments to a JSON string if they are a dictionary.

        Args:
            v: The arguments value. Can be a dict (converted to JSON string),
                str (used as-is), or `None` (converted to "{}").

        Returns:
            The arguments as a JSON string.
        """
        if isinstance(v, dict):
            return json.dumps(v)
        elif v is None:
            return "{}"
        return v


class ToolCall(MistralBase):
    r"""Represents a tool call made by the model during generation.

    A tool call wraps a FunctionCall with metadata including a unique ID.

    Attributes:
        id: Unique identifier for this tool call. Must be a non-empty string
            for tokenizer version >= v13. Defaults to "null" for backwards compatibility.
        type: The tool type. Must be ToolTypes.function.
        function: The FunctionCall containing the function name and arguments.

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
        r"""Convert this tool call to OpenAI format.

        Returns:
            Dictionary matching OpenAI's tool call schema.
        """
        return self.model_dump()

    @classmethod
    def from_openai(cls, tool_call: dict[str, Any]) -> "ToolCall":
        r"""Create a ToolCall from an OpenAI tool call definition.

        Args:
            tool_call: Dictionary matching OpenAI's tool call schema.

        Returns:
            ToolCall instance with fields parsed from the OpenAI format.
        """
        return cls.model_validate_ignore_extra(tool_call)


ToolType = TypeVar("ToolType", bound=Tool)

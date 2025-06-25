from typing import Any, Dict, Generic, List, Optional, Union

from mistral_common.base import MistralBase
from mistral_common.protocol.instruct.converters import (
    _check_openai_fields_names,
    _is_openai_field_name,
    convert_openai_messages,
    convert_openai_tools,
)
from mistral_common.protocol.instruct.messages import ChatMessage, ChatMessageType
from mistral_common.protocol.instruct.tool_calls import ToolType


class FIMRequest(MistralBase):
    r"""A valid Fill in the Middle completion request to be tokenized.

    Attributes:
        prompt: The prompt to be completed.
        suffix: The suffix of the prompt. If provided, the model will generate text between the prompt and the suffix.

    Examples:
        >>> request = FIMRequest(prompt="Hello, my name is", suffix=" and I live in New York.")
    """

    prompt: str
    suffix: Optional[str] = None


class InstructRequest(MistralBase, Generic[ChatMessageType, ToolType]):
    """A valid Instruct request to be tokenized.

    Attributes:
        messages: The history of the conversation.
        system_prompt: The system prompt to be used for the conversation.
        available_tools: The tools available to the assistant.
        truncate_at_max_tokens: The maximum number of tokens to truncate the conversation at.
        continue_final_message: Whether to continue the final message.

    Examples:
        >>> from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
        >>> request = InstructRequest(
        ...     messages=[UserMessage(content="Hello, how are you?")], system_prompt="You are a helpful assistant."
        ... )
    """

    messages: List[ChatMessageType]
    system_prompt: Optional[str] = None
    available_tools: Optional[List[ToolType]] = None
    truncate_at_max_tokens: Optional[int] = None
    continue_final_message: bool = False

    def to_openai(self, **kwargs: Any) -> Dict[str, List[Dict[str, Any]]]:
        r"""Convert the request messages and tools into the OpenAI format.

        Args:
            kwargs: Additional parameters to be added to the request.

        Returns:
            The request in the OpenAI format.

        Examples:
            >>> from mistral_common.protocol.instruct.messages import UserMessage
            >>> from mistral_common.protocol.instruct.tool_calls import Tool, Function
            >>> request = InstructRequest(messages=[UserMessage(content="Hello, how are you?")])
            >>> request.to_openai(temperature=0.15, stream=True)
            {'continue_final_message': False, 'messages': [{'role': 'user', 'content': 'Hello, how are you?'}], 'temperature': 0.15, 'stream': True}
            >>> request = InstructRequest(
            ...     messages=[UserMessage(content="Hello, how are you?")],
            ...     available_tools=[
            ...     Tool(function=Function(
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
            ... )])
            >>> request.to_openai()
            {'continue_final_message': False, 'messages': [{'role': 'user', 'content': 'Hello, how are you?'}], 'tools': [{'type': 'function', 'function': {'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['location']}}}]}
        """  # noqa: E501

        # Handle messages, tools, and truncate_at_max_tokens separately.
        openai_request: Dict[str, Any] = self.model_dump(
            exclude={"messages", "available_tools", "truncate_at_max_tokens"}, exclude_none=True
        )

        for kwarg in kwargs:
            # Check for duplicate keyword arguments.
            if kwarg in openai_request:
                raise ValueError(f"Duplicate keyword argument: {kwarg}")
            # Check if kwarg should have been set in the request.
            # This occurs when the field is different between the Mistral and OpenAI API.
            elif kwarg in InstructRequest.model_fields:
                raise ValueError(f"Keyword argument {kwarg} is already set in the request.")
            # Check if the keyword argument is a valid OpenAI field name.
            elif not _is_openai_field_name(kwarg):
                raise ValueError(f"Invalid keyword argument: {kwarg}, it should be an OpenAI field name.")

        openai_messages: list[dict[str, Any]] = []
        if self.system_prompt is not None:
            openai_messages.append({"role": "system", "content": self.system_prompt})

        for message in self.messages:
            openai_messages.append(message.to_openai())

        openai_request["messages"] = openai_messages
        if self.available_tools is not None:
            # Rename available_tools to tools
            openai_request["tools"] = [tool.to_openai() for tool in self.available_tools]

        if self.truncate_at_max_tokens is not None:  # Rename to max_tokens
            raise NotImplementedError("Truncating at max tokens is not implemented for OpenAI requests.")

        openai_request.update(kwargs)

        return openai_request

    @classmethod
    def from_openai(
        cls,
        messages: List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, Any]]]]]]],
        tools: Optional[List[Dict[str, Any]]] = None,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> "InstructRequest":
        r"""Create an instruct request from the OpenAI format.

        Args:
            messages: The messages in the OpenAI format.
            tools: The tools in the OpenAI format.
            continue_final_message: Whether to continue the final message.
            **kwargs: Additional keyword arguments to pass to the constructor. These should be the same as the fields
                of the request class or the OpenAI API equivalent.

        Returns:
            The instruct request.
        """
        # Handle the case where the tools are passed as `available_tools`.
        # This is to maintain compatibility with the OpenAI API.
        if "available_tools" in kwargs:
            if tools is None:
                tools = kwargs.pop("available_tools")
            else:
                raise ValueError("Cannot specify both `tools` and `available_tools`.")

        _check_openai_fields_names(set(cls.model_fields.keys()), set(kwargs.keys()))

        converted_messages: list[ChatMessage] = convert_openai_messages(messages)

        converted_tools = convert_openai_tools(tools) if tools is not None else None

        return cls(
            messages=converted_messages,  # type: ignore[arg-type]
            available_tools=converted_tools,  # type: ignore[arg-type]
            continue_final_message=continue_final_message,
            **kwargs,
        )

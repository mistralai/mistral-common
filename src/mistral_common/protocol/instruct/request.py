from enum import Enum
from typing import Any, Generic

from pydantic import Field

from mistral_common.base import MistralBase
from mistral_common.protocol.base import BaseCompletionRequest
from mistral_common.protocol.instruct.converters import (
    _check_openai_fields_names,
    _is_openai_field_name,
    convert_openai_messages,
    convert_openai_tools,
)
from mistral_common.protocol.instruct.messages import (
    ChatMessage,
    ChatMessageType,
)
from mistral_common.protocol.instruct.tool_calls import Tool, ToolChoice, ToolType


class ResponseFormats(str, Enum):
    r"""Enum of the different formats of an instruct response.

    Attributes:
        text: The response is a plain text.
        json: The response is a JSON object.

    Examples:
        >>> response_format = ResponseFormats.text
    """

    text = "text"
    json = "json_object"


class ResponseFormat(MistralBase):
    r"""The format of the response.

    Attributes:
        type: The type of the response.

    Examples:
        >>> response_format = ResponseFormat(type=ResponseFormats.text)
    """

    type: ResponseFormats = ResponseFormats.text


class ChatCompletionRequest(BaseCompletionRequest, Generic[ChatMessageType]):
    r"""Request for a chat completion.

    Attributes:
        model: The model to use for the chat completion.
        messages: The messages to use for the chat completion.
        response_format: The format of the response.
        tools: The tools to use for the chat completion.
        tool_choice: The tool choice to use for the chat completion.
        truncate_for_context_length: Whether to truncate the messages for the context length.
        continue_final_message: Whether to continue the final message.

    Examples:
        >>> from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
        >>> from mistral_common.protocol.instruct.tool_calls import ToolTypes, Function
        >>> request = ChatCompletionRequest(
        ...     messages=[
        ...         UserMessage(content="Hello!"),
        ...         AssistantMessage(content="Hi! How can I help you?"),
        ...     ],
        ...     response_format=ResponseFormat(type=ResponseFormats.text),
        ...     tools=[Tool(type=ToolTypes.function, function=Function(name="get_weather", parameters={}))],
        ...     tool_choice=ToolChoice.auto,
        ...     truncate_for_context_length=True,
        ... )
    """

    model: str | None = None
    messages: list[ChatMessageType]
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    tools: list[Tool] | None = None
    tool_choice: ToolChoice = ToolChoice.auto
    truncate_for_context_length: bool = False
    continue_final_message: bool = False

    def to_openai(self, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        r"""Convert the request messages and tools into the OpenAI format.

        Args:
            kwargs: Additional parameters to be added to the request.

        Returns:
            The request in the OpenAI format.

        Examples:
            >>> from mistral_common.protocol.instruct.messages import UserMessage
            >>> from mistral_common.protocol.instruct.tool_calls import Tool, Function
            >>> request = ChatCompletionRequest(messages=[UserMessage(content="Hello, how are you?")], temperature=0.15)
            >>> request.to_openai(stream=True)
            {'temperature': 0.15, 'top_p': 1.0, 'response_format': {'type': 'text'}, 'tool_choice': 'auto', 'continue_final_message': False, 'messages': [{'role': 'user', 'content': 'Hello, how are you?'}], 'stream': True}
            >>> request = ChatCompletionRequest(messages=[UserMessage(content="Hello, how are you?")], tools=[
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
            {'temperature': 0.7, 'top_p': 1.0, 'response_format': {'type': 'text'}, 'tool_choice': 'auto', 'continue_final_message': False, 'messages': [{'role': 'user', 'content': 'Hello, how are you?'}], 'tools': [{'type': 'function', 'function': {'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['location']}}}]}
        """  # noqa: E501

        # Handle messages and tools separately.
        openai_request: dict[str, Any] = self.model_dump(
            exclude={"messages", "tools", "truncate_for_context_length"}, exclude_none=True
        )

        # Rename random_seed to seed.
        seed = openai_request.pop("random_seed", None)
        if seed is not None:
            openai_request["seed"] = seed

        if self.truncate_for_context_length:
            raise NotImplementedError("Truncating for context length is not implemented for OpenAI requests.")

        for kwarg in kwargs:
            # Check for duplicate keyword arguments.
            if kwarg in openai_request:
                raise ValueError(f"Duplicate keyword argument: {kwarg}")
            # Check if kwarg should have been set in the request.
            # This occurs when the field is different between the Mistral and OpenAI API.
            elif kwarg in ChatCompletionRequest.model_fields:
                raise ValueError(f"Keyword argument {kwarg} is already set in the request.")
            # Check if kwarg is a valid OpenAI field name.
            elif not _is_openai_field_name(kwarg):
                raise ValueError(f"Invalid keyword argument: {kwarg}, it should be an OpenAI field name.")

        openai_messages = []
        for message in self.messages:
            openai_messages.append(message.to_openai())

        openai_request["messages"] = openai_messages
        if self.tools is not None:
            openai_request["tools"] = [tool.to_openai() for tool in self.tools]

        openai_request.update(kwargs)

        return openai_request

    @classmethod
    def from_openai(
        cls,
        messages: list[dict[str, str | list[dict[str, str | dict[str, Any]]]]],
        tools: list[dict[str, Any]] | None = None,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> "ChatCompletionRequest":
        r"""Create a chat completion request from the OpenAI format.

        Args:
            messages: The messages in the OpenAI format.
            tools: The tools in the OpenAI format.
            continue_final_message: Whether to continue the final message.
            **kwargs: Additional keyword arguments to pass to the constructor. These should be the same as the fields
                of the request class or the OpenAI API equivalent.


        Returns:
            The chat completion request.
        """
        if "seed" in kwargs and "random_seed" in kwargs:
            raise ValueError("Cannot specify both `seed` and `random_seed`.")

        random_seed = kwargs.pop("seed", None) or kwargs.pop("random_seed", None)

        _check_openai_fields_names(set(cls.model_fields.keys()), set(kwargs.keys()))

        converted_messages: list[ChatMessage] = convert_openai_messages(messages)

        converted_tools = convert_openai_tools(tools) if tools is not None else None

        return cls(
            messages=converted_messages,  # type: ignore[arg-type]
            tools=converted_tools,
            random_seed=random_seed,
            continue_final_message=continue_final_message,
            **kwargs,
        )


class InstructRequest(MistralBase, Generic[ChatMessageType, ToolType]):
    r"""A valid Instruct request to be tokenized.

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

    messages: list[ChatMessageType]
    system_prompt: str | None = None
    available_tools: list[ToolType] | None = None
    truncate_at_max_tokens: int | None = None
    continue_final_message: bool = False

    def to_openai(self, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
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
        openai_request: dict[str, Any] = self.model_dump(
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

        if self.truncate_at_max_tokens is not None:
            raise NotImplementedError("Truncating at max tokens is not implemented for OpenAI requests.")

        openai_request.update(kwargs)

        return openai_request

    @classmethod
    def from_openai(
        cls,
        messages: list[dict[str, str | list[dict[str, str | dict[str, Any]]]]],
        tools: list[dict[str, Any]] | None = None,
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

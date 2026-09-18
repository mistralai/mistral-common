from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any, Generic

from pydantic import Field, TypeAdapter, model_validator

from mistral_common.base import MistralBase
from mistral_common.deprecation import warn_once
from mistral_common.exceptions import InvalidMessageStructureException
from mistral_common.protocol.base import BaseCompletionRequest
from mistral_common.protocol.instruct.converters import (
    convert_openai_messages,
    convert_openai_tools,
)
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ChatMessage,
    ChatMessageType,
    ReasoningFieldFormat,
)
from mistral_common.protocol.instruct.tool_calls import Tool, ToolChoice, ToolChoiceEnum, ToolType

_CONTINUE_FINAL_MESSAGE_KEY = "continue_final_message"
_CONTINUE_FINAL_MESSAGE_ERROR = "continue_final_message=True requires final message to be an assistant."


def _map_continue_final_message(
    messages: Iterable[dict[str, Any] | ChatMessageType], continue_final_message: bool
) -> list[Any]:
    r"""Copy messages and apply continuation to the final assistant message."""
    copied_messages = list(messages)
    if not continue_final_message:
        return copied_messages

    if not copied_messages:
        raise InvalidMessageStructureException(_CONTINUE_FINAL_MESSAGE_ERROR)

    if isinstance(copied_messages[-1], dict):
        if copied_messages[-1].get("role") != "assistant":
            raise InvalidMessageStructureException(_CONTINUE_FINAL_MESSAGE_ERROR)
        if "prefix" in copied_messages[-1]:
            TypeAdapter(bool).validate_python(copied_messages[-1]["prefix"])
        copied_messages[-1] = {**copied_messages[-1], "prefix": True}
    elif isinstance(copied_messages[-1], AssistantMessage):
        copied_messages[-1] = copied_messages[-1].model_copy(update={"prefix": True})
    else:
        raise InvalidMessageStructureException(_CONTINUE_FINAL_MESSAGE_ERROR)

    return copied_messages


class ResponseFormats(str, Enum):
    r"""Enum of the different formats for an instruct response.

    Attributes:
        text: Response will be plain text.
        json: Response will be a valid JSON object.

    Examples:
        >>> response_format = ResponseFormats.text
    """

    text = "text"
    json = "json_object"


class ReasoningEffort(str, Enum):
    r"""Controls the amount of reasoning effort the model applies during generation.

    Attributes:
        none: No reasoning effort; model generates responses directly.
        high: High reasoning effort for complex tasks; model spends more compute
            on reasoning before generating the final response.

    Note:
        Supported for tokenizer >= v15 only. Earlier versions will raise an error.
    """

    none = "none"
    high = "high"


class ModelSettings(MistralBase):
    r"""Model configuration settings for instruct requests.

    Encapsulates model-specific settings that influence inference behavior.
    Currently supports reasoning effort configuration.

    Attributes:
        reasoning_effort: Controls reasoning effort. If `None` (default), the model
            uses its default reasoning behavior. Requires tokenizer >= v15.
    """

    reasoning_effort: ReasoningEffort | None = None

    @staticmethod
    def none() -> "ModelSettings":
        r"""Create a ModelSettings instance with all fields set to `None`.

        Returns:
            ModelSettings with `reasoning_effort=None`.
        """
        return ModelSettings()


class ResponseFormat(MistralBase):
    r"""Configuration for the response format.

    Attributes:
        type: The response format type. Use `ResponseFormats.text` for plain text
            or `ResponseFormats.json` for JSON output.

    Examples:
        >>> response_format = ResponseFormat(type=ResponseFormats.text)
    """

    type: ResponseFormats = ResponseFormats.text


class ChatCompletionRequest(BaseCompletionRequest, Generic[ChatMessageType]):
    r"""Request for a chat completion.

    Main entry point for creating instruct model requests. Contains all configuration
    for a single chat completion including messages, tools, and model settings.

    Attributes:
        model: Name of the model to use. Required in serving mode; optional in
            other modes. If `None`, the default model will be used.
        messages: List of chat messages (user, assistant, system, tool). Must not be empty.
        response_format: Format of the response (text or JSON). Defaults to text.
        tools: List of available tools for the model to use. If `None`, no tools are available.
        tool_choice: Strategy for tool selection. Options: auto (model decides),
            none (no tools), any/required (deprecated, use required). Default: auto.
        truncate_for_context_length: If `True`, automatically truncate messages to
            fit within the model's context length. Default: `False`.
        reasoning_effort: Controls reasoning effort (none or high). Requires tokenizer
            >= v15. If `None`, uses model default.

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
        ...     tool_choice=ToolChoiceEnum.auto,
        ...     truncate_for_context_length=True,
        ... )
    """

    model: str | None = None
    messages: list[ChatMessageType]
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    tools: list[Tool] | None = None
    tool_choice: ToolChoice = ToolChoiceEnum.auto
    truncate_for_context_length: bool = False
    reasoning_effort: ReasoningEffort | None = None

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_continue_final_message(cls, values: Any) -> Any:
        r"""Translate legacy `continue_final_message` parameter into `AssistantMessage.prefix`.

        This validator handles the deprecated `continue_final_message` parameter by
        converting it to the new `AssistantMessage.prefix` format.

        Args:
            values: The raw input values being validated.

        Returns:
            The values with `continue_final_message` translated to message prefix.
        """
        if not isinstance(values, dict) or _CONTINUE_FINAL_MESSAGE_KEY not in values:
            return values

        copied_values = dict(values)
        continue_final_message = TypeAdapter(bool).validate_python(copied_values.pop(_CONTINUE_FINAL_MESSAGE_KEY))
        warn_once(
            key="ChatCompletionRequest.continue_final_message",
            message=(
                "`continue_final_message` passed directly to ChatCompletionRequest is deprecated. "
                "Set `AssistantMessage.prefix` instead. Will be removed in 1.13.0."
            ),
            category=DeprecationWarning,
            stacklevel=4,
        )
        messages = copied_values.get("messages")
        if isinstance(messages, Iterable) and not isinstance(messages, (str, bytes, Mapping)):
            copied_values["messages"] = _map_continue_final_message(
                messages=messages,
                continue_final_message=continue_final_message,
            )
        elif continue_final_message:
            raise InvalidMessageStructureException(_CONTINUE_FINAL_MESSAGE_ERROR)
        return copied_values

    def to_openai(
        self,
        reasoning_field_format: ReasoningFieldFormat | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        r"""Convert this request to OpenAI ChatCompletion API format.

        Args:
            reasoning_field_format: Format for converting thinking/thinking chunks in
                assistant messages to OpenAI's reasoning field. If `None`, uses default.
                See `AssistantMessage.to_openai` for available formats.
            **kwargs: Additional OpenAI-specific parameters to include in the output
                (e.g., stream, temperature, `top_p`). Must not conflict with existing fields.

        Returns:
            Dictionary matching the OpenAI ChatCompletion request schema.

        Raises:
            ValueError: If kwargs contains duplicate or conflicting keys.
            NotImplementedError: If `truncate_for_context_length` is `True` (not implemented).

        Examples:
            >>> from mistral_common.protocol.instruct.messages import UserMessage
            >>> from mistral_common.protocol.instruct.tool_calls import Tool, Function
            >>> request = ChatCompletionRequest(messages=[UserMessage(content="Hello, how are you?")], temperature=0.15)
            >>> request.to_openai(stream=True)
            {'temperature': 0.15, 'top_p': 1.0, 'response_format': {'type': 'text'}, 'continue_final_message': False, 'messages': [{'role': 'user', 'content': 'Hello, how are you?'}], 'tool_choice': 'auto', 'stream': True}
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
            {'temperature': 0.7, 'top_p': 1.0, 'response_format': {'type': 'text'}, 'continue_final_message': False, 'messages': [{'role': 'user', 'content': 'Hello, how are you?'}], 'tools': [{'type': 'function', 'function': {'name': 'get_current_weather', 'description': 'Get the current weather in a given location', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}, 'required': ['location']}, 'strict': False}}], 'tool_choice': 'auto'}
        """  # noqa: E501

        # Handle messages and tools separately.
        openai_request: dict[str, Any] = self.model_dump(
            exclude={"messages", "tools", "truncate_for_context_length", "tool_choice"}, exclude_none=True
        )

        # Rename random_seed to seed.
        seed = openai_request.pop("random_seed", None)
        if seed is not None:
            openai_request["seed"] = seed

        reasoning_effort = openai_request.pop("reasoning_effort", None)
        openai_request["continue_final_message"] = bool(
            self.messages and isinstance(self.messages[-1], AssistantMessage) and self.messages[-1].prefix
        )
        if reasoning_effort is not None:
            openai_request["reasoning_effort"] = reasoning_effort

        if self.truncate_for_context_length:
            raise NotImplementedError("Truncating for context length is not implemented for OpenAI requests.")

        for kwarg in kwargs:
            if kwarg in openai_request:
                raise ValueError(f"Duplicate keyword argument: {kwarg}")
            elif kwarg in ChatCompletionRequest.model_fields:
                raise ValueError(f"Keyword argument {kwarg} is already set in the request.")

        openai_messages = []
        for message in self.messages:
            if isinstance(message, AssistantMessage):
                openai_messages.append(message.to_openai(reasoning_field_format=reasoning_field_format))
            else:
                openai_messages.append(message.to_openai())

        openai_request["messages"] = openai_messages
        if self.tools is not None:
            openai_request["tools"] = [tool.to_openai() for tool in self.tools]

        openai_tool_choice: str | dict[str, Any]
        match self.tool_choice:
            case ToolChoiceEnum.auto | ToolChoiceEnum.none:
                openai_tool_choice = self.tool_choice
            case ToolChoiceEnum.required | ToolChoiceEnum.any:
                openai_tool_choice = ToolChoiceEnum.required.value
            case _:
                openai_tool_choice = self.tool_choice.model_dump()

        openai_request["tool_choice"] = openai_tool_choice

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
        r"""Create a ChatCompletionRequest from OpenAI ChatCompletion request format.

        Args:
            messages: List of message dicts in OpenAI format. Each dict must have
                a "role" key and optionally "content" or other role-specific fields.
            tools: List of tool dicts in OpenAI format, or `None`. Each tool dict must
                have "type" and "function" keys for function tools.
            continue_final_message: If `True` and the last message is an assistant,
                sets `AssistantMessage.prefix=True` on it.
            **kwargs: Additional request parameters. Supports both OpenAI names
                (e.g., "seed") and mistral-common names (e.g., `random_seed`).
                Cannot specify both "seed" and `random_seed`.

        Returns:
            A ChatCompletionRequest instance with messages and tools converted
            from OpenAI format.

        Raises:
            ValueError: If both "seed" and `random_seed` are specified in kwargs.
        """
        if "seed" in kwargs and "random_seed" in kwargs:
            raise ValueError("Cannot specify both `seed` and `random_seed`.")

        random_seed = kwargs.pop("seed", None)
        if random_seed is None:
            random_seed = kwargs.pop("random_seed", None)

        filtered_kwargs = cls._filter_cls_fields(kwargs)

        converted_messages: list[ChatMessage] = convert_openai_messages(messages)
        converted_messages = _map_continue_final_message(
            messages=converted_messages,
            continue_final_message=TypeAdapter(bool).validate_python(continue_final_message),
        )

        converted_tools = convert_openai_tools(tools) if tools is not None else None

        return cls(
            # Pydantic cannot express the runtime-selected request class's message generic here.
            messages=converted_messages,  # type: ignore[arg-type]
            tools=converted_tools,
            random_seed=random_seed,
            **filtered_kwargs,
        )


class InstructRequest(MistralBase, Generic[ChatMessageType, ToolType]):
    r"""Internal representation of a validated instruct request ready for tokenization.

    This class represents the fully normalized and validated request that will be
    passed to the tokenizer. It separates system prompts from regular messages and
    handles tool configuration.

    Note:
        This class is intended for internal use only. External users should use
        ChatCompletionRequest to build requests and convert to/from OpenAI format.

    Attributes:
        messages: List of chat messages (user and assistant only; system is separate).
        system_prompt: System prompt string, or `None` if no system prompt.
        available_tools: List of tools available to the assistant. If `None`, no tools
            are available. Tools are separate from messages here.
        truncate_at_max_tokens: Maximum token count for truncation. If `None`, no
            truncation is performed. If set, messages will be truncated from the
            beginning to fit within this limit.
        settings: Model configuration settings. Defaults to all `None` values.

    Examples:
        >>> from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
        >>> request = InstructRequest(
        ...     messages=[UserMessage(content="Hello, how are you?")],
        ...     system_prompt="You are a helpful assistant."
        ... )
    """

    messages: list[ChatMessageType]
    system_prompt: str | None = None
    available_tools: list[ToolType] | None = None
    truncate_at_max_tokens: int | None = None
    settings: ModelSettings = Field(default_factory=ModelSettings.none)

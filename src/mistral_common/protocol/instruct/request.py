from enum import Enum
from typing import Any, Generic

from pydantic import ConfigDict, Field, field_validator

from mistral_common.base import MistralBase
from mistral_common.exceptions import InvalidRequestException
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
from mistral_common.utils.json_utils import validate_json_schema_by_draft7


class ResponseFormats(str, Enum):
    r"""Enum of the different formats of an instruct response.

    Attributes:
        text: The response is a plain text.
        json: The response is a JSON object.
        json_schema: The response follows a custom JSON schema.

    Examples:
        >>> response_format = ResponseFormats.text
    """

    text = "text"
    json = "json_object"
    json_schema = "json_schema"


class ReasoningEffort(str, Enum):
    r"""Controls how much reasoning effort the model should apply.

    Attributes:
        none: No additional reasoning effort.
        high: High reasoning effort for complex tasks.
    """

    none = "none"
    high = "high"


class ModelSettings(MistralBase):
    r"""Model configuration settings for instruct requests.

    This class encapsulates various model configuration options that can be
    passed to the model during inference. Currently supports reasoning effort
    configuration, but can be extended with additional settings in the future.

    Attributes:
        reasoning_effort: Controls how much reasoning effort the model should apply when
            generating responses. Supported for tokenizer >= v15 and not supported for earlier versions.
        json_schema: The JSON schema to enforce on the response, derived from the request's
            response format. Supported for tokenizer >= v15 and not supported for earlier versions.
    """

    reasoning_effort: ReasoningEffort | None = None
    json_schema: dict[str, Any] | None = None

    @staticmethod
    def none() -> "ModelSettings":
        r"""Create a ModelSettings instance with default (None) values."""
        return ModelSettings()


class JsonSchema(MistralBase):
    r"""A named JSON schema for structured responses.

    Attributes:
        name: The schema name.
        description: An optional description of the schema.
        custom_schema: The JSON schema (aliased ``schema``).
        strict: Whether the model must strictly adhere to the schema.

    Examples:
        >>> schema = JsonSchema(name="obj", schema={"type": "object"})
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str | None = None
    custom_schema: dict = Field(..., alias="schema")
    strict: bool = False

    @field_validator("custom_schema")
    @classmethod
    def validate_custom_schema(cls, value: dict) -> dict:
        r"""Validate the schema against JSON Schema Draft 7."""
        validate_json_schema_by_draft7(value=value)
        return value


class SchemaRenderingMode(str, Enum):
    r"""Purpose for which a response-format schema is rendered.

    Attributes:
        grammar: Render for guided-decoding grammars.
        model_settings: Render for encoding into model settings.
    """

    grammar = "grammar"
    model_settings = "model_settings"


class ResponseFormat(MistralBase):
    r"""The format of the response.

    Attributes:
        type: The type of the response.
        json_schema: The JSON schema when ``type`` is ``json_schema``.

    Examples:
        >>> response_format = ResponseFormat(type=ResponseFormats.text)
    """

    type: ResponseFormats = ResponseFormats.text
    json_schema: JsonSchema | None = None

    def get_schema(self, purpose: SchemaRenderingMode) -> dict[str, Any] | None:
        r"""Return the JSON schema to enforce for this response format.

        Args:
            purpose: Why the schema is being rendered.

        Returns:
            The schema dict, or None when no constraint applies.

        Raises:
            InvalidRequestException: If ``type`` is ``json_schema`` but no schema is set.
        """
        schema: dict[str, Any] | None
        if self.type == ResponseFormats.json_schema:
            if self.json_schema is None:
                raise InvalidRequestException("Response format `json_schema` must define the schema")
            schema = (
                self.json_schema.custom_schema
                if (self.json_schema.strict or purpose == SchemaRenderingMode.model_settings)
                else {"type": "object"}
            )
        elif self.type == ResponseFormats.json:
            schema = {"anyOf": [{"type": "object"}, {"type": "array"}]}
        else:
            schema = None
        return schema


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
        reasoning_effort: Controls how much reasoning effort the model should apply.

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
    continue_final_message: bool = False
    reasoning_effort: ReasoningEffort | None = None

    def to_openai(
        self,
        reasoning_field_format: ReasoningFieldFormat | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        r"""Convert the request messages and tools into the OpenAI format.

        Args:
            reasoning_field_format: Format for converting thinking chunks in assistant messages.
                See `AssistantMessage.to_openai` for details.
            kwargs: Additional parameters to be added to the request.

        Returns:
            The request in the OpenAI format.

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
            exclude={"messages", "tools", "truncate_for_context_length", "tool_choice"},
            exclude_none=True,
            by_alias=True,
        )

        # Rename random_seed to seed.
        seed = openai_request.pop("random_seed", None)
        if seed is not None:
            openai_request["seed"] = seed

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

        random_seed = kwargs.pop("seed", None)
        if random_seed is None:
            random_seed = kwargs.pop("random_seed", None)

        filtered_kwargs = cls._filter_cls_fields(kwargs)

        converted_messages: list[ChatMessage] = convert_openai_messages(messages)

        converted_tools = convert_openai_tools(tools) if tools is not None else None

        return cls(
            messages=converted_messages,  # type: ignore[arg-type]
            tools=converted_tools,
            random_seed=random_seed,
            continue_final_message=continue_final_message,
            **filtered_kwargs,
        )


class InstructRequest(MistralBase, Generic[ChatMessageType, ToolType]):
    r"""A valid Instruct request to be tokenized.

    Note:
        This class is intended for internal use only. External users should use `ChatCompletionRequest` to build
        requests and to convert to and from the OpenAI format.

    Attributes:
        messages: The history of the conversation.
        system_prompt: The system prompt to be used for the conversation.
        available_tools: The tools available to the assistant.
        truncate_at_max_tokens: The maximum number of tokens to truncate the conversation at.
        continue_final_message: Whether to continue the final message.
        settings: Model configuration settings for the request.

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
    settings: ModelSettings = Field(default_factory=ModelSettings.none)

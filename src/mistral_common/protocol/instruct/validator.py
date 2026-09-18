import re
from collections.abc import Sequence
from enum import Enum
from typing import Generic

from jsonschema import Draft7Validator, SchemaError
from typing_extensions import assert_never

from mistral_common.exceptions import (
    InvalidAssistantMessageException,
    InvalidFunctionCallException,
    InvalidMessageStructureException,
    InvalidRequestException,
    InvalidSystemPromptException,
    InvalidToolException,
    InvalidToolMessageException,
    InvalidToolSchemaException,
    InvalidUserMessageException,
    MistralCommonException,
)
from mistral_common.protocol.instruct.chunk import (
    AudioChunk,
    AudioURLChunk,
    ContentChunk,
    ImageChunk,
    ImageURLChunk,
    TextChunk,
    ThinkChunk,
)
from mistral_common.protocol.instruct.messages import (
    UATS,
    AssistantMessage,
    AssistantMessageType,
    FinetuningAssistantMessage,
    Roles,
    SystemMessage,
    SystemMessageType,
    ToolMessageType,
    UserMessage,
    UserMessageType,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    FunctionCall,
    Tool,
    ToolCall,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion

_NULL_TOOL_CALL_ID = "null"
_TOOL_CALL_ID_REGEX = re.compile(r"^[a-zA-Z0-9]{9}$")
_INVALID_TOOL_CALL_ID_MESSAGE = (
    f"Tool call id must be a non-empty string other than '{_NULL_TOOL_CALL_ID}' for tokenizer version 13 or newer."
)


def _validate_content_chunk_types(
    content: str | Sequence[ContentChunk] | None,
    allowed: tuple[type[ContentChunk], ...],
    role: str,
    exception_cls: type[MistralCommonException],
) -> None:
    r"""Validate that all content chunks in a list are of allowed types.

    Args:
        content: Content to validate. If not a list, returns immediately.
        allowed: Tuple of ContentChunk subclasses that are allowed.
        role: The message role (for error messages).
        exception_cls: Exception class to raise if validation fails.

    Raises:
        exception_cls: If content is a list containing any chunk not in allowed.
            Error message includes the role and list of invalid chunk type names.
    """
    if not isinstance(content, list):
        return
    invalid = sorted({type(chunk).__name__ for chunk in content if not isinstance(chunk, allowed)})
    if invalid:
        raise exception_cls(f"Unexpected content chunk types in {role} message: {invalid}")


class ValidationMode(str, Enum):
    r"""Validation mode controlling which validation rules are applied.

    Attributes:
        serving: Strict validation for production serving. Requires model to be
            specified and enforces all constraints.
        finetuning: Validation for finetuning scenarios. May allow some fields
            that are not allowed in serving mode.
        test: Lenient validation for testing. Skips some production constraints.
        agnostic: Pipeline-agnostic validation. Applies all content checks
            without assuming a downstream pipeline, so no terminal-role rule is
            enforced.

    Examples:
        >>> mode = ValidationMode.serving
    """

    serving = "serving"
    finetuning = "finetuning"
    test = "test"
    agnostic = "agnostic"


class MistralRequestValidator(Generic[UserMessageType, AssistantMessageType, ToolMessageType, SystemMessageType]):
    r"""Validates Mistral chat completion requests and messages.

    Performs comprehensive validation of request structure, message content,
    tool definitions, and model settings according to the specified validation mode.

    Examples:
        >>> from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
        >>> validator = MistralRequestValidator()
        >>> messages = [UserMessage(content="Hello how are you ?")]
        >>> validator.validate_messages(messages)
    """

    _allow_tool_call_and_content: bool = False

    def __init__(self, mode: ValidationMode = ValidationMode.test):
        r"""Initialize the MistralRequestValidator.

        Args:
            mode: The validation mode to use. Options:
                - `ValidationMode.serving`: Strict production validation
                - `ValidationMode.finetuning`: Finetuning-specific validation
                - `ValidationMode.test`: Lenient testing validation (default)
                - `ValidationMode.agnostic`: Pipeline-agnostic validation that
                    applies all content checks without terminal-role rules
        """
        self._mode = mode

    @property
    def mode(self) -> ValidationMode:
        r"""The validation mode this validator enforces.

        Returns:
            The ValidationMode (serving, finetuning, test, or agnostic) this
            instance was constructed with.
        """
        return self._mode

    def validate_messages(self, messages: list[UATS]) -> None:
        r"""Validate a list of messages.

        Checks message structure (alternating user/assistant, valid roles) and
        content (valid chunk types for each role, valid tool calls, etc.).

        Args:
            messages: List of message objects to validate. Each message must be
                a subclass of BaseMessage with a valid role.

        Raises:
            InvalidMessageStructureException: If message structure is invalid
                (e.g., consecutive assistant messages without intervening user message).
            InvalidUserMessageException: If a user message has invalid content.
            InvalidAssistantMessageException: If an assistant message has invalid content.
            InvalidToolMessageException: If a tool message has invalid content.
            InvalidSystemPromptException: If a system message has invalid content.
            InvalidToolCallException: If a tool call in an assistant message is invalid.

        Examples:
            >>> from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
            >>> validator = MistralRequestValidator()
            >>> messages = [AssistantMessage(content="Hi"), UserMessage(content="Hello")]
            >>> validator.validate_messages(messages)
        """
        self._validate_message_list_structure(messages)
        self._validate_message_list_content(messages)

    def validate_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest[UATS]:
        r"""Validate a complete ChatCompletionRequest.

        Performs full validation including:
        - Model field (required in serving mode)
        - Message list structure and content
        - Tool definitions and tool calls
        - Model settings

        Args:
            request: The ChatCompletionRequest to validate.

        Returns:
            The same request object after validation. Note: validation may mutate
            the request in-place for some normalizations.

        Raises:
            InvalidRequestException: If request fails validation (missing model in
                serving mode, invalid messages, invalid tools, etc.).

        Examples:
            >>> from mistral_common.protocol.instruct.messages import UserMessage
            >>> validator = MistralRequestValidator()
            >>> request = ChatCompletionRequest(messages=[UserMessage(content="Hello")])
            >>> validated_request = validator.validate_request(request)
        """

        if self._mode == ValidationMode.serving:
            if request.model is None:
                raise InvalidRequestException("Model name parameter is required for serving mode")

        # Validate the messages
        self.validate_messages(request.messages)

        # Validate the tools
        self._validate_tools(request.tools or [])

        self._validate_model_settings(request)

        return request

    def _validate_function(self, function: Function) -> None:
        r"""Check that the function schema and name are valid.

        The schema must be a valid JSON Schema (Draft 7) and the name must
        match `^[a-zA-Z0-9_-]{1,64}$`.

        Raises:
            InvalidToolSchemaException: If the parameters schema is not valid JSON Schema.
            InvalidToolException: If the function name is invalid.
        """
        try:
            Draft7Validator.check_schema(function.parameters)
        except SchemaError as e:
            raise InvalidToolSchemaException(f"Invalid tool schema: {e.message}")

        if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", function.name):
            raise InvalidToolException(
                f"Function name was {function.name} but must be a-z, A-Z, 0-9, "
                "or contain underscores and dashes, with a maximum length of 64."
            )

    def _validate_tools(self, tools: list[Tool]) -> None:
        r"""Check that every tool's function schema and name are valid.

        Raises:
            InvalidToolSchemaException: If any tool's parameters schema is not
                valid JSON Schema.
            InvalidToolException: If any tool's function name is invalid.
        """

        for tool in tools:
            self._validate_function(tool.function)

    def _validate_user_message(self, message: UserMessageType) -> None:
        self._validate_user_content_chunks(message.content)

    def _validate_user_content_chunks(self, content: str | Sequence[ContentChunk] | None) -> None:
        r"""v1/v2 user messages accept text content only (image >= v3, audio >= v7)."""
        _validate_content_chunk_types(content, (TextChunk,), "user", InvalidUserMessageException)

    def _validate_assistant_content_chunks(self, content: str | Sequence[ContentChunk] | None) -> None:
        r"""Pre-v11 assistant messages accept text content only."""
        _validate_content_chunk_types(content, (TextChunk,), "assistant", InvalidAssistantMessageException)

    def _validate_system_content_chunks(self, content: str | Sequence[ContentChunk] | None) -> None:
        r"""v1-v3 system messages accept text content only."""
        _validate_content_chunk_types(content, (TextChunk,), "system", InvalidSystemPromptException)

    def _validate_tool_content_chunks(self, content: str | Sequence[ContentChunk] | None) -> None:
        r"""Pre-v15 tool messages accept text content only."""
        _validate_content_chunk_types(content, (TextChunk,), "tool", InvalidToolMessageException)

    def _validate_tool_message(self, message: ToolMessageType) -> None:
        r"""Check that a tool message's name, content chunks, and ID are valid.

        Raises:
            InvalidToolMessageException: If the optional tool name is set but
                does not match `^[a-zA-Z0-9_-]{1,64}$`, or content chunks are
                not text.
        """
        self._validate_tool_content_chunks(message.content)
        if message.name is not None:
            if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", message.name):
                raise InvalidToolMessageException(
                    f"Function name was {message.name} but must be a-z, A-Z, 0-9, "
                    "or contain underscores and dashes, with a maximum length of 64."
                )
        self._validate_tool_message_id(message)

    def _validate_tool_message_id(self, message: ToolMessageType) -> None:
        return

    def _validate_system_message(self, message: SystemMessageType) -> None:
        r"""Check that a system message has content and valid content chunks.

        Raises:
            InvalidSystemPromptException: If content is `None` or contains
                non-text chunks.
        """
        if message.content is None:
            raise InvalidSystemPromptException("System prompt must have content")
        self._validate_system_content_chunks(message.content)

    def _validate_function_call(self, function_call: FunctionCall) -> None:
        r"""Check that a function call's name is valid.

        Raises:
            InvalidFunctionCallException: If the name does not match
                `^[a-zA-Z0-9_-]{1,64}$`.
        """
        if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", function_call.name):
            raise InvalidFunctionCallException(
                f"Function name was {function_call.name} but must be a-z, A-Z, 0-9, "
                "or contain underscores and dashes, with a maximum length of 64."
            )

    def _validate_tool_call(self, tool_call: ToolCall, is_last_message: bool) -> None:
        r"""Check that a tool call has a valid function call.

        Raises:
            InvalidFunctionCallException: If the function call's name is invalid.
        """

        self._validate_function_call(tool_call.function)

    def _validate_assistant_message(self, message: AssistantMessageType, is_last_message: bool = False) -> None:
        r"""Check that an assistant message's content, tool calls, and prefix are valid.

        Content and `tool_calls` are mutually exclusive unless
        `_allow_tool_call_and_content` is `True`. Tool calls are each validated.
        In finetuning mode, FinetuningAssistantMessage weights must be 0 or 1.
        A prefix message must be the last message in the conversation.

        Args:
            message: The assistant message to validate.
            is_last_message: `True` if this is the last message of the conversation,
                required to allow `prefix=True`.

        Raises:
            InvalidAssistantMessageException: If content and `tool_calls` are both
                present or both absent, a tool call is invalid, the weight is
                invalid, or `prefix=True` on a non-last message.
        """

        self._validate_assistant_content_chunks(message.content)

        # Validate that the message has either text or tool_calls
        # but not both and not neither.
        if (not self._allow_tool_call_and_content) and (bool(message.content) == bool(message.tool_calls)):
            raise InvalidAssistantMessageException(
                "Assistant message must have either content or tool_calls, but not both."
            )

        # If we have tool calls, validate them
        if message.tool_calls is not None:
            # Validate that the tool calls are valid
            for tool_call in message.tool_calls:
                self._validate_tool_call(tool_call, is_last_message=is_last_message)

        if self._mode in {ValidationMode.finetuning, ValidationMode.agnostic} and isinstance(
            message, FinetuningAssistantMessage
        ):
            if message.weight is not None and message.weight not in [0, 1]:
                raise InvalidAssistantMessageException("Assistant message weight must be either 0 or 1")

        if message.prefix:
            if not is_last_message:
                raise InvalidAssistantMessageException("Assistant message with prefix True must be last message")
            # note : we already validate that assistant message has content 3 lines up.

    def _validate_tool_calls_followed_by_tool_messages(self, messages: list[UATS]) -> None:
        r"""Check that every tool call is followed by a matching tool message.

        Each assistant message's `tool_calls` must be answered by exactly as many
        tool messages before the next assistant message. In serving mode the
        counts must balance exactly; in finetuning mode extra tool responses
        are rejected but missing ones are tolerated.

        Raises:
            InvalidMessageStructureException: If tool calls and tool messages
                do not match up per the active mode's rule.
        """
        prev_role = None
        expected_tool_messages = 0
        for message in messages:
            if prev_role is None:
                prev_role = message.role
                continue

            if message.role == Roles.tool:
                expected_tool_messages -= 1
            elif message.role == Roles.assistant:
                # if we have an assistant message and we have not received all the function calls
                # we need to raise an exception
                if expected_tool_messages != 0:
                    raise InvalidMessageStructureException("Not the same number of function calls and responses")

                if message.tool_calls is not None:
                    # Validate that the number of function calls and responses are the same
                    expected_tool_messages = len(message.tool_calls)

            prev_role = message.role

        if expected_tool_messages != 0 and self._mode == ValidationMode.serving:
            raise InvalidMessageStructureException("Not the same number of function calls and responses")
        elif expected_tool_messages < 0 and self._mode == ValidationMode.finetuning:
            raise InvalidMessageStructureException("More tool responses than tool calls")

    def _validate_message_order(self, messages: list[UATS]) -> None:
        r"""Check that consecutive roles are in a valid sequence.

        Allowed transitions: system can be followed by user/assistant/system,
        user by assistant/system/user, assistant by assistant/user/tool, and
        tool by assistant/tool/user.

        Raises:
            InvalidMessageStructureException: If a message's role does not
                follow a valid transition from the previous role.
        """
        previous_role = None
        for message in messages:
            current_role = message.role

            if previous_role is not None:
                if previous_role == Roles.system:
                    expected_roles = {Roles.user, Roles.assistant, Roles.system}
                elif previous_role == Roles.user:
                    expected_roles = {Roles.assistant, Roles.system, Roles.user}
                elif previous_role == Roles.assistant:
                    expected_roles = {Roles.assistant, Roles.user, Roles.tool}
                elif previous_role == Roles.tool:
                    expected_roles = {Roles.assistant, Roles.tool, Roles.user}
                else:
                    assert_never(previous_role)

                if current_role not in expected_roles:
                    raise InvalidMessageStructureException(
                        f"Unexpected role '{current_role}' after role '{previous_role}'"
                    )

            previous_role = current_role

    def _validate_last_message(self, message: UATS) -> None:
        r"""Check that the last message's role is valid for the mode.

        In finetuning mode the last message must be an assistant without
        `prefix=True`. In other modes it must be a user or tool message, or an
        assistant message with `prefix=True` (continuation). In agnostic mode,
        no pipeline-specific final-role rule is enforced.

        Raises:
            InvalidMessageStructureException: If the last message's role or
                prefix is invalid for the mode.
        """

        if self._mode == ValidationMode.agnostic:
            return
        last_message_role = message.role
        if self._mode == ValidationMode.finetuning:
            if last_message_role != Roles.assistant:
                raise InvalidMessageStructureException(
                    f"Expected last role Assistant for finetuning but got {last_message_role}"
                )
            if isinstance(message, AssistantMessage) and message.prefix:
                raise InvalidMessageStructureException("Cannot continue final message in finetuning mode")
        else:
            bad_assistant = isinstance(message, AssistantMessage) and not message.prefix
            bad_role = message.role not in {Roles.user, Roles.tool}
            if bad_assistant and bad_role:
                raise InvalidMessageStructureException(
                    "Expected last role User or Tool (or Assistant with prefix) "
                    f"for serving but got {last_message_role}"
                )

    def _validate_message_list_structure(self, messages: list[UATS]) -> None:
        r"""Check the overall structure of the conversation.

        The conversation must have at least one message; a single message must
        be a user or system message; and message order, last-message role, and
        tool call/result pairing are all validated.

        Raises:
            InvalidMessageStructureException: If the conversation is empty,
            starts with a non-user/system message, or violates message order
            or tool call pairing rules.
        """

        if len(messages) == 0:
            raise InvalidMessageStructureException("Conversation must have at least one message")

        # If we have one message it must be a user or a system message
        if len(messages) == 1:
            if messages[0].role not in {Roles.user, Roles.system}:
                raise InvalidMessageStructureException("Conversation must start with a user message or system message")

        # Always check the last message if in fine-tuning mode
        if self._mode == ValidationMode.finetuning or len(messages) > 1:
            self._validate_last_message(messages[-1])

        self._validate_message_order(messages)
        self._validate_tool_calls_followed_by_tool_messages(messages)

    def _validate_message_list_content(self, messages: list[UATS]) -> None:
        r"""Check each message's content according to its role.

        Dispatches to the per-role validators (`_validate_user_message`,
        `_validate_assistant_message`, `_validate_tool_message`,
        `_validate_system_message`).

        Raises:
            InvalidRequestException: If a message has an unsupported role.
            MistralCommonException: Subclasses raised by the per-role validators.
        """

        for idx, message in enumerate(messages):
            if message.role == Roles.user:
                self._validate_user_message(message)
            elif message.role == Roles.assistant:
                self._validate_assistant_message(message, is_last_message=idx == len(messages) - 1)
            elif message.role == Roles.tool:
                self._validate_tool_message(message)
            elif message.role == Roles.system:
                self._validate_system_message(message)
            else:
                raise InvalidRequestException(f"Unsupported message type {type(message)}")

    def _validate_model_settings(self, request: ChatCompletionRequest) -> None:
        if (reasoning_effort := request.reasoning_effort) is not None:
            raise InvalidRequestException(f"{reasoning_effort=} is not supported for this model")


class MistralRequestValidatorV3(MistralRequestValidator):
    r"""Validator for v3 Mistral requests.

    This validator adds additional validation for tool call IDs.

    Examples:
        >>> validator = MistralRequestValidatorV3()
    """

    def _validate_user_content_chunks(self, content: str | Sequence[ContentChunk] | None) -> None:
        r"""v3 user messages accept text and image chunks (audio >= v7)."""
        _validate_content_chunk_types(
            content, (TextChunk, ImageChunk, ImageURLChunk), "user", InvalidUserMessageException
        )

    def _validate_tool_message_id(self, message: ToolMessageType) -> None:
        r"""Check that a tool message's call ID is defined.

        Raises:
            InvalidRequestException: If the tool call ID is None.
        """
        if message.tool_call_id is None:
            raise InvalidRequestException("Tool call id has to be defined.")

        if not _TOOL_CALL_ID_REGEX.match(message.tool_call_id):
            raise InvalidToolMessageException(
                f"Tool call id was {message.tool_call_id} but must be a-z, A-Z, 0-9, with a length of 9."
            )

    def _validate_tool_call_id(self, tool_call: ToolCall, is_last_message: bool) -> None:
        r"""Check that a tool call ID is valid for the mode.

        The "null" ID is only allowed for the last assistant message in
        finetuning mode. All other IDs must match `^[a-zA-Z0-9]{9}$`.

        Args:
            tool_call: The tool call whose ID is validated.
            is_last_message: `True` if the parent message is the last message of
                the conversation.

        Raises:
            InvalidFunctionCallException: If the ID is "null" in a context that
                does not allow it, or the ID does not match the expected format.
        """
        if tool_call.id == _NULL_TOOL_CALL_ID:
            match self._mode:
                case ValidationMode.finetuning | ValidationMode.agnostic:
                    if not is_last_message:
                        raise InvalidFunctionCallException(
                            "Tool call id of assistant message that is not last has to be defined in "
                            f"{self._mode.value} mode."
                        )
                    return
                case ValidationMode.serving:
                    raise InvalidFunctionCallException("Tool call id has to be defined in serving mode.")
                case _:
                    raise InvalidFunctionCallException(
                        f"Tool call id '{_NULL_TOOL_CALL_ID}' is only allowed for the last assistant message "
                        "in finetuning mode."
                    )

        if not _TOOL_CALL_ID_REGEX.match(tool_call.id):
            raise InvalidFunctionCallException(
                f"Tool call id was {tool_call.id} but must be a-z, A-Z, 0-9, with a length of 9."
            )

    def _validate_tool_call(self, tool_call: ToolCall, is_last_message: bool) -> None:
        self._validate_tool_call_id(tool_call, is_last_message=is_last_message)
        self._validate_function_call(tool_call.function)

    def _validate_last_message(self, message: UATS) -> None:
        super()._validate_last_message(message)

        if self._mode == ValidationMode.finetuning:
            # in finetuning mode it has to be an assistant message
            # as checked by parent `_validate_last_message`
            if message.tool_calls is not None:
                for tool_call in message.tool_calls:
                    self._validate_tool_call(tool_call, is_last_message=True)


class MistralRequestValidatorV5(MistralRequestValidatorV3):
    r"""Validator for v5 Mistral requests.

    This validator allows for both tool calls and content in the assistant message.

    Note:
        For requests containing audio, this validator ensures that no system prompt is present.

    Examples:
        >>> validator = MistralRequestValidatorV5()
    """

    _allow_tool_call_and_content: bool = True

    def _validate_user_content_chunks(self, content: str | Sequence[ContentChunk] | None) -> None:
        r"""v7+ user messages accept text, image and audio chunks."""
        _validate_content_chunk_types(
            content,
            (TextChunk, ImageChunk, ImageURLChunk, AudioChunk, AudioURLChunk),
            "user",
            InvalidUserMessageException,
        )

    def _validate_system_content_chunks(self, content: str | Sequence[ContentChunk] | None) -> None:
        r"""v7+ system messages accept text, audio and thinking chunks."""
        _validate_content_chunk_types(
            content, (TextChunk, AudioChunk, ThinkChunk), "system", InvalidSystemPromptException
        )

    def _validate_system_prompt_and_audio(self, messages: list[UATS]) -> None:
        r"""Validates that system prompts and audio chunks are not used together in v5."""

        def _is_system(message: UATS) -> bool:
            return isinstance(message, SystemMessage)

        def _has_audio(message: UATS) -> bool:
            return (
                isinstance(message, UserMessage)
                and isinstance(message.content, list)
                and any(isinstance(chunk, (AudioChunk, AudioURLChunk)) for chunk in message.content)
            )

        has_sp = any(_is_system(message=message) for message in messages)
        has_sp_and_audio = has_sp and any(_has_audio(message=message) for message in messages)

        if has_sp_and_audio:
            sp_indexes = [i for i, message in enumerate(messages) if _is_system(message=message)]
            audio_indexes = [i for i, message in enumerate(messages) if _has_audio(message=message)]
            raise ValueError(
                f"Found system messages at indexes {sp_indexes} and audio chunks in messages at indexes {audio_indexes}"
                ". This is not allowed prior to the tokenizer version 13."
            )

    def _validate_message_list_structure(self, messages: list[UATS]) -> None:
        super()._validate_message_list_structure(messages=messages)
        self._validate_system_prompt_and_audio(messages)


class MistralRequestValidatorV11(MistralRequestValidatorV5):
    r"""Validator for v11 Mistral requests.

    This validator extends v5 functionality by:
    - Adding stricter tool call/result pairing validation.
    - Allowing thinking chunks in assistant messages.
    - Allowing system prompts with audio chunks
    """

    def _validate_tool_calls_followed_by_tool_messages(self, messages: list[UATS]) -> None:
        r"""Check tool call/result pairing by ID.

        Extends the base check with ID-level rules: tool results must reference
        pending call IDs without duplicates, and IDs must be unique within an
        assistant message.

        Raises:
            InvalidMessageStructureException: If tool calls and tool messages
                do not pair up by ID per the active mode's rule, or IDs are
                duplicated or unexpected.
        """
        prev_role = None
        expected_tool_ids: set[str] = set()
        observed_tool_ids: set[str] = set()
        for message in messages:
            if prev_role is None:
                prev_role = message.role
                continue

            if message.role == Roles.tool:
                tool_call_id = message.tool_call_id
                if tool_call_id in observed_tool_ids:
                    raise InvalidMessageStructureException(f"Duplicate tool call id {tool_call_id} in tool results")
                if tool_call_id not in expected_tool_ids:
                    raise InvalidMessageStructureException(f"Unexpected tool call id {tool_call_id} in tool results")
                observed_tool_ids.add(tool_call_id)

            elif message.role == Roles.assistant:
                # if we have an assistant message and we have not received all the function calls
                # we need to raise an exception
                if len(expected_tool_ids) != len(observed_tool_ids):
                    raise InvalidMessageStructureException("Not the same number of function calls and responses")

                expected_tool_ids.clear()
                observed_tool_ids.clear()
                if message.tool_calls is not None:
                    # Validate that the number of function calls and ids are the same
                    for tool_call in message.tool_calls:
                        if tool_call.id in expected_tool_ids:
                            raise InvalidMessageStructureException(
                                f"Duplicate tool call id {tool_call.id} in assistant message"
                            )
                        expected_tool_ids.add(tool_call.id)

            prev_role = message.role

        if len(expected_tool_ids) != len(observed_tool_ids) and self._mode == ValidationMode.serving:
            raise InvalidMessageStructureException("Not the same number of function calls and responses")
        elif len(expected_tool_ids) < len(observed_tool_ids) and self._mode in {
            ValidationMode.finetuning,
            ValidationMode.agnostic,
        }:
            raise InvalidMessageStructureException("More tool responses than tool calls")

    def _validate_assistant_content_chunks(self, content: str | Sequence[ContentChunk] | None) -> None:
        r"""v11+ assistant messages accept text and thinking chunks."""
        _validate_content_chunk_types(content, (TextChunk, ThinkChunk), "assistant", InvalidAssistantMessageException)

    def _validate_system_prompt_and_audio(self, messages: list[UATS]) -> None:
        r"""Allow system prompts and audio chunks to coexist."""
        return


class MistralRequestValidatorV13(MistralRequestValidatorV11):
    r"""Validator for v13 Mistral requests."""

    def _validate_tool_message_id(self, message: ToolMessageType) -> None:
        if not message.tool_call_id or message.tool_call_id == _NULL_TOOL_CALL_ID:
            raise InvalidToolMessageException(_INVALID_TOOL_CALL_ID_MESSAGE)

    def _validate_tool_call_id(self, tool_call: ToolCall, is_last_message: bool) -> None:
        if not tool_call.id or tool_call.id == _NULL_TOOL_CALL_ID:
            raise InvalidFunctionCallException(_INVALID_TOOL_CALL_ID_MESSAGE)


class MistralRequestValidatorV15(MistralRequestValidatorV13):
    r"""Validator for v15 Mistral requests."""

    def _validate_system_content_chunks(self, content: str | Sequence[ContentChunk] | None) -> None:
        r"""v15 system messages accept text and audio but reject thinking chunks."""
        _validate_content_chunk_types(content, (TextChunk, AudioChunk), "system", InvalidSystemPromptException)

    def _validate_tool_content_chunks(self, content: str | Sequence[ContentChunk] | None) -> None:
        r"""v15 tool messages accept all content chunk types except thinking (rejected at ToolMessage model level)."""
        return

    def _validate_model_settings(self, request: ChatCompletionRequest) -> None:
        pass


def get_validator(version: TokenizerVersion, mode: ValidationMode) -> MistralRequestValidator:
    r"""Get the appropriate validator for a given tokenizer version and validation mode.

    The validator version matches the tokenizer version and enforces the
    content chunk types, tool call ID rules, and model settings supported by
    that version.

    Args:
        version: The tokenizer version the validator should match.
        mode: The validation mode (serving, finetuning, test, or agnostic)
            controlling which constraints are enforced.

    Returns:
        The validator instance appropriate for the version and mode.
    """
    validator: MistralRequestValidator
    match version:
        case TokenizerVersion.v1 | TokenizerVersion.v2:
            validator = MistralRequestValidator(mode=mode)
        case TokenizerVersion.v3:
            validator = MistralRequestValidatorV3(mode=mode)
        case TokenizerVersion.v7:
            validator = MistralRequestValidatorV5(mode=mode)
        case TokenizerVersion.v11:
            validator = MistralRequestValidatorV11(mode=mode)
        case TokenizerVersion.v13:
            validator = MistralRequestValidatorV13(mode=mode)
        case TokenizerVersion.v15:
            validator = MistralRequestValidatorV15(mode=mode)
        case _:
            assert_never(f"Unsupported tokenizer version: {version}")

    return validator

import json
from typing import Sequence

from mistral_common.experimental.utils import (
    _split_integer_list_by_value,
    _split_tokens_by_one_occurrence_control_token,
)
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenizer, TokenizerVersion


class InvalidToolCallError(ValueError):
    r"""Raised when tool call tokens cannot be decoded into a valid tool call."""


class InvalidArgsToolCallError(InvalidToolCallError):
    r"""Raised when a tool call's arguments are invalid, e.g. not valid JSON."""


def _split_content_and_tool_calls(
    tokens: list[int], tool_call_token_id: int
) -> tuple[list[int], tuple[list[int], ...]]:
    r"""Split the content and tool calls from a list of tokens.

    The content is the first sequence of tokens that does not start with the
    tool call token ID. The tool calls are the remaining sequences, each
    starting with the tool call token ID.

    Args:
        tokens: The token IDs to split.
        tool_call_token_id: The token ID that indicates the start of a tool
            call.

    Returns:
        A tuple of (`content_tokens`, `tool_call_tokens`): the content token IDs
        and a tuple of per-tool-call token ID sequences. Both are empty when
        tokens is empty.
    """
    if not tokens:
        return [], ()

    maybe_content_and_tools_calls = _split_integer_list_by_value(tokens, tool_call_token_id)

    has_content = maybe_content_and_tools_calls[0][0] != tool_call_token_id
    if has_content:
        content_tokens = maybe_content_and_tools_calls[0]
        tools_calls_tokens = maybe_content_and_tools_calls[1:]
    else:
        content_tokens = []
        tools_calls_tokens = maybe_content_and_tools_calls

    return content_tokens, tools_calls_tokens


def _decode_tool_calls_v2_up_to_v7(tool_call_tokens: list[int], tokenizer: Tokenizer) -> list[ToolCall]:
    r"""Decode a list of tool call tokens into a list of tool calls for tokenizer versions v2 to v7.

    Note:
        Expects the tool call tokens to be in the format:

        `[TOOL_CALLS][{"id": "call_id", "name": "name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ...]`
        or

        `[TOOL_CALLS][{"name": "name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ...]`

    Args:
        tool_call_tokens: The token IDs to decode.
        tokenizer: The tokenizer to use for decoding.

    Returns:
        The decoded tool calls.

    Raises:
        InvalidToolCallError: If the decoded payload is not a JSON list.
        InvalidArgsToolCallError: If the payload is not valid JSON or a tool
            call's arguments are not a dict.
    """
    tool_calls_list_string = tokenizer.decode(tool_call_tokens, special_token_policy=SpecialTokenPolicy.IGNORE)
    try:
        tool_calls_decoded_list = json.loads(tool_calls_list_string)
    except json.JSONDecodeError as e:
        raise InvalidToolCallError(
            "Invalid tool call tokenization. Expected a JSON list of tool calls.",
        ) from e

    if not isinstance(tool_calls_decoded_list, list):
        raise InvalidToolCallError("Invalid tool call tokenization. Expected a list of tool calls.")

    for tool_call in tool_calls_decoded_list:
        if not isinstance(tool_call, dict) or "name" not in tool_call:
            raise InvalidToolCallError("Invalid tool call tokenization. Expected a dict with a name.")
        if "arguments" not in tool_call or not isinstance(tool_call["arguments"], dict):
            raise InvalidArgsToolCallError("Invalid tool call arguments tokenization. Expected a dict.")

    return [
        ToolCall(
            id=tool_call.get("id", "null"),
            function=FunctionCall(name=tool_call["name"], arguments=tool_call["arguments"]),
        )
        for tool_call in tool_calls_decoded_list
    ]


def _split_v11_tool_call_tokens(
    tool_call_tokens: list[int], tokenizer: Tokenizer, control_token: str
) -> tuple[list[int], list[int]]:
    r"""Split v11+ tool call tokens around a control token that must occur exactly once.

    Args:
        tool_call_tokens: The token IDs to split.
        tokenizer: The tokenizer used to resolve the control token.
        control_token: The control token to split on, e.g. `[ARGS]`.

    Returns:
        A tuple of (`before`, `after`): the token IDs before and after the control token.

    Raises:
        InvalidToolCallError: If the control token is missing or appears more than once.
    """
    try:
        return _split_tokens_by_one_occurrence_control_token(
            list_=tool_call_tokens, tokenizer=tokenizer, control_token=control_token
        )
    except ValueError as e:
        raise InvalidToolCallError(f"Invalid tool call tokenization. {e}") from e


def _decode_v11_tool_call_arguments(args_tokens: list[int], tokenizer: Tokenizer) -> str:
    r"""Decode v11+ tool call argument tokens and check they form a JSON object.

    Args:
        args_tokens: The token IDs following the `[ARGS]` control token.
        tokenizer: The tokenizer to use for decoding.

    Returns:
        The arguments serialized as a JSON object string, as `FunctionCall` stores them.

    Raises:
        InvalidArgsToolCallError: If the arguments are not valid JSON or not a JSON object.
    """
    try:
        arguments = json.loads(tokenizer.decode(args_tokens, special_token_policy=SpecialTokenPolicy.IGNORE))
    except json.JSONDecodeError as e:
        raise InvalidArgsToolCallError("Invalid tokenized tool call arguments.") from e
    # Mirror the v2-v7 contract: FunctionCall would otherwise raise a pydantic error for lists/scalars
    # and silently accept strings or null.
    if not isinstance(arguments, dict):
        raise InvalidArgsToolCallError("Invalid tool call arguments tokenization. Expected a dict.")
    return json.dumps(arguments)


def _decode_tool_call_v11_with_call_id(tool_call_tokens: list[int], tokenizer: Tokenizer) -> ToolCall:
    r"""Decode a list of tool call tokens into a tool call for tokenizer version v11 with call ID.

    Note:
        Expects the tool call tokens to be in the format:

        `[TOOL_CALLS]name[CALL_ID]call_id[ARGS]{"arg1": "value1", "arg2": "value2"}`

    Args:
        tool_call_tokens: The token IDs to decode.
        tokenizer: The tokenizer to use for decoding.

    Returns:
        The decoded tool call with its call ID.

    Raises:
        InvalidToolCallError: If [CALL_ID] or [ARGS] is missing or appears more than once.
        InvalidArgsToolCallError: If the arguments are not valid JSON or not a JSON object.
    """
    name, call_id_and_args = _split_v11_tool_call_tokens(
        tool_call_tokens=tool_call_tokens, tokenizer=tokenizer, control_token="[CALL_ID]"
    )

    call_id, args = _split_v11_tool_call_tokens(
        tool_call_tokens=call_id_and_args, tokenizer=tokenizer, control_token="[ARGS]"
    )

    return ToolCall(
        id=tokenizer.decode(call_id),
        function=FunctionCall(
            name=tokenizer.decode(name),
            arguments=_decode_v11_tool_call_arguments(args_tokens=args, tokenizer=tokenizer),
        ),
    )


def _decode_tool_call_v11(tool_call_tokens: list[int], tokenizer: Tokenizer) -> ToolCall:
    r"""Decode a list of tool call tokens into a tool call for tokenizer version v11 without call ID.

    Note:
        Expects the tool call tokens to be in the format:

        `[TOOL_CALLS]name[ARGS]{"arg1": "value1", "arg2": "value2"}`

    Args:
        tool_call_tokens: The token IDs to decode.
        tokenizer: The tokenizer to use for decoding.

    Returns:
        The decoded tool call.

    Raises:
        InvalidToolCallError: If [ARGS] is missing or appears more than once.
        InvalidArgsToolCallError: If the arguments are not valid JSON or not a JSON object.
    """
    name, args = _split_v11_tool_call_tokens(
        tool_call_tokens=tool_call_tokens, tokenizer=tokenizer, control_token="[ARGS]"
    )
    return ToolCall(
        function=FunctionCall(
            name=tokenizer.decode(name, special_token_policy=SpecialTokenPolicy.IGNORE),
            arguments=_decode_v11_tool_call_arguments(args_tokens=args, tokenizer=tokenizer),
        ),
    )


def _decode_tool_calls(tool_call_tokens: Sequence[list[int]], tokenizer: Tokenizer) -> list[ToolCall]:
    r"""Decode a list of tool call tokens into a list of tool calls.

    Dispatches to the version-specific decoder.

    Note:
        Each list of tool call tokens are expected to be in the format:
        - v2 to v7: `[TOOL_CALLS][{"name": "name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ...]`
        - v11+ without call ID: `[TOOL_CALLS]name[ARGS]{"arg1": "value1", "arg2": "value2"}`
        - v11+ with call ID: `[TOOL_CALLS]name[CALL_ID]call_id[ARGS]{"arg1": "value1", "arg2": "value2"}`

    Args:
        tool_call_tokens: A list of tool call token ID lists, one per tool call.
        tokenizer: The tokenizer to use for decoding.

    Returns:
        The list of decoded tool calls.

    Raises:
        ValueError: If the tokenizer version is v1 (no tool call support), or
            version-specific decoding errors.
    """
    tools_calls = []
    for tool_call in tool_call_tokens:
        if tokenizer.version == TokenizerVersion.v1:
            raise ValueError("Tool calls are not supported for tokenizer version v1.")
        elif tokenizer.version <= TokenizerVersion.v7:
            tools_calls.extend(_decode_tool_calls_v2_up_to_v7(tool_call, tokenizer))
        elif tokenizer.version == TokenizerVersion.v11 and tokenizer.get_special_token("[CALL_ID]") in tool_call:
            tools_calls.append(_decode_tool_call_v11_with_call_id(tool_call, tokenizer))
        else:
            tools_calls.append(_decode_tool_call_v11(tool_call, tokenizer))

    return tools_calls

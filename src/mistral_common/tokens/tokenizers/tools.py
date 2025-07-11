import json
from typing import List, Sequence, Tuple

from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenizer, TokenizerVersion


class InvalidToolCallError(ValueError):
    pass


class InvalidArgsToolCallError(InvalidToolCallError):
    pass


def _split_integer_list_by_value(list_: List[int], value: int) -> Tuple[List[int], ...]:
    r"""Split a list of integers by a given value.

    Args:
        list_: The list to split.
        value: The value to split the list by.

    Returns:
        A tuple of lists of integers.

    Examples:
        >>> _split_integer_list_by_value([1, 2, 3, 4, 5], 3)
        ([1, 2], [3, 4, 5])
        >>> _split_integer_list_by_value([1, 2, 3, 4, 5], 6)
        ([1, 2, 3, 4, 5],)
        >>> _split_integer_list_by_value([1, 2, 3, 4, 5], 1)
        ([1, 2, 3, 4, 5],)
        >>> _split_integer_list_by_value([1, 2, 3, 4, 5, 3, 5, 6, 7], 3)
        ([1, 2], [3, 4, 5], [3, 5, 6, 7])
    """
    result = [list_[0]]
    for i, item in enumerate(list_[1:]):
        if item == value:
            return (result, *_split_integer_list_by_value(list_[1 + i :], value))
        result.append(item)
    return (result,)


def _split_tokens_by_one_occurence_control_token(
    list_: List[int], tokenizer: Tokenizer, control_token: str
) -> Tuple[List[int], List[int]]:
    r"""Split a list of integers by a given control token.

    Raises:
        InvalidToolCallError: If the control token is not found in the list or if it is found more than once.
    """
    control_token_id = tokenizer.get_control_token(control_token)
    first, *rest = _split_integer_list_by_value(list_, control_token_id)
    if len(rest) == 0:
        raise InvalidToolCallError(f"Control token {control_token} not found in the list of tokens.")
    if len(rest) > 1:
        raise InvalidToolCallError(f"Control token {control_token} found more than once in the list of tokens.")
    return first, rest[0]


def split_content_and_tool_calls(tokens: List[int], tool_call_token_id: int) -> tuple[List[int], Tuple[List[int], ...]]:
    r"""Split the content and tool calls from a list of tokens.

    The content is the first sequence of tokens that does not start with the tool call token ID.
    The tool calls are the remaining sequences of tokens that start with the tool call token ID.

    Args:
        tokens: The list of tokens.
        tool_call_token_id: The token ID that indicates the start of a tool call.

    Returns:
        A tuple containing the content and tool calls.
    """
    maybe_content_and_tools_calls = _split_integer_list_by_value(tokens, tool_call_token_id)

    has_content = maybe_content_and_tools_calls[0][0] != tool_call_token_id
    if has_content:
        content_tokens = maybe_content_and_tools_calls[0]
        tools_calls_tokens = maybe_content_and_tools_calls[1:]
    else:
        content_tokens = []
        tools_calls_tokens = maybe_content_and_tools_calls

    return content_tokens, tools_calls_tokens


def _decode_tool_calls_v2_up_to_v7(tool_call_tokens: list[int], tokenizer: Tokenizer) -> List[ToolCall]:
    r"""Decode a list of tool call tokens into a list of tool calls for tokenizer versions v2 to v7.

    Note:
        Expects the tool call tokens to be in the format:

        `[TOOL_CALLS][{"id": "call_id", "name": "name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ...]`
        or

        `[TOOL_CALLS][{"name": "name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ...]`
    """
    tool_calls_list_string = tokenizer.decode(tool_call_tokens, special_token_policy=SpecialTokenPolicy.IGNORE)
    try:
        tool_calls_decoded_list = json.loads(tool_calls_list_string)
        for tool_call in tool_calls_decoded_list:
            # Check that the tool call arguments are dicts.
            if "arguments" not in tool_call or not isinstance(tool_call["arguments"], dict):
                raise InvalidArgsToolCallError("Invalid tool call arguments tokenization. Expected a dict.")
    except json.JSONDecodeError as e:
        raise InvalidToolCallError(
            "Invalid tool call tokenization. Expected a JSON list of tool calls.",
        ) from e

    if not isinstance(tool_calls_decoded_list, list):
        raise InvalidToolCallError("Invalid tool call tokenization. Expected a list of tool calls.")

    return [
        ToolCall(
            id=tool_call.get("id", "null"),
            function=FunctionCall(name=tool_call["name"], arguments=tool_call["arguments"]),
        )
        for tool_call in tool_calls_decoded_list
    ]


def _decode_tool_call_v11_with_call_id(tool_call_tokens: list[int], tokenizer: Tokenizer) -> ToolCall:
    r"""Decode a list of tool call tokens into a tool call for tokenizer version v11 with call ID.

    Note:
        Expects the tool call tokens to be in the format:

        `[TOOL_CALLS]name[CALL_ID]call_id[ARGS]{"arg1": "value1", "arg2": "value2"}`

    """
    name, call_id_and_args = _split_tokens_by_one_occurence_control_token(tool_call_tokens, tokenizer, "[CALL_ID]")

    call_id, args = _split_tokens_by_one_occurence_control_token(call_id_and_args, tokenizer, "[ARGS]")

    try:
        tool_call = ToolCall(
            id=tokenizer.decode(call_id),
            function=FunctionCall(
                name=tokenizer.decode(name),
                arguments=json.loads(tokenizer.decode(args, special_token_policy=SpecialTokenPolicy.IGNORE)),
            ),
        )
    except json.JSONDecodeError as e:
        raise InvalidArgsToolCallError("Invalid tokenized tool call arguments.") from e
    return tool_call


def _decode_tool_call_v11(tool_call_tokens: list[int], tokenizer: Tokenizer) -> ToolCall:
    r"""Decode a list of tool call tokens into a tool call for tokenizer version v11 without call ID.

    Note:
        Expects the tool call tokens to be in the format:

        `[TOOL_CALLS]name[ARGS]{"arg1": "value1", "arg2": "value2"}`
    """
    name, args = _split_tokens_by_one_occurence_control_token(tool_call_tokens, tokenizer, "[ARGS]")
    try:
        tool_call = ToolCall(
            id="",
            function=FunctionCall(
                name=tokenizer.decode(name, special_token_policy=SpecialTokenPolicy.IGNORE),
                arguments=json.loads(tokenizer.decode(args, special_token_policy=SpecialTokenPolicy.IGNORE)),
            ),
        )
    except json.JSONDecodeError as e:
        raise InvalidArgsToolCallError("Invalid tokenized tool call arguments.") from e
    return tool_call


def decode_tool_calls(tool_call_tokens: Sequence[list[int]], tokenizer: Tokenizer) -> List[ToolCall]:
    r"""Decode a list of tool call tokens into a list of tool calls.

    Note:
        Each list of tool call tokens are expected to be in the format:
        - v2 to v7: `[TOOL_CALLS][{"name": "name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ...]`
        - v11+ without call ID: `[TOOL_CALLS]name[ARGS]{"arg1": "value1", "arg2": "value2"}`
        - v11+ with call ID: `[TOOL_CALLS]name[CALL_ID]call_id[ARGS]{"arg1": "value1", "arg2": "value2"}`

    Args:
        tool_call_tokens: A list of lists of tokens.
        tokenizer: The tokenizer to use for decoding.

    Returns:
        The list of decoded tool calls.
    """
    tools_calls = []
    for tool_call in tool_call_tokens:
        if tokenizer.version == TokenizerVersion.v1:
            raise ValueError("Tool calls are not supported for tokenizer version v1.")
        elif tokenizer.version in [TokenizerVersion.v2, TokenizerVersion.v3, TokenizerVersion.v7]:
            tools_calls.extend(_decode_tool_calls_v2_up_to_v7(tool_call, tokenizer))
        elif tokenizer.version == TokenizerVersion.v11 and tokenizer.get_control_token("[CALL_ID]") in tool_call:
            tools_calls.append(_decode_tool_call_v11_with_call_id(tool_call, tokenizer))
        else:
            tools_calls.append(_decode_tool_call_v11(tool_call, tokenizer))

    return tools_calls

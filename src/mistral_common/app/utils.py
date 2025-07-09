from typing import List, Sequence, Tuple

from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, Tokenizer, TokenizerVersion


class InvalidtoolCallError(ValueError):
    pass


def split_integer_list_by_value(list_: List[int], value: int) -> Tuple[List[int], ...]:
    r"""Split a list of integers by a given value.

    Args:
        list_: The list to split.
        value: The value to split the list by.

    Returns:
        A tuple of lists of integers.

    Examples:
        >>> split_integer_list_by_value([1, 2, 3, 4, 5], 3)
        ([1, 2], [3, 4, 5])
        >>> split_integer_list_by_value([1, 2, 3, 4, 5], 6)
        ([1, 2, 3, 4, 5],)
        >>> split_integer_list_by_value([1, 2, 3, 4, 5], 1)
        ([1, 2, 3, 4, 5])
        >>> split_integer_list_by_value([1, 2, 3, 4, 5, 3, 5, 6, 7], 3)
        ([1, 2], [3, 4, 5], [3, 5, 6, 7])
    """
    result = [list_[0]]
    for i, item in enumerate(list_[1:]):
        if item == value:
            return (result, *split_integer_list_by_value(list_[1 + i :], value))
        result.append(item)
    return (result,)


def find_content_tool_calls(tokens: List[int], tool_call_token_id: int) -> tuple[List[int], Tuple[List[int], ...]]:
    r"""Find the content and tool calls in a list of tokens.

    The content is the first sequence of tokens that does not start with the tool call token ID.
    The tool calls are the remaining sequences of tokens that start with the tool call token ID.

    Args:
        tokens: The list of tokens.
        tool_call_token_id: The token ID that indicates the start of a tool call.
    Returns:
        A tuple containing the content and tool calls.
    """
    maybe_content_and_tools_calls = split_integer_list_by_value(tokens, tool_call_token_id)

    has_content = maybe_content_and_tools_calls[0][0] != tool_call_token_id
    if has_content:
        content_tokens = maybe_content_and_tools_calls[0]
        tools_calls_tokens = maybe_content_and_tools_calls[1:]
    else:
        content_tokens = []
        tools_calls_tokens = maybe_content_and_tools_calls

    return content_tokens, tools_calls_tokens


def decode_tool_call(tool_call_tokens: Sequence[list[int]], tokenizer: Tokenizer) -> List[ToolCall]:
    r"""Decode a list of tool call tokens into a list of tool calls."""
    tools_calls = []
    for tool_call in tool_call_tokens:
        if tokenizer.version in [
            TokenizerVersion.v1,
            TokenizerVersion.v2,
            TokenizerVersion.v3,
            TokenizerVersion.v7,
            TokenizerVersion.v11,
        ]:
            name, *call_id_and_args = split_integer_list_by_value(tool_call, tokenizer.get_control_token("[CALL_ID]"))
            if len(call_id_and_args) != 1:
                raise InvalidtoolCallError("Invalid tool call tokenization. Missing [CALL_ID] token.")
            call_id, *maybe_args = split_integer_list_by_value(
                call_id_and_args[0], tokenizer.get_control_token("[ARGS]")
            )
            if len(maybe_args) != 1:
                raise InvalidtoolCallError("Invalid tool call tokenization. Missing [ARGS] token.")
            args = maybe_args[0]
            tools_calls.append(
                ToolCall(
                    id=tokenizer.decode(call_id),
                    function=FunctionCall(
                        name=tokenizer.decode(name),
                        arguments=tokenizer.decode(args, special_token_policy=SpecialTokenPolicy.IGNORE),
                    ),
                )
            )
        else:
            name, *maybe_args = split_integer_list_by_value(tool_call, tokenizer.get_control_token("[ARGS]"))
            if len(maybe_args) != 1:
                raise InvalidtoolCallError("Invalid tool call tokenization. Missing [ARGS] token.")
            args = maybe_args[0]
            tools_calls.append(
                ToolCall(
                    id="",
                    function=FunctionCall(
                        name=tokenizer.decode(name, special_token_policy=SpecialTokenPolicy.IGNORE),
                        arguments=tokenizer.decode(args, special_token_policy=SpecialTokenPolicy.IGNORE),
                    ),
                )
            )
    return tools_calls

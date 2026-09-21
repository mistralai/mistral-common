from mistral_common.tokens.tokenizers.base import Tokenizer


def _split_integer_list_by_value(list_: list[int], value: int) -> tuple[list[int], ...]:
    r"""Split a list of integers into segments starting at each occurrence of value.

    The first segment collects elements before the first occurrence; each
    occurrence of value starts a new segment that begins with value itself.

    Args:
        list_: The list to split. Must be non-empty.
        value: The value that starts each new segment.

    Returns:
        A tuple of segments. Contains a single segment (the input list) when
        value never occurs.

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


def _split_tokens_by_one_occurrence_control_token(
    list_: list[int], tokenizer: Tokenizer, control_token: str
) -> tuple[list[int], list[int]]:
    r"""Split a token list into (before, after) around a single control token.

    Args:
        list_: The token IDs to split.
        tokenizer: Tokenizer used to resolve the control token to its ID.
        control_token: The special token string (e.g., "[TOOL_CALLS]") that
            must occur exactly once in list_.

    Returns:
        A tuple of (first, second): the tokens before the control token and
        the tokens after it.

    Raises:
        ValueError: If the control token is not found in the list or if it is
            found more than once.
    """
    control_token_id = tokenizer.get_special_token(control_token)
    first, *rest = _split_integer_list_by_value(list_, control_token_id)
    if len(rest) == 0:
        raise ValueError(f"Control token {control_token} not found in the list of tokens.")
    if len(rest) > 1:
        raise ValueError(f"Control token {control_token} found more than once in the list of tokens.")
    return first, rest[0]

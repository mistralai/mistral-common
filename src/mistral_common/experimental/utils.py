from typing import List, Tuple

from mistral_common.tokens.tokenizers.base import Tokenizer


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
        ValueError: If the control token is not found in the list or if it is found more than once.
    """
    control_token_id = tokenizer.get_control_token(control_token)
    first, *rest = _split_integer_list_by_value(list_, control_token_id)
    if len(rest) == 0:
        raise ValueError(f"Control token {control_token} not found in the list of tokens.")
    if len(rest) > 1:
        raise ValueError(f"Control token {control_token} found more than once in the list of tokens.")
    return first, rest[0]

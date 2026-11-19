import pytest

from mistral_common.experimental.utils import (
    _split_integer_list_by_value,
    _split_tokens_by_one_occurrence_control_token,
)
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.utils.special_tokens import get_special_tokens, quick_vocab


@pytest.mark.parametrize(
    ("list_", "value", "expected"),
    [
        pytest.param([1, 2, 3, 4, 5], 3, ([1, 2], [3, 4, 5]), id="one-split"),
        pytest.param([1, 2, 3, 4, 5], 6, ([1, 2, 3, 4, 5],), id="value-absent"),
        pytest.param([1, 2, 3, 4, 5], 1, ([1, 2, 3, 4, 5],), id="value-at-start"),
        pytest.param([1, 2, 3, 4, 5, 3, 5, 6, 7], 3, ([1, 2], [3, 4, 5], [3, 5, 6, 7]), id="multiple-splits"),
    ],
)
def test_split_integer_list_by_value(list_: list[int], value: int, expected: tuple[list[int], ...]) -> None:
    assert _split_integer_list_by_value(list_, value) == expected


@pytest.fixture(scope="module")
def v7_tokenizer() -> Tekkenizer:
    return Tekkenizer(
        quick_vocab([b"a", b"b"]),
        special_tokens=get_special_tokens(TokenizerVersion.v7),
        pattern=r".+",
        vocab_size=256 + 100,
        num_special_tokens=100,
        version=TokenizerVersion.v7,
    )


def test_split_tokens_by_one_occurrence_control_token(v7_tokenizer: Tekkenizer) -> None:
    tool_calls_id = v7_tokenizer.get_special_token("[TOOL_CALLS]")
    tokens = [1, 2, tool_calls_id, 3, 4]
    # The second segment starts with the control token itself.
    assert _split_tokens_by_one_occurrence_control_token(tokens, v7_tokenizer, "[TOOL_CALLS]") == (
        [1, 2],
        [tool_calls_id, 3, 4],
    )


def test_split_tokens_by_one_occurrence_control_token_not_found(v7_tokenizer: Tekkenizer) -> None:
    with pytest.raises(ValueError, match=r"Control token \[TOOL_CALLS\] not found in the list of tokens."):
        _split_tokens_by_one_occurrence_control_token([1, 2, 3], v7_tokenizer, "[TOOL_CALLS]")


def test_split_tokens_by_one_occurrence_control_token_duplicated(v7_tokenizer: Tekkenizer) -> None:
    tool_calls_id = v7_tokenizer.get_special_token("[TOOL_CALLS]")
    with pytest.raises(ValueError, match=r"Control token \[TOOL_CALLS\] found more than once in the list of tokens."):
        _split_tokens_by_one_occurrence_control_token(
            [1, tool_calls_id, 2, tool_calls_id], v7_tokenizer, "[TOOL_CALLS]"
        )

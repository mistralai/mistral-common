import pytest

from mistral_common.experimental.think import _split_content_and_think_chunks


@pytest.mark.parametrize(
    "tokens, expected",
    [
        ([], []),
        ([1, 2, 3, 4, 5], [([1, 2, 3, 4, 5], False)]),
        ([1, 2, 3, 100, 4, 5, 200, 6, 7, 8], [([1, 2, 3], False), ([100, 4, 5, 200], True), ([6, 7, 8], False)]),
        (
            [100, 1, 2, 3, 200, 4, 5, 6, 100, 7, 8, 9, 200],
            [([100, 1, 2, 3, 200], True), ([4, 5, 6], False), ([100, 7, 8, 9, 200], True)],
        ),
        ([100, 1, 2, 3, 200, 100, 4, 5, 6, 200], [([100, 1, 2, 3, 200], True), ([100, 4, 5, 6, 200], True)]),
        ([100, 1, 2, 3, 200, 100, 5, 6, 7], [([100, 1, 2, 3, 200], True), ([100, 5, 6, 7], True)]),
    ],
)
def test_split_content_and_think_chunks(tokens: list[int], expected: list[tuple[list[int], bool]]) -> None:
    content_or_think_chunks = _split_content_and_think_chunks(tokens, 100, 200)
    assert content_or_think_chunks == expected


@pytest.mark.parametrize(
    "tokens, error_message",
    [
        ([100, 1, 2, 3, 100, 200, 100, 5, 6, 7], r"Nested think chunks are not allowed."),
        ([100, 1, 2, 3, 200, 200, 5, 6, 7, 100], r"End think token found without a begin think token."),
    ],
)
def test_split_content_and_think_chunks_error(tokens: list[int], error_message: str) -> None:
    with pytest.raises(ValueError, match=error_message):
        _split_content_and_think_chunks(tokens, 100, 200)

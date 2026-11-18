import pytest

from mistral_common.experimental.think import _split_content_and_think_chunks

_BEGIN_THINK = 100
_END_THINK = 200


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        pytest.param([], [], id="empty"),
        pytest.param([1, 2, 3, 4, 5], [([1, 2, 3, 4, 5], False)], id="plain-content"),
        pytest.param(
            [1, 2, 3, 100, 4, 5, 200, 6, 7, 8],
            [([1, 2, 3], False), ([100, 4, 5, 200], True), ([6, 7, 8], False)],
            id="content-think-content",
        ),
        pytest.param(
            [100, 1, 2, 3, 200, 4, 5, 6, 100, 7, 8, 9, 200],
            [([100, 1, 2, 3, 200], True), ([4, 5, 6], False), ([100, 7, 8, 9, 200], True)],
            id="think-content-think",
        ),
        pytest.param(
            [100, 1, 2, 3, 200, 100, 4, 5, 6, 200],
            [([100, 1, 2, 3, 200], True), ([100, 4, 5, 6, 200], True)],
            id="adjacent-think-chunks",
        ),
        pytest.param(
            [100, 1, 2, 3, 200, 100, 5, 6, 7],
            [([100, 1, 2, 3, 200], True), ([100, 5, 6, 7], True)],
            id="unclosed-think-chunk",
        ),
    ],
)
def test_split_content_and_think_chunks(tokens: list[int], expected: list[tuple[list[int], bool]]) -> None:
    assert _split_content_and_think_chunks(tokens, _BEGIN_THINK, _END_THINK) == expected


@pytest.mark.parametrize(
    ("tokens", "error_message"),
    [
        pytest.param(
            [100, 1, 2, 3, 100, 200, 100, 5, 6, 7],
            r"Nested think chunks are not allowed.",
            id="nested-think-chunks",
        ),
        pytest.param(
            [100, 1, 2, 3, 200, 200, 5, 6, 7, 100],
            r"End think token found without a begin think token.",
            id="end-without-begin",
        ),
    ],
)
def test_split_content_and_think_chunks_error(tokens: list[int], error_message: str) -> None:
    with pytest.raises(ValueError, match=error_message):
        _split_content_and_think_chunks(tokens, _BEGIN_THINK, _END_THINK)

from mistral_common.experimental.utils import _split_integer_list_by_value


def test_split_integer_list_by_value() -> None:
    # Test 1: One split
    assert _split_integer_list_by_value([1, 2, 3, 4, 5], 3) == ([1, 2], [3, 4, 5])

    # Test 2: No value
    assert _split_integer_list_by_value([1, 2, 3, 4, 5], 6) == ([1, 2, 3, 4, 5],)

    # Test 3: No split
    assert _split_integer_list_by_value([1, 2, 3, 4, 5], 1) == ([1, 2, 3, 4, 5],)

    # Test 4: Multiple splits
    assert _split_integer_list_by_value([1, 2, 3, 4, 5, 3, 5, 6, 7], 3) == ([1, 2], [3, 4, 5], [3, 5, 6, 7])

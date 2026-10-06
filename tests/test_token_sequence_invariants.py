import pytest

def pad_or_truncate_sequence(tokens: list[int], max_len: int, pad_id: int = 0) -> list[int]:
    if max_len <= 0:
        return []
    if len(tokens) >= max_len:
        return tokens[:max_len]
    return tokens + [pad_id] * (max_len - len(tokens))

def test_truncation():
    assert pad_or_truncate_sequence([1, 2, 3, 4, 5], 3) == [1, 2, 3]

def test_padding():
    assert pad_or_truncate_sequence([1, 2], 5, pad_id=-1) == [1, 2, -1, -1, -1]

def test_exact_length():
    assert pad_or_truncate_sequence([1, 2, 3], 3) == [1, 2, 3]

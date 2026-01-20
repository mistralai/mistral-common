import pytest

from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy


def test_special_token_policy_backward_compatibility() -> None:
    assert SpecialTokenPolicy(0) == SpecialTokenPolicy.IGNORE
    assert SpecialTokenPolicy(1) == SpecialTokenPolicy.KEEP
    assert SpecialTokenPolicy(2) == SpecialTokenPolicy.RAISE

    with pytest.raises(ValueError, match=r"3 is not a valid SpecialTokenPolicy"):
        SpecialTokenPolicy(3) is None
    with pytest.raises(ValueError, match=r"'invalid' is not a valid SpecialTokenPolicy"):
        SpecialTokenPolicy("invalid") is None

import pytest

from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy


@pytest.mark.parametrize(
    "policy,expected",
    [
        (SpecialTokenPolicy.IGNORE, SpecialTokenPolicy.IGNORE),
        ("ignore", SpecialTokenPolicy.IGNORE),
        (SpecialTokenPolicy.KEEP, SpecialTokenPolicy.KEEP),
        ("keep", SpecialTokenPolicy.KEEP),
        (SpecialTokenPolicy.RAISE, SpecialTokenPolicy.RAISE),
        ("raise", SpecialTokenPolicy.RAISE),
    ],
)
def test_special_token_policy_validate(policy: SpecialTokenPolicy | str, expected: SpecialTokenPolicy) -> None:
    assert SpecialTokenPolicy.validate(policy) == expected


@pytest.mark.parametrize("policy", [3, "invalid"])
def test_special_token_policy_validate_rejects_invalid_values(policy: object) -> None:
    with pytest.raises(ValueError, match=r"Invalid `special_token_policy`"):
        SpecialTokenPolicy.validate(policy)  # type: ignore[arg-type]

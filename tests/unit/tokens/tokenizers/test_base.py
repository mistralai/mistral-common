import operator
from typing import Callable

import pytest

from mistral_common.tokens.tokenizers.base import TokenizerVersion

ALL_VERSIONS = list(TokenizerVersion)

# Expected values of each externally-consumed property, keyed by version.
_EXPECTED: dict[str, dict[TokenizerVersion, bool]] = {
    "has_tools": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: True,
        TokenizerVersion.v3: True,
        TokenizerVersion.v7: True,
        TokenizerVersion.v11: True,
        TokenizerVersion.v13: True,
        TokenizerVersion.v15: True,
    },
    "uses_pre_v11_tool_format": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: True,
        TokenizerVersion.v3: True,
        TokenizerVersion.v7: True,
        TokenizerVersion.v11: False,
        TokenizerVersion.v13: False,
        TokenizerVersion.v15: False,
    },
    "supports_model_settings": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: False,
        TokenizerVersion.v3: False,
        TokenizerVersion.v7: False,
        TokenizerVersion.v11: False,
        TokenizerVersion.v13: False,
        TokenizerVersion.v15: True,
    },
}

_VERSIONED_PROPERTIES = list(_EXPECTED)


class TestTokenizerVersionProperties:
    @pytest.mark.parametrize("version", ALL_VERSIONS)
    @pytest.mark.parametrize("prop", _VERSIONED_PROPERTIES)
    def test_property_matches_expected_value(self, version: TokenizerVersion, prop: str) -> None:
        assert getattr(version, prop) is _EXPECTED[prop][version]


class TestTokenizerVersionComparisons:
    @pytest.mark.parametrize("other", [TokenizerVersion.v11, "v11"])
    def test_orders_by_version_number_not_by_string(self, other: str | TokenizerVersion) -> None:
        # As plain strings "v2" > "v11"; versions must compare by number.
        assert TokenizerVersion.v2 < other
        assert TokenizerVersion.v2 <= other
        assert not TokenizerVersion.v2 > other
        assert not TokenizerVersion.v2 >= other

    @pytest.mark.parametrize("other", [TokenizerVersion.v7, "v7"])
    def test_equal_versions(self, other: str | TokenizerVersion) -> None:
        assert TokenizerVersion.v7 <= other
        assert TokenizerVersion.v7 >= other
        assert not TokenizerVersion.v7 < other
        assert not TokenizerVersion.v7 > other

    def test_str_on_the_left(self) -> None:
        assert "v2" < TokenizerVersion.v11
        assert "v11" >= TokenizerVersion.v2
        assert not "v11" <= TokenizerVersion.v2

    @pytest.mark.parametrize("op", [operator.lt, operator.le, operator.gt, operator.ge])
    def test_unknown_version_string_raises(self, op: Callable[[object, object], bool]) -> None:
        with pytest.raises(ValueError, match="'v99' is not a valid TokenizerVersion"):
            op(TokenizerVersion.v7, "v99")

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

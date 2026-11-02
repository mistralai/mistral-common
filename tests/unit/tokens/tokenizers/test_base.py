import pytest

from mistral_common.tokens.tokenizers.base import TokenizerVersion

ALL_VERSIONS = list(TokenizerVersion)

# Expected values of each pure-version property, keyed by version.
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
    "uses_system_prompt_tokens": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: False,
        TokenizerVersion.v3: False,
        TokenizerVersion.v7: True,
        TokenizerVersion.v11: True,
        TokenizerVersion.v13: True,
        TokenizerVersion.v15: True,
    },
    "tools_prior_last_user": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: True,
        TokenizerVersion.v3: True,
        TokenizerVersion.v7: True,
        TokenizerVersion.v11: True,
        TokenizerVersion.v13: False,
        TokenizerVersion.v15: False,
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
    "uses_v13_tool_format": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: False,
        TokenizerVersion.v3: False,
        TokenizerVersion.v7: False,
        TokenizerVersion.v11: False,
        TokenizerVersion.v13: True,
        TokenizerVersion.v15: True,
    },
    "uses_tool_content_format": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: False,
        TokenizerVersion.v3: False,
        TokenizerVersion.v7: True,
        TokenizerVersion.v11: True,
        TokenizerVersion.v13: False,
        TokenizerVersion.v15: False,
    },
    "uses_call_id_in_tool_calls": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: False,
        TokenizerVersion.v3: False,
        TokenizerVersion.v7: False,
        TokenizerVersion.v11: True,
        TokenizerVersion.v13: False,
        TokenizerVersion.v15: False,
    },
    "uses_plain_thinking": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: False,
        TokenizerVersion.v3: False,
        TokenizerVersion.v7: False,
        TokenizerVersion.v11: True,
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
    "forbids_assistant_content_with_tools": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: True,
        TokenizerVersion.v3: True,
        TokenizerVersion.v7: False,
        TokenizerVersion.v11: False,
        TokenizerVersion.v13: False,
        TokenizerVersion.v15: False,
    },
    "validates_assistant_non_empty": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: False,
        TokenizerVersion.v3: True,
        TokenizerVersion.v7: True,
        TokenizerVersion.v11: True,
        TokenizerVersion.v13: True,
        TokenizerVersion.v15: True,
    },
    "tool_supports_multimodal": {
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

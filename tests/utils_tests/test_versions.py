import pytest

from mistral_common.tokens.tokenizers.base import TokenizerVersion
from tests.utils.versions import (
    ALL_VERSIONS,
    AUDIO_VERSIONS,
    MODEL_SETTINGS_VERSIONS,
    SPM_VERSIONS,
    TEKKEN_VERSIONS,
    THINK_VERSIONS,
    TestConfig,
    config_id,
)


class TestAllVersions:
    def test_all_versions_equals_enum_tuple(self) -> None:
        assert ALL_VERSIONS == tuple(TokenizerVersion)

    def test_all_versions_is_tuple(self) -> None:
        assert isinstance(ALL_VERSIONS, tuple)

    def test_all_versions_non_empty(self) -> None:
        assert len(ALL_VERSIONS) > 0


# Explicit membership, enumerated by hand from `TokenizerVersion` rather than derived from the
# comprehensions under test in `tests/utils/versions.py`. Adding a new `TokenizerVersion` member
# must fail these tests until someone consciously classifies it here.
_EXPECTED_SPM_VERSIONS = frozenset({TokenizerVersion.v1, TokenizerVersion.v2, TokenizerVersion.v3, TokenizerVersion.v7})
_EXPECTED_TEKKEN_VERSIONS = frozenset(
    {TokenizerVersion.v3, TokenizerVersion.v7, TokenizerVersion.v11, TokenizerVersion.v13, TokenizerVersion.v15}
)
_EXPECTED_AUDIO_VERSIONS = frozenset(
    {TokenizerVersion.v7, TokenizerVersion.v11, TokenizerVersion.v13, TokenizerVersion.v15}
)
_EXPECTED_THINK_VERSIONS = frozenset({TokenizerVersion.v13, TokenizerVersion.v15})
_EXPECTED_MODEL_SETTINGS_VERSIONS = frozenset({TokenizerVersion.v15})


class TestSpmVersions:
    @pytest.mark.parametrize("version", ALL_VERSIONS, ids=lambda version: version.value)
    def test_spm_versions_membership(self, version: TokenizerVersion) -> None:
        assert (version in SPM_VERSIONS) == (version in _EXPECTED_SPM_VERSIONS)

    def test_spm_versions_is_tuple(self) -> None:
        assert isinstance(SPM_VERSIONS, tuple)


class TestTekkenVersions:
    @pytest.mark.parametrize("version", ALL_VERSIONS, ids=lambda version: version.value)
    def test_tekken_versions_membership(self, version: TokenizerVersion) -> None:
        assert (version in TEKKEN_VERSIONS) == (version in _EXPECTED_TEKKEN_VERSIONS)

    def test_tekken_versions_is_tuple(self) -> None:
        assert isinstance(TEKKEN_VERSIONS, tuple)


class TestAudioVersions:
    @pytest.mark.parametrize("version", ALL_VERSIONS, ids=lambda version: version.value)
    def test_audio_versions_membership(self, version: TokenizerVersion) -> None:
        assert (version in AUDIO_VERSIONS) == (version in _EXPECTED_AUDIO_VERSIONS)

    def test_audio_versions_is_tuple(self) -> None:
        assert isinstance(AUDIO_VERSIONS, tuple)


class TestThinkVersions:
    @pytest.mark.parametrize("version", ALL_VERSIONS, ids=lambda version: version.value)
    def test_think_versions_membership(self, version: TokenizerVersion) -> None:
        assert (version in THINK_VERSIONS) == (version in _EXPECTED_THINK_VERSIONS)

    def test_think_versions_is_tuple(self) -> None:
        assert isinstance(THINK_VERSIONS, tuple)


class TestModelSettingsVersions:
    @pytest.mark.parametrize("version", ALL_VERSIONS, ids=lambda version: version.value)
    def test_model_settings_versions_membership(self, version: TokenizerVersion) -> None:
        assert (version in MODEL_SETTINGS_VERSIONS) == (version in _EXPECTED_MODEL_SETTINGS_VERSIONS)

    def test_model_settings_versions_is_tuple(self) -> None:
        assert isinstance(MODEL_SETTINGS_VERSIONS, tuple)


class TestConfigId:
    def test_config_id_version_only(self) -> None:
        config = TestConfig(version=TokenizerVersion.v13)
        assert config_id(config) == "v13"

    def test_config_id_with_image_and_think(self) -> None:
        config = TestConfig(version=TokenizerVersion.v13, image=True, think=True)
        assert config_id(config) == "v13_img_think"

    def test_config_id_with_spm(self) -> None:
        config = TestConfig(version=TokenizerVersion.v3, spm=True)
        assert config_id(config) == "v3_spm"

    def test_config_id_with_audio(self) -> None:
        config = TestConfig(version=TokenizerVersion.v7, audio=True)
        assert config_id(config) == "v7_aud"

    def test_config_id_with_plain_think(self) -> None:
        config = TestConfig(version=TokenizerVersion.v11, plain_think=True)
        assert config_id(config) == "v11_plain_think"

    def test_config_id_suffix_order(self) -> None:
        config = TestConfig(
            version=TokenizerVersion.v7,
            spm=True,
            image=True,
            audio=True,
            think=True,
            plain_think=True,
        )
        assert config_id(config) == "v7_spm_img_aud_think_plain_think"

    @pytest.mark.parametrize(
        "config_a,config_b",
        [
            (
                TestConfig(version=TokenizerVersion.v13, image=True),
                TestConfig(version=TokenizerVersion.v13, audio=True),
            ),
            (
                TestConfig(version=TokenizerVersion.v13),
                TestConfig(version=TokenizerVersion.v7),
            ),
        ],
    )
    def test_config_id_collision_free(self, config_a: TestConfig, config_b: TestConfig) -> None:
        assert config_id(config_a) != config_id(config_b)


class TestTestConfig:
    def test_not_collected_by_pytest(self) -> None:
        assert TestConfig.__test__ is False

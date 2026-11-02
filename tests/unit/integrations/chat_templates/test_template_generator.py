import pytest

from mistral_common.integrations.chat_templates.template_generator import TemplateConfig
from mistral_common.tokens.tokenizers.base import TokenizerVersion


class TestSpmDependentVersionFacts:
    def test_validates_assistant_non_empty_v3_spm_exception(self) -> None:
        # The version-level property assumes Tekken; the v3 spm variant does not validate.
        tekken = TemplateConfig(version=TokenizerVersion.v3, spm=False)
        spm = TemplateConfig(version=TokenizerVersion.v3, spm=True)

        assert tekken.version.validates_assistant_non_empty is True
        assert spm.version.validates_assistant_non_empty is True

    @pytest.mark.parametrize("version", [v for v in TokenizerVersion if v < TokenizerVersion.v11])
    def test_spm_does_not_change_other_version_facts(self, version: TokenizerVersion) -> None:
        # spm is rejected for v11+, so only pre-v11 versions have an spm variant.
        tekken = TemplateConfig(version=version)
        spm = TemplateConfig(version=version, spm=True)

        assert spm.version.has_tools == tekken.version.has_tools
        assert spm.version.uses_v13_tool_format == tekken.version.uses_v13_tool_format

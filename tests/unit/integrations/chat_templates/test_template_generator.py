import pytest

from mistral_common.integrations.chat_templates.template_generator import TemplateConfig
from mistral_common.tokens.tokenizers.base import TokenizerVersion


class TestSpmDependentValidation:
    def test_validates_assistant_non_empty_v3_spm_exception(self) -> None:
        # The v3 spm variant does not validate assistant non-emptiness; the
        # condition lives inline in the generator, so this only guards the
        # config flags the condition reads.
        tekken = TemplateConfig(version=TokenizerVersion.v3, spm=False)
        spm = TemplateConfig(version=TokenizerVersion.v3, spm=True)

        assert tekken.spm is False
        assert spm.spm is True

    @pytest.mark.parametrize("version", [v for v in TokenizerVersion if v < TokenizerVersion.v11])
    def test_spm_variant_is_constructible_pre_v11(self, version: TokenizerVersion) -> None:
        # spm is rejected for v11+, so only pre-v11 versions have an spm variant.
        config = TemplateConfig(version=version, spm=True)

        assert config.spm is True
        assert config.version == version

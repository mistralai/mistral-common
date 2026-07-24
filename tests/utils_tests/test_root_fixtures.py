from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.utils.versions import SPM_VERSIONS, TEKKEN_VERSIONS


class TestTekkenizerFixture:
    def test_is_tekkenizer(self, tekkenizer: Tekkenizer) -> None:
        assert isinstance(tekkenizer, Tekkenizer)

    def test_version_in_tekken_versions(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.version in TEKKEN_VERSIONS


class TestSpmFixture:
    def test_is_sentencepiece(self, spm: SentencePieceTokenizer) -> None:
        assert isinstance(spm, SentencePieceTokenizer)

    def test_version_in_spm_versions(self, spm: SentencePieceTokenizer) -> None:
        assert spm.version in SPM_VERSIONS

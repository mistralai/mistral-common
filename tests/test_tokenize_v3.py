from pathlib import Path
from tempfile import NamedTemporaryFile

from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer, is_sentencepiece


def test_is_spm() -> None:
    # this is valid
    for suffix in list(TokenizerVersion.__members__) + ["v3m1"]:
        with NamedTemporaryFile(suffix=".model." + suffix) as f:
            assert is_sentencepiece(f.name)

    with NamedTemporaryFile(suffix=".model") as f:
        assert is_sentencepiece(f.name)

    # this is not valid
    with NamedTemporaryFile(suffix=".model.vx") as f:
        assert not is_sentencepiece(f.name)


def test_spm_version() -> None:
    directory = Path(__file__).parent.parent / "src" / "mistral_common" / "data"

    for file in directory.iterdir():
        if not file.is_file() or str(file).endswith(".json"):
            continue
        suffix = file.suffix[1:].split("m")[0]
        if suffix == "model":
            assert SentencePieceTokenizer(str(file)).version == TokenizerVersion.v1
        else:
            assert SentencePieceTokenizer(str(file)).version == TokenizerVersion(suffix)

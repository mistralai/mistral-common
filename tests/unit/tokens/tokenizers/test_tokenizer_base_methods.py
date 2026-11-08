import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import mistral_common
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens, Tokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import quick_vocab

_SP_MODEL = Path(mistral_common.__file__).parent / "data" / "tokenizer.model.v1"


def _tekkenizer() -> Tekkenizer:
    special_tokens = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
    vocab = quick_vocab([b"hello"])
    return Tekkenizer(
        vocab=vocab,
        special_tokens=special_tokens,
        pattern=r".+",
        vocab_size=len(vocab) + len(special_tokens),
        num_special_tokens=len(special_tokens),
        version=TokenizerVersion.v3,
    )


@pytest.fixture(scope="module", params=["tekken", "sentencepiece"])
def tokenizer(request: pytest.FixtureRequest) -> Tokenizer:
    if request.param == "tekken":
        return _tekkenizer()
    return SentencePieceTokenizer(model_path=_SP_MODEL)


@pytest.mark.parametrize(
    "module", ["mistral_common.tokens.tokenizers.tekken", "mistral_common.tokens.tokenizers.sentencepiece"]
)
def test_import_leaves_warning_filters_unchanged(module: str) -> None:
    code = (
        "import warnings\n"
        "import mistral_common.tokens.tokenizers.base\n"
        "before = list(warnings.filters)\n"
        f"import {module}\n"
        "added = [f for f in warnings.filters if f not in before]\n"
        "assert warnings.filters == before, added\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_get_control_token_warns_once_at_caller(tokenizer: Tokenizer) -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert tokenizer.get_control_token(SpecialTokens.eos.value) == tokenizer.eos_id
        assert tokenizer.get_control_token(SpecialTokens.bos.value) == tokenizer.bos_id

    assert len(record) == 1
    assert record[0].category is FutureWarning
    assert "`get_control_token` is deprecated. Use `get_special_token` instead." in str(record[0].message)
    assert record[0].filename == __file__


def test_get_control_token_honors_command_line_error_filter() -> None:
    code = (
        "import mistral_common.tokens.tokenizers.tekken\n"
        "from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer\n"
        f"SentencePieceTokenizer(model_path={str(_SP_MODEL)!r}).get_control_token('</s>')\n"
    )
    result = subprocess.run([sys.executable, "-W", "error::FutureWarning", "-c", code], capture_output=True, text=True)
    assert result.returncode != 0
    assert "FutureWarning: `get_control_token` is deprecated." in result.stderr


def test_sentencepiece_control_tokens_warns_once_at_caller() -> None:
    sp_tokenizer = SentencePieceTokenizer(model_path=_SP_MODEL)
    other = SentencePieceTokenizer(model_path=_SP_MODEL)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert sp_tokenizer._control_tokens == sp_tokenizer.special_ids
        assert other._control_tokens == other.special_ids

    assert len(record) == 1
    assert record[0].category is FutureWarning
    assert "`_control_tokens` is deprecated." in str(record[0].message)
    assert record[0].filename == __file__


@pytest.mark.parametrize("tokenizer_cls", [Tekkenizer, SentencePieceTokenizer])
@pytest.mark.parametrize("name", ["get_control_token", "_to_string"])
def test_shared_methods_are_inherited_from_tokenizer(tokenizer_cls: type[Tokenizer], name: str) -> None:
    assert name not in vars(tokenizer_cls)
    assert getattr(tokenizer_cls, name) is getattr(Tokenizer, name)


def test_shared_methods_match_tokenizer_behavior(tokenizer: Tokenizer) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for token in (SpecialTokens.bos.value, SpecialTokens.eos.value):
            assert tokenizer.get_control_token(token) == tokenizer.get_special_token(token)

    tokens = [tokenizer.bos_id, *tokenizer.encode("hello", bos=False, eos=False), tokenizer.eos_id]
    assert tokenizer._to_string(tokens) == tokenizer.decode(tokens, special_token_policy=SpecialTokenPolicy.KEEP)
    assert tokenizer._to_string(tokens).startswith("<s>")
    assert tokenizer._to_string(tokens).endswith("</s>")

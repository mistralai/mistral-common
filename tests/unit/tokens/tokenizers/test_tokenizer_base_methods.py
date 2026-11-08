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

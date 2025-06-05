import pytest

from mistral_common.tokens.tokenizers.base import FIMRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


@pytest.fixture()
def tokenizer() -> MistralTokenizer:
    return MistralTokenizer.v3()


def test_encode_fim(tokenizer: MistralTokenizer) -> None:
    tokenized = tokenizer.encode_fim(FIMRequest(prompt="def f(", suffix="return a + b"))
    assert tokenized.text == "<s>[SUFFIX]return▁a▁+▁b[PREFIX]▁def▁f("

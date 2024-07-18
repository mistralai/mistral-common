import pytest
from mistral_common.exceptions import TokenizerException
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import (
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
)


class TestMistralToknizer:
    def test_from_model(self) -> None:
        assert isinstance(MistralTokenizer.from_model("open-mistral-7b").instruct_tokenizer, InstructTokenizerV1)
        assert isinstance(MistralTokenizer.from_model("open-mixtral-8x7b").instruct_tokenizer, InstructTokenizerV1)
        assert isinstance(MistralTokenizer.from_model("mistral-embed").instruct_tokenizer, InstructTokenizerV1)
        assert isinstance(MistralTokenizer.from_model("mistral-small").instruct_tokenizer, InstructTokenizerV2)
        assert isinstance(MistralTokenizer.from_model("mistral-large").instruct_tokenizer, InstructTokenizerV2)
        assert isinstance(MistralTokenizer.from_model("open-mixtral-8x22b").instruct_tokenizer, InstructTokenizerV3)

        # Test partial matches
        assert isinstance(MistralTokenizer.from_model("mistral-small-latest").instruct_tokenizer, InstructTokenizerV2)
        assert isinstance(MistralTokenizer.from_model("mistral-small-240401").instruct_tokenizer, InstructTokenizerV2)

        with pytest.raises(TokenizerException):
            MistralTokenizer.from_model("unknown-model")

    def test_decode(self) -> None:
        tokenizer = MistralTokenizer.v3()

        prompt = "This is a complicated te$t, ain't it?"

        for bos, eos in [[False, False], [True, True]]:
            encoded = tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=bos, eos=eos)

            assert tokenizer.decode(encoded) == prompt
            assert tokenizer.instruct_tokenizer.decode(encoded) == prompt

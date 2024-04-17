import pytest
from mistral_common.exceptions import TokenizerException
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceInstructTokenizerV1,
    SentencePieceInstructTokenizerV2,
    SentencePieceInstructTokenizerV3,
)


class TestMistralToknizer:
    def test_from_model(self) -> None:
        assert isinstance(
            MistralTokenizer.from_model("open-mistral-7B").instruct_tokenizer, SentencePieceInstructTokenizerV1
        )
        assert isinstance(
            MistralTokenizer.from_model("open-mixtral-8x7B").instruct_tokenizer, SentencePieceInstructTokenizerV1
        )
        assert isinstance(
            MistralTokenizer.from_model("mistral-embed").instruct_tokenizer, SentencePieceInstructTokenizerV1
        )
        assert isinstance(
            MistralTokenizer.from_model("mistral-small").instruct_tokenizer, SentencePieceInstructTokenizerV2
        )
        assert isinstance(
            MistralTokenizer.from_model("mistral-large").instruct_tokenizer, SentencePieceInstructTokenizerV2
        )
        assert isinstance(
            MistralTokenizer.from_model("open-mixtral-8x22B").instruct_tokenizer, SentencePieceInstructTokenizerV3
        )

        # Test partial matches
        assert isinstance(
            MistralTokenizer.from_model("mistral-small-latest").instruct_tokenizer, SentencePieceInstructTokenizerV2
        )
        assert isinstance(
            MistralTokenizer.from_model("mistral-small-240401").instruct_tokenizer, SentencePieceInstructTokenizerV2
        )

        with pytest.raises(TokenizerException):
            MistralTokenizer.from_model("unknown-model")

from typing import Optional, Union
from unittest.mock import patch

import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


class TestMistralToknizer:
    def test_from_model(self) -> None:
        assert isinstance(MistralTokenizer.from_model("open-mistral-7B").instruct_tokenizer, InstructTokenizerV1)
        assert isinstance(MistralTokenizer.from_model("open-mixtral-8x7B").instruct_tokenizer, InstructTokenizerV1)
        assert isinstance(MistralTokenizer.from_model("mistral-embed").instruct_tokenizer, InstructTokenizerV1)
        assert isinstance(MistralTokenizer.from_model("mistral-small").instruct_tokenizer, InstructTokenizerV2)
        assert isinstance(MistralTokenizer.from_model("mistral-large").instruct_tokenizer, InstructTokenizerV2)
        assert isinstance(MistralTokenizer.from_model("open-mixtral-8x22B").instruct_tokenizer, InstructTokenizerV3)

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

    def test_from_hf_hub(self) -> None:
        def _mocked_hf_download(
            repo_id: str, token: Optional[Union[bool, str]] = None, revision: Optional[str] = None
        ) -> str:
            if repo_id == "mistralai/Mistral-7B-Instruct-v0.1":
                return str(MistralTokenizer._data_path() / "tokenizer.model.v1")
            elif repo_id == "mistralai/Pixtral-Large-Instruct-2411":
                return str(MistralTokenizer._data_path() / "tekken_240911.json")
            else:
                raise ValueError(f"Unknown repo_id: {repo_id}")

        with patch("mistral_common.tokens.tokenizers.mistral.download_tokenizer_from_hf_hub", _mocked_hf_download):
            tokenizer = MistralTokenizer.from_hf_hub("mistralai/Mistral-7B-Instruct-v0.1")
            assert isinstance(tokenizer.instruct_tokenizer, InstructTokenizerV1)

            tokenizer = MistralTokenizer.from_hf_hub("mistralai/Pixtral-Large-Instruct-2411")
            assert isinstance(tokenizer.instruct_tokenizer, InstructTokenizerV3)

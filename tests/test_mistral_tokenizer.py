import multiprocessing
from typing import List, Optional, Tuple, Union
from unittest.mock import patch

import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

SPM_SPECIAL_WHITESPACE = "▁"
SPM_WHITESPACE = "▁"


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

    @pytest.mark.parametrize(
        ["special_token_policy", "is_tekken"],
        [
            (None, False),
            (None, True),
            (SpecialTokenPolicy.IGNORE, False),
            (SpecialTokenPolicy.IGNORE, True),
            (SpecialTokenPolicy.KEEP, False),
            (SpecialTokenPolicy.KEEP, True),
            (SpecialTokenPolicy.RAISE, False),
            (SpecialTokenPolicy.RAISE, True),
        ],
    )
    def test_decode(self, special_token_policy: Optional[SpecialTokenPolicy], is_tekken: bool) -> None:
        tokenizer = MistralTokenizer.v3(is_tekken=is_tekken)

        prompt = "This is a complicated te$t, ain't it?"

        for bos, eos in [[False, False], [True, True]]:
            encoded = tokenizer.instruct_tokenizer.tokenizer.encode(prompt, bos=bos, eos=eos)

            if special_token_policy is None or special_token_policy == SpecialTokenPolicy.IGNORE:
                assert tokenizer.decode(encoded, special_token_policy) == prompt
                assert tokenizer.instruct_tokenizer.decode(encoded, special_token_policy) == prompt
                assert tokenizer.instruct_tokenizer.tokenizer.decode(encoded, special_token_policy) == prompt

            elif special_token_policy == SpecialTokenPolicy.KEEP:
                if bos:
                    bos_piece = "<s>" if is_tekken else f"<s>{SPM_SPECIAL_WHITESPACE}"
                    expected = bos_piece + prompt
                elif not bos and not is_tekken:
                    expected = SPM_SPECIAL_WHITESPACE + prompt
                else:
                    expected = prompt
                if eos:
                    eos_piece = "</s>"
                    expected += eos_piece

                if not is_tekken:
                    expected = expected.replace(" ", SPM_WHITESPACE)

                assert tokenizer.decode(encoded, special_token_policy) == expected
                assert tokenizer.instruct_tokenizer.decode(encoded, special_token_policy) == expected
                assert tokenizer.instruct_tokenizer.tokenizer.decode(encoded, special_token_policy) == expected

            elif special_token_policy == SpecialTokenPolicy.RAISE:
                if bos or eos:
                    with pytest.raises(ValueError):
                        tokenizer.decode(encoded, special_token_policy)

    def test_from_hf_hub(self) -> None:
        def _mocked_hf_download(
            repo_id: str,
            token: Optional[Union[bool, str]] = None,
            revision: Optional[str] = None,
            force_download: bool = False,
            local_files_only: bool = False,
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


def _worker_decode_function(
    tokenizer_instance_and_token_ids_and_validation_mode: Tuple[MistralTokenizer, List[int], ValidationMode],
) -> str:
    tokenizer_instance, token_ids, validation_mode = tokenizer_instance_and_token_ids_and_validation_mode
    assert tokenizer_instance._chat_completion_request_validator._mode == validation_mode
    return tokenizer_instance.decode(token_ids)


@pytest.mark.parametrize(
    ["tokenizer_file", "validation_mode", "token_ids", "expected"],
    [
        (
            "tokenizer.model.v1",
            ValidationMode.test,
            [1, 733, 16289, 28793, 17121, 22526, 13, 13, 28708, 733, 28748, 16289, 28793],
            "[INST] SYSTEM\n\na [/INST]",
        ),
        (
            "tekken_240911.json",
            ValidationMode.finetuning,
            [1091, 3174, 3074, 1093, 126205, 1267, 1097, 1766, 1047, 3174, 3074, 1093],
            "[INST] SYSTEM\n\na [/INST]",
        ),
    ],
)
def test_tokenizer_is_pickleable_with_multiprocessing(
    tokenizer_file: str, validation_mode: ValidationMode, token_ids: List[int], expected: str
) -> None:
    tokenizer_path = str(MistralTokenizer._data_path() / tokenizer_file)
    tokenizer = MistralTokenizer.from_file(tokenizer_path, validation_mode)

    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(_worker_decode_function, [(tokenizer, token_ids, validation_mode)])

    assert len(results) == 1
    assert results[0] == expected

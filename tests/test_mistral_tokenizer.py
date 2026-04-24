import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.imports import is_sentencepiece_installed
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from tests.test_tekken import write_tekken_json_with_config

SPM_SPECIAL_WHITESPACE = "▁"
SPM_WHITESPACE = "▁"


class TestMistralToknizer:
    def test_from_model(self) -> None:
        assert isinstance(MistralTokenizer.from_model("mistral-medium-2312").instruct_tokenizer, InstructTokenizerV1)
        assert isinstance(MistralTokenizer.from_model("mistral-tiny-2312").instruct_tokenizer, InstructTokenizerV2)
        assert isinstance(MistralTokenizer.from_model("mistral-large-2402").instruct_tokenizer, InstructTokenizerV2)
        assert isinstance(
            MistralTokenizer.from_model("open-mixtral-8x22b-2404").instruct_tokenizer, InstructTokenizerV3
        )

        with pytest.raises(TokenizerException):
            MistralTokenizer.from_model("unknown-model")

    @pytest.mark.parametrize(
        ["special_token_policy", "is_tekken"],
        [
            (SpecialTokenPolicy.IGNORE, False),
            (SpecialTokenPolicy.IGNORE, True),
            (SpecialTokenPolicy.KEEP, False),
            (SpecialTokenPolicy.KEEP, True),
            (SpecialTokenPolicy.RAISE, False),
            (SpecialTokenPolicy.RAISE, True),
        ],
    )
    def test_decode(self, special_token_policy: SpecialTokenPolicy, is_tekken: bool) -> None:
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
            token: bool | str | None = None,
            revision: str | None = None,
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
    tokenizer_instance_and_token_ids_and_validation_mode: tuple[MistralTokenizer, list[int], ValidationMode],
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
    tokenizer_file: str, validation_mode: ValidationMode, token_ids: list[int], expected: str
) -> None:
    tokenizer_path = str(MistralTokenizer._data_path() / tokenizer_file)
    tokenizer = MistralTokenizer.from_file(tokenizer_path, validation_mode)

    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(_worker_decode_function, [(tokenizer, token_ids, validation_mode)])

    assert len(results) == 1
    assert results[0] == expected


def test_mistral_tokenizer_version_property() -> None:
    tokenizer_v1 = MistralTokenizer.from_model("mistral-medium-2312")
    assert (
        tokenizer_v1.version
        == tokenizer_v1.instruct_tokenizer.version
        == tokenizer_v1.instruct_tokenizer.tokenizer.version
        == TokenizerVersion.v1
    )

    tokenizer_v3 = MistralTokenizer.from_model("open-mixtral-8x22b-2404")
    assert (
        tokenizer_v3.version
        == tokenizer_v3.instruct_tokenizer.version
        == tokenizer_v3.instruct_tokenizer.tokenizer.version
        == TokenizerVersion.v3
    )


def test_mistral_tokenizer_mode_property() -> None:
    tokenizer_path = str(MistralTokenizer._data_path() / "tokenizer.model.v1")

    for mode in [ValidationMode.serving, ValidationMode.finetuning, ValidationMode.test]:
        tokenizer = MistralTokenizer.from_file(tokenizer_path, mode)
        assert tokenizer.mode == tokenizer._chat_completion_request_validator.mode == mode


@pytest.mark.parametrize(
    ["default_file_mode", "explicit_mode", "expected_mode"],
    [
        ("serving", None, ValidationMode.serving),
        ("serving", ValidationMode.finetuning, ValidationMode.finetuning),
        (None, None, ValidationMode.test),
    ],
)
def test_from_file_mode_resolution(
    tmp_path: Path,
    default_file_mode: str | None,
    explicit_mode: ValidationMode | None,
    expected_mode: ValidationMode,
) -> None:
    tokpath = tmp_path / "tekken.json"
    write_tekken_json_with_config(tokpath, default_validation_mode=default_file_mode)

    tokenizer = MistralTokenizer.from_file(tokpath, mode=explicit_mode)

    assert tokenizer.mode == expected_mode


@pytest.mark.skipif(
    not is_sentencepiece_installed(),
    reason="sentencepiece not installed",
)
def test_sentencepiece_default_validation_mode_is_none() -> None:
    spm_path = MistralTokenizer._data_path() / "tokenizer.model.v1"
    spm_tokenizer = SentencePieceTokenizer(str(spm_path))

    assert spm_tokenizer.default_validation_mode is None

    tokenizer = MistralTokenizer.from_file(str(spm_path))

    assert tokenizer.mode == ValidationMode.test

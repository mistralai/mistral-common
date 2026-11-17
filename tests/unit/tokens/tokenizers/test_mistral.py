r"""Tests for the `MistralTokenizer` façade: round trips, loading, decode, chat forwarding and FIM."""

import copy
import json
import multiprocessing
import pickle
import shutil
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from mistral_common.exceptions import TokenizerException
from mistral_common.protocol.fim.request import FIMRequest
from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import MistralRequestValidatorV3, ValidationMode
from mistral_common.tokens.tokenizers.base import InstructRequest, SpecialTokenPolicy, Tokenized, TokenizerVersion
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV1,
    InstructTokenizerV3,
    InstructTokenizerV11,
    InstructTokenizerV13,
    InstructTokenizerV15,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.test_tekken import get_special_tokens, quick_vocab
from tests.test_tokenizer_v7_audio import (
    SYNTHETIC_V7_INSTRUCT,
    build_synthetic_v7_audio_tokenizer,
    valid_transcription_request,
)
from tests.utils import assert_tokenized_equal, decode_keep

BUNDLED_TOKENIZERS = [
    "tokenizer.model.v1",
    "mistral_instruct_tokenizer_240216.model.v2",
    "mistral_instruct_tokenizer_240323.model.v3",
    "mistral_instruct_tokenizer_241114.model.v7",
    "mistral_instruct_tokenizer_241114.model.v7m1",
    "tekken_240718.json",
    "tekken_240911.json",
]

ROUND_TRIPS = pytest.mark.parametrize(
    "round_trip", [lambda t: pickle.loads(pickle.dumps(t)), copy.deepcopy], ids=["pickle", "deepcopy"]
)

REQUEST: ChatCompletionRequest = ChatCompletionRequest(
    model="test-model",
    messages=[UserMessage(content="Hello world"), AssistantMessage(content="Hi!"), UserMessage(content="Bye")],
)


class _CustomValidator(MistralRequestValidatorV3):
    pass


class _CustomNormalizer(InstructRequestNormalizer):
    pass


@pytest.mark.parametrize("tokenizer_file", BUNDLED_TOKENIZERS)
@ROUND_TRIPS
def test_bundled_tokenizer_round_trip(
    tokenizer_file: str, round_trip: Callable[[MistralTokenizer], MistralTokenizer]
) -> None:
    tokenizer = MistralTokenizer.from_file(MistralTokenizer._data_path() / tokenizer_file, ValidationMode.serving)
    expected = tokenizer.encode_chat_completion(REQUEST).tokens

    restored = round_trip(tokenizer)

    assert restored is not tokenizer
    assert restored.mode == ValidationMode.serving
    assert restored.instruct_tokenizer.tokenizer.version == tokenizer.instruct_tokenizer.tokenizer.version
    assert restored.encode_chat_completion(REQUEST).tokens == expected
    assert restored.decode(expected) == tokenizer.decode(expected)


@pytest.mark.parametrize("tokenizer_file", ["mistral_instruct_tokenizer_241114.model.v7m1", "tekken_240911.json"])
@ROUND_TRIPS
def test_image_tokenizer_round_trip(
    tokenizer_file: str, round_trip: Callable[[MistralTokenizer], MistralTokenizer]
) -> None:
    tokenizer = MistralTokenizer.from_file(MistralTokenizer._data_path() / tokenizer_file)
    image = Image.new("RGB", (40, 24), "red")
    request: ChatCompletionRequest = ChatCompletionRequest(
        messages=[UserMessage(content=[TextChunk(text="Describe"), ImageChunk(image=image)])]
    )
    expected = tokenizer.encode_chat_completion(request)

    restored = round_trip(tokenizer).encode_chat_completion(request)

    assert restored.tokens == expected.tokens
    assert len(restored.images) == len(expected.images) == 1
    np.testing.assert_array_equal(restored.images[0], expected.images[0])


@ROUND_TRIPS
def test_audio_tokenizer_round_trip(round_trip: Callable[[MistralTokenizer], MistralTokenizer]) -> None:
    # Built in memory, so the tokenizer has no source file to reload from.
    tokenizer = build_synthetic_v7_audio_tokenizer(SYNTHETIC_V7_INSTRUCT)
    request = valid_transcription_request()
    expected = tokenizer.encode_transcription(request)

    restored = round_trip(tokenizer).encode_transcription(request)

    assert restored.tokens == expected.tokens
    assert len(restored.audios) == len(expected.audios) == 1
    np.testing.assert_array_equal(restored.audios[0].audio_array, expected.audios[0].audio_array)


@ROUND_TRIPS
def test_round_trip_keeps_custom_validator_and_normalizer(
    round_trip: Callable[[MistralTokenizer], MistralTokenizer],
) -> None:
    loaded = MistralTokenizer.from_file(MistralTokenizer._data_path() / "mistral_instruct_tokenizer_240323.model.v3")
    tokenizer = MistralTokenizer(
        loaded.instruct_tokenizer,
        validator=_CustomValidator(mode=ValidationMode.serving),
        request_normalizer=_CustomNormalizer(
            UserMessage, AssistantMessage, ToolMessage, SystemMessage, InstructRequest, None
        ),
    )

    restored = round_trip(tokenizer)

    assert type(restored._chat_completion_request_validator) is _CustomValidator
    assert type(restored._instruct_request_normalizer) is _CustomNormalizer
    assert restored.mode == ValidationMode.serving
    assert restored.encode_chat_completion(REQUEST).tokens == tokenizer.encode_chat_completion(REQUEST).tokens


@pytest.mark.parametrize("tokenizer_file", ["tokenizer.model.v1", "tekken_240911.json"])
@pytest.mark.parametrize("lose_source", ["file_removed", "cwd_changed"])
@ROUND_TRIPS
def test_round_trip_without_the_source_file(
    tokenizer_file: str,
    lose_source: str,
    round_trip: Callable[[MistralTokenizer], MistralTokenizer],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutil.copy(MistralTokenizer._data_path() / tokenizer_file, tmp_path / tokenizer_file)
    monkeypatch.chdir(tmp_path)
    tokenizer = MistralTokenizer.from_file(tokenizer_file)
    expected = tokenizer.encode_chat_completion(REQUEST).tokens

    if lose_source == "file_removed":
        (tmp_path / tokenizer_file).unlink()
    else:
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.chdir(tmp_path / "elsewhere")

    assert round_trip(tokenizer).encode_chat_completion(REQUEST).tokens == expected


# fmt: off
_V1_SPM_TEST_TOKENS = [1, 733, 16289, 28793, 17121, 22526, 13, 13, 28708, 733, 28748, 16289, 28793]
_V3_TEKKEN_FINETUNING_TOKENS = [1091, 3174, 3074, 1093, 126205, 1267, 1097, 1766, 1047, 3174, 3074, 1093]
_V3_SPM_WITH_SUFFIX_TOKENS = [1, 13, 1575, 1032, 1416, 1055, 11, 1569, 1053, 29500]
_V3_TEKKEN_WITH_SUFFIX_TOKENS = [1, 16, 3004, 1261, 1867, 1289, 14, 3149, 1284, 1040]
_V3_SPM_WITHOUT_SUFFIX_TOKENS = [1, 13, 11, 1569, 1053, 29500]
# fmt: on

_V1_SPM_TEST_EXPECTED = Tokenized(tokens=_V1_SPM_TEST_TOKENS)
_V3_TEKKEN_FINETUNING_EXPECTED = Tokenized(tokens=_V3_TEKKEN_FINETUNING_TOKENS)
_V3_SPM_WITH_SUFFIX_EXPECTED = Tokenized(tokens=_V3_SPM_WITH_SUFFIX_TOKENS)
_V3_TEKKEN_WITH_SUFFIX_EXPECTED = Tokenized(tokens=_V3_TEKKEN_WITH_SUFFIX_TOKENS)
_V3_SPM_WITHOUT_SUFFIX_EXPECTED = Tokenized(tokens=_V3_SPM_WITHOUT_SUFFIX_TOKENS)

_DECODE_PROMPT = "This is a complicated te$t, ain't it?"
_DECODED_PROMPT_TEXT = "This is a complicated te$t, ain't it?"
_V3_SPM_PLAIN_KEEP_TEXT = "▁This▁is▁a▁complicated▁te$t,▁ain't▁it?"
_V3_SPM_BOS_EOS_KEEP_TEXT = "<s>▁This▁is▁a▁complicated▁te$t,▁ain't▁it?</s>"
_V3_TEKKEN_BOS_EOS_KEEP_TEXT = "<s>This is a complicated te$t, ain't it?</s>"
_PICKLED_DECODE_TEXT = "[INST] SYSTEM\n\na [/INST]"
_V3_SPM_WITH_SUFFIX_TEXT = "<s>[SUFFIX]return▁a▁+▁b[PREFIX]▁def▁f("
_V3_TEKKEN_WITH_SUFFIX_TEXT = "<s>[SUFFIX]return a + b[PREFIX]def f("
_V3_SPM_WITHOUT_SUFFIX_TEXT = "<s>[SUFFIX][PREFIX]▁def▁f("

_DECODE_SPECIAL_TOKENS_MESSAGE = r"^Decoding `tokens` that contain special tokens"
_NO_MAX_MODEL_LEN_MESSAGE = r"^encoding a chat completion request with truncation, but no max model len was provided$"
_FIM_NOT_AVAILABLE_MESSAGE = r"^FIM not available for v1$"
_UNRECOGNIZED_FILE_MESSAGE = r"^Unrecognized tokenizer file: .+[/\\]tokenizer\.json$"
_MULTIMODAL_SPM_MESSAGE = r"^Multimodal tokenizer is currently only supported for tekken$"


@pytest.fixture(scope="module")
def v1_tokenizer() -> MistralTokenizer:
    r"""Bundled v1 SentencePiece façade."""
    return MistralTokenizer.v1()


@pytest.fixture(scope="module")
def v3_spm_tokenizer() -> MistralTokenizer:
    r"""Bundled v3 SentencePiece façade."""
    return MistralTokenizer.v3(is_tekken=False)


@pytest.fixture(scope="module")
def v3_tekken_tokenizer() -> MistralTokenizer:
    r"""Bundled v3 Tekken façade."""
    return MistralTokenizer.v3(is_tekken=True)


def _fake_download_tokenizer_from_hf_hub(
    repo_id: str,
    token: bool | str | None = None,
    revision: str | None = None,
    force_download: bool = False,
    local_files_only: bool = False,
) -> str:
    r"""Return the bundled file for the two known repositories."""
    if repo_id == "mistralai/Mistral-7B-Instruct-v0.1":
        return str(MistralTokenizer._data_path() / "tokenizer.model.v1")
    elif repo_id == "mistralai/Pixtral-Large-Instruct-2411":
        return str(MistralTokenizer._data_path() / "tekken_240911.json")
    else:
        raise ValueError(f"Unknown repo_id: {repo_id}")


def _worker_decode_function(
    tokenizer_instance_and_token_ids_and_validation_mode: tuple[MistralTokenizer, list[int], ValidationMode],
) -> str:
    r"""Decode in a worker process, checking the pickled validation mode."""
    tokenizer_instance, token_ids, validation_mode = tokenizer_instance_and_token_ids_and_validation_mode
    assert tokenizer_instance._chat_completion_request_validator._mode == validation_mode
    return tokenizer_instance.decode(tokens=token_ids)


def _build_mocks() -> tuple[MagicMock, MagicMock, MagicMock]:
    r"""Build the validator, normalizer and encoder mocks for a façade."""
    validator = MagicMock()
    normalizer = MagicMock()
    normalizer.from_chat_completion_request.return_value = InstructRequest(messages=[UserMessage(content="a")])
    encoder = MagicMock()
    return validator, normalizer, encoder


def _build_chat_request(truncate_for_context_length: bool) -> ChatCompletionRequest:
    r"""Build a minimal chat request with the given truncation flag."""
    return ChatCompletionRequest(
        messages=[UserMessage(content="a")], truncate_for_context_length=truncate_for_context_length
    )


def _write_tekken_file(directory: Path, file_name: str, version: TokenizerVersion) -> Path:
    r"""Write a Tekken file. Synthetic formatting-only tokenizer; its IDs are not released-vocabulary evidence."""
    path = directory / file_name
    path.write_text(
        json.dumps(
            obj={
                "config": {
                    "pattern": r".+",
                    "default_vocab_size": 256 + 100,
                    "default_num_special_tokens": 100,
                    "version": version.value,
                },
                "vocab": quick_vocab(),
                "special_tokens": get_special_tokens(tokenizer_version=version),
            }
        ),
        encoding="utf-8",
    )
    return path


def _build_with_suffix_request() -> FIMRequest:
    r"""Build a FIM request with both prompt and suffix."""
    return FIMRequest(prompt="def f(", suffix="return a + b")


def _build_without_suffix_request() -> FIMRequest:
    r"""Build a FIM request with a prompt only."""
    return FIMRequest(prompt="def f(")


@pytest.mark.parametrize(
    argnames="repo_id, expected_type",
    argvalues=[
        pytest.param("mistralai/Mistral-7B-Instruct-v0.1", InstructTokenizerV1, id="v1-spm"),
        pytest.param("mistralai/Pixtral-Large-Instruct-2411", InstructTokenizerV3, id="v3-tekken-mm"),
    ],
)
def test_from_hf_hub(repo_id: str, expected_type: type) -> None:
    r"""Load the bundled file the mocked Hub download returns for each repository."""
    with patch(
        target="mistral_common.tokens.tokenizers.mistral.download_tokenizer_from_hf_hub",
        new=_fake_download_tokenizer_from_hf_hub,
    ):
        tokenizer = MistralTokenizer.from_hf_hub(repo_id=repo_id)
    assert type(tokenizer.instruct_tokenizer) is expected_type


def test_from_hf_hub_forwards_arguments() -> None:
    r"""Forward exactly the five download arguments and the mode to the loader."""
    with patch(
        target="mistral_common.tokens.tokenizers.mistral.download_tokenizer_from_hf_hub",
        return_value=str(MistralTokenizer._data_path() / "tokenizer.model.v1"),
    ) as download:
        tokenizer = MistralTokenizer.from_hf_hub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            token="t",
            revision="r",
            force_download=True,
            local_files_only=True,
            mode=ValidationMode.serving,
        )
    download.assert_called_once_with(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        token="t",
        revision="r",
        force_download=True,
        local_files_only=True,
    )
    assert tokenizer.mode == ValidationMode.serving


@pytest.mark.parametrize(
    argnames="version, expected_type",
    argvalues=[
        pytest.param(TokenizerVersion.v11, InstructTokenizerV11, id="v11"),
        pytest.param(TokenizerVersion.v13, InstructTokenizerV13, id="v13"),
        pytest.param(TokenizerVersion.v15, InstructTokenizerV15, id="v15"),
    ],
)
def test_from_file(version: TokenizerVersion, expected_type: type, tmp_path: Path) -> None:
    r"""Dispatch a synthetic Tekken file of each later version to its instruct tokenizer class."""
    path = _write_tekken_file(directory=tmp_path, file_name="tekken.json", version=version)
    tokenizer = MistralTokenizer.from_file(tokenizer_filename=path)
    assert type(tokenizer.instruct_tokenizer) is expected_type
    assert tokenizer.version == version


def test_from_file_rejected(tmp_path: Path) -> None:
    r"""Reject a valid Tekken file whose name marks neither backend."""
    path = _write_tekken_file(directory=tmp_path, file_name="tokenizer.json", version=TokenizerVersion.v11)
    with pytest.raises(expected_exception=TokenizerException, match=_UNRECOGNIZED_FILE_MESSAGE):
        MistralTokenizer.from_file(tokenizer_filename=path)


def test_v3_rejected() -> None:
    r"""Reject a SentencePiece multimodal v3 tokenizer."""
    with pytest.raises(expected_exception=ValueError, match=_MULTIMODAL_SPM_MESSAGE):
        MistralTokenizer.v3(is_tekken=False, is_mm=True)


@pytest.mark.parametrize(
    argnames="tokenizer_file, validation_mode, token_ids, expected_text",
    argvalues=[
        pytest.param(
            "tokenizer.model.v1",
            ValidationMode.test,
            _V1_SPM_TEST_TOKENS,
            _PICKLED_DECODE_TEXT,
            id="v1-spm-test",
        ),
        pytest.param(
            "tekken_240911.json",
            ValidationMode.finetuning,
            _V3_TEKKEN_FINETUNING_TOKENS,
            _PICKLED_DECODE_TEXT,
            id="v3-tekken-finetuning",
        ),
    ],
)
def test_pickle_with_multiprocessing(
    tokenizer_file: str,
    validation_mode: ValidationMode,
    token_ids: list[int],
    expected_text: str,
) -> None:
    r"""Reconstruct the façade in a worker process and decode the pinned tokens there."""
    tokenizer = MistralTokenizer.from_file(
        tokenizer_filename=str(MistralTokenizer._data_path() / tokenizer_file), mode=validation_mode
    )
    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(func=_worker_decode_function, iterable=[(tokenizer, token_ids, validation_mode)])
    assert results == [expected_text]


@pytest.mark.parametrize(
    argnames="config, expected_version",
    argvalues=[
        pytest.param("v1", TokenizerVersion.v1, id="v1"),
        pytest.param("v3_spm", TokenizerVersion.v3, id="v3-spm"),
    ],
)
def test_version(request: pytest.FixtureRequest, config: str, expected_version: TokenizerVersion) -> None:
    r"""The façade, instruct and raw tokenizer versions agree."""
    tokenizer: MistralTokenizer = request.getfixturevalue(argname=f"{config}_tokenizer")
    assert (
        tokenizer.version
        == tokenizer.instruct_tokenizer.version
        == tokenizer.instruct_tokenizer.tokenizer.version
        == expected_version
    )


@pytest.mark.parametrize(
    argnames="mode",
    argvalues=[
        pytest.param(ValidationMode.serving, id="serving"),
        pytest.param(ValidationMode.finetuning, id="finetuning"),
        pytest.param(ValidationMode.test, id="test"),
    ],
)
def test_mode(mode: ValidationMode) -> None:
    r"""The façade mode delegates to the validator."""
    tokenizer = MistralTokenizer.from_file(
        tokenizer_filename=str(MistralTokenizer._data_path() / "tokenizer.model.v1"), mode=mode
    )
    assert tokenizer.mode == tokenizer._chat_completion_request_validator.mode == mode


@pytest.mark.parametrize(
    argnames="config, special_token_policy, bos_eos, expected_text",
    argvalues=[
        pytest.param("v3_spm", SpecialTokenPolicy.IGNORE, False, _DECODED_PROMPT_TEXT, id="v3-spm-plain-ignore"),
        pytest.param("v3_spm", SpecialTokenPolicy.KEEP, False, _V3_SPM_PLAIN_KEEP_TEXT, id="v3-spm-plain-keep"),
        pytest.param("v3_spm", SpecialTokenPolicy.RAISE, False, _DECODED_PROMPT_TEXT, id="v3-spm-plain-raise"),
        pytest.param("v3_spm", SpecialTokenPolicy.IGNORE, True, _DECODED_PROMPT_TEXT, id="v3-spm-bos-eos-ignore"),
        pytest.param("v3_spm", SpecialTokenPolicy.KEEP, True, _V3_SPM_BOS_EOS_KEEP_TEXT, id="v3-spm-bos-eos-keep"),
        pytest.param("v3_tekken", SpecialTokenPolicy.IGNORE, False, _DECODED_PROMPT_TEXT, id="v3-tekken-plain-ignore"),
        pytest.param("v3_tekken", SpecialTokenPolicy.KEEP, False, _DECODED_PROMPT_TEXT, id="v3-tekken-plain-keep"),
        pytest.param("v3_tekken", SpecialTokenPolicy.RAISE, False, _DECODED_PROMPT_TEXT, id="v3-tekken-plain-raise"),
        pytest.param("v3_tekken", SpecialTokenPolicy.IGNORE, True, _DECODED_PROMPT_TEXT, id="v3-tekken-bos-eos-ignore"),
        pytest.param(
            "v3_tekken", SpecialTokenPolicy.KEEP, True, _V3_TEKKEN_BOS_EOS_KEEP_TEXT, id="v3-tekken-bos-eos-keep"
        ),
    ],
)
def test_decode(
    request: pytest.FixtureRequest,
    config: str,
    special_token_policy: SpecialTokenPolicy,
    bos_eos: bool,
    expected_text: str,
) -> None:
    r"""Decode the prompt under each policy at all three layers."""
    tokenizer: MistralTokenizer = request.getfixturevalue(argname=f"{config}_tokenizer")
    tokens = tokenizer.instruct_tokenizer.tokenizer.encode(s=_DECODE_PROMPT, bos=bos_eos, eos=bos_eos)
    assert tokenizer.decode(tokens=tokens, special_token_policy=special_token_policy) == expected_text
    instruct_text = tokenizer.instruct_tokenizer.decode(tokens=tokens, special_token_policy=special_token_policy)
    assert instruct_text == expected_text
    raw_text = tokenizer.instruct_tokenizer.tokenizer.decode(tokens=tokens, special_token_policy=special_token_policy)
    assert raw_text == expected_text


@pytest.mark.parametrize(
    argnames="config, expected_exception, expected_message",
    argvalues=[
        pytest.param("v3_spm", ValueError, _DECODE_SPECIAL_TOKENS_MESSAGE, id="v3-spm-bos-eos-raise"),
        pytest.param("v3_tekken", ValueError, _DECODE_SPECIAL_TOKENS_MESSAGE, id="v3-tekken-bos-eos-raise"),
    ],
)
def test_decode_rejected(
    request: pytest.FixtureRequest,
    config: str,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    r"""Reject RAISE decoding of control-bearing tokens at the façade."""
    tokenizer: MistralTokenizer = request.getfixturevalue(argname=f"{config}_tokenizer")
    tokens = tokenizer.instruct_tokenizer.tokenizer.encode(s=_DECODE_PROMPT, bos=True, eos=True)
    with pytest.raises(expected_exception=expected_exception, match=expected_message):
        tokenizer.decode(tokens=tokens, special_token_policy=SpecialTokenPolicy.RAISE)


@pytest.mark.parametrize(
    argnames="truncate_for_context_length, expected_truncate_at_max_tokens",
    argvalues=[
        pytest.param(False, None, id="truncation-disabled"),
        pytest.param(True, 128, id="truncation-enabled"),
    ],
)
def test_encode_chat_completion(
    truncate_for_context_length: bool,
    expected_truncate_at_max_tokens: int | None,
) -> None:
    r"""Forward validation, normalization and encoding, setting the truncation limit."""
    validator, normalizer, encoder = _build_mocks()
    tokenizer: MistralTokenizer = MistralTokenizer(
        instruct_tokenizer=encoder, validator=validator, request_normalizer=normalizer
    )
    request = _build_chat_request(truncate_for_context_length=truncate_for_context_length)
    result = tokenizer.encode_chat_completion(request=request, max_model_input_len=128)
    validator.validate_request.assert_called_once_with(request)
    normalizer.from_chat_completion_request.assert_called_once_with(validator.validate_request.return_value)
    encoder.encode_instruct.assert_called_once_with(normalizer.from_chat_completion_request.return_value)
    assert normalizer.from_chat_completion_request.return_value.truncate_at_max_tokens == (
        expected_truncate_at_max_tokens
    )
    assert result is encoder.encode_instruct.return_value


def test_encode_chat_completion_rejected() -> None:
    r"""Reject truncation without a limit after validation, before normalization."""
    validator, normalizer, encoder = _build_mocks()
    tokenizer: MistralTokenizer = MistralTokenizer(
        instruct_tokenizer=encoder, validator=validator, request_normalizer=normalizer
    )
    request = _build_chat_request(truncate_for_context_length=True)
    with pytest.raises(expected_exception=TokenizerException, match=_NO_MAX_MODEL_LEN_MESSAGE):
        tokenizer.encode_chat_completion(request=request)
    validator.validate_request.assert_called_once_with(request)
    normalizer.from_chat_completion_request.assert_not_called()
    encoder.encode_instruct.assert_not_called()


@pytest.mark.parametrize(
    argnames="config, request_builder, expected, expected_text",
    argvalues=[
        pytest.param(
            "v3_spm",
            _build_with_suffix_request,
            _V3_SPM_WITH_SUFFIX_EXPECTED,
            _V3_SPM_WITH_SUFFIX_TEXT,
            id="v3-spm-with-suffix",
        ),
        pytest.param(
            "v3_tekken",
            _build_with_suffix_request,
            _V3_TEKKEN_WITH_SUFFIX_EXPECTED,
            _V3_TEKKEN_WITH_SUFFIX_TEXT,
            id="v3-tekken-with-suffix",
        ),
        pytest.param(
            "v3_spm",
            _build_without_suffix_request,
            _V3_SPM_WITHOUT_SUFFIX_EXPECTED,
            _V3_SPM_WITHOUT_SUFFIX_TEXT,
            id="v3-spm-without-suffix",
        ),
    ],
)
def test_encode_fim(
    request: pytest.FixtureRequest,
    config: str,
    request_builder: Callable[[], FIMRequest],
    expected: Tokenized,
    expected_text: str,
) -> None:
    r"""Encode a FIM request through the public façade, pinning tokens and KEEP text."""
    tokenizer: MistralTokenizer = request.getfixturevalue(argname=f"{config}_tokenizer")
    tokenized = tokenizer.encode_fim(request=request_builder())
    assert_tokenized_equal(actual=tokenized, expected=expected)
    assert decode_keep(tokenizer=tokenizer, tokenized=tokenized) == expected_text


def test_encode_fim_rejected(request: pytest.FixtureRequest) -> None:
    r"""Reject a v1 FIM request at the public façade."""
    tokenizer: MistralTokenizer = request.getfixturevalue(argname="v1_tokenizer")
    fim_request = _build_with_suffix_request()
    with pytest.raises(expected_exception=TokenizerException, match=_FIM_NOT_AVAILABLE_MESSAGE):
        tokenizer.encode_fim(request=fim_request)

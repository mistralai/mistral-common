import copy
import pickle
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from PIL import Image

from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from mistral_common.protocol.instruct.normalize import InstructRequestNormalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.validator import MistralRequestValidatorV3, ValidationMode
from mistral_common.tokens.tokenizers.base import InstructRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.test_tokenizer_v7_audio import (
    SYNTHETIC_V7_INSTRUCT,
    build_synthetic_v7_audio_tokenizer,
    valid_transcription_request,
)

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


class _LegacyPickledTokenizer:
    r"""Pickles to the `(from_file, (path, mode))` form that `MistralTokenizer.__reduce__` used to produce."""

    def __init__(self, path: Path, mode: ValidationMode) -> None:
        self.path = path
        self.mode = mode

    def __reduce__(self) -> tuple[Callable, tuple[Path, ValidationMode]]:
        return MistralTokenizer.from_file, (self.path, self.mode)


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


@pytest.mark.parametrize("tokenizer_file", ["mistral_instruct_tokenizer_240323.model.v3", "tekken_240911.json"])
def test_pickle_from_file_based_form_still_loads(tokenizer_file: str) -> None:
    path = MistralTokenizer._data_path() / tokenizer_file
    data = pickle.dumps(_LegacyPickledTokenizer(path, ValidationMode.serving))

    restored = pickle.loads(data)

    assert isinstance(restored, MistralTokenizer)
    assert restored.mode == ValidationMode.serving
    assert restored.encode_chat_completion(REQUEST).tokens == (
        MistralTokenizer.from_file(path).encode_chat_completion(REQUEST).tokens
    )

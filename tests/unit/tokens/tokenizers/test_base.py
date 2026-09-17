import operator
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

import mistral_common.deprecation
from mistral_common.protocol.fim.request import FIMRequest
from mistral_common.protocol.instruct.chunk import ContentChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import Tool
from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import Audio, AudioEncoder
from mistral_common.tokens.tokenizers.base import (
    InstructTokenizer,
    SpecialTokenPolicy,
    Tokenized,
    Tokenizer,
    TokenizerVersion,
)
from mistral_common.tokens.tokenizers.image import ImageEncoder


class StubTokenizer(Tokenizer):
    @property
    def n_words(self) -> int:
        return 4

    @property
    def special_ids(self) -> set[int]:
        return {0, 3}

    @property
    def num_special_tokens(self) -> int:
        return 2

    @property
    def model_settings_builder(self) -> None:
        return None

    def vocab(self) -> list[str]:
        return ["<unk>", "a", "b", "</s>"]

    def id_to_piece(self, token_id: int) -> str:
        return self.vocab()[token_id]

    @property
    def bos_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return 3

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 0

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        del s, bos, eos
        return [1, 2]

    def decode(self, tokens: list[int], special_token_policy: SpecialTokenPolicy = SpecialTokenPolicy.IGNORE) -> str:
        del tokens
        return special_token_policy.value

    def get_special_token(self, s: str) -> int:
        return {"<unk>": 0, "</s>": 3}[s]

    def is_special(self, token: int | np.integer | str) -> bool:
        if isinstance(token, str):
            return token in {"<unk>", "</s>"}
        return int(token) in self.special_ids

    @property
    def version(self) -> TokenizerVersion:
        return TokenizerVersion.v15

    def _to_string(self, tokens: list[int]) -> str:
        return "".join(self.id_to_piece(token_id=token) for token in tokens)

    @property
    def file_path(self) -> Path:
        return Path("tokenizer.model")


class StubInstructTokenizer(InstructTokenizer[InstructRequest, FIMRequest, Tokenized, AssistantMessage]):
    def __init__(
        self,
        tokenizer: Tokenizer,
        image_encoder: ImageEncoder | None = None,
        audio_encoder: AudioEncoder | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder
        super().__init__(tokenizer=tokenizer, image_encoder=image_encoder, audio_encoder=audio_encoder)

    def encode_instruct(self, request: InstructRequest) -> Tokenized:
        del request
        return Tokenized(tokens=[1])

    def encode_transcription(self, request: TranscriptionRequest) -> Tokenized:
        del request
        return Tokenized(tokens=[2])

    def encode_speech_request(self, request: SpeechRequest) -> Tokenized:
        del request
        return Tokenized(tokens=[3])

    def decode(self, tokens: list[int], special_token_policy: SpecialTokenPolicy) -> str:
        return f"{tokens}:{special_token_policy.value}"

    def encode_fim(self, request: FIMRequest) -> Tokenized:
        del request
        return Tokenized(tokens=[4])

    def encode_user_message(
        self,
        message: UserMessage,
        available_tools: list[Tool] | None,
        is_last: bool,
        is_first: bool,
        system_prompt: str | None = None,
        force_img_first: bool = False,
    ) -> tuple[list[int], list[np.ndarray], list[Audio]]:
        del message, available_tools, is_last, is_first, system_prompt, force_img_first
        return [5], [], []

    def encode_user_content(
        self,
        content: str | list[ContentChunk],
        is_last: bool,
        system_prompt: str | None = None,
        force_img_first: bool = False,
    ) -> tuple[list[int], list[np.ndarray], list[Audio]]:
        del content, is_last, system_prompt, force_img_first
        return [6], [], []

    def _to_string(self, tokens: list[int]) -> str:
        return str(tokens)


@pytest.fixture(autouse=True)
def clear_tokenized_text_warning() -> Iterator[None]:
    was_warned = "Tokenized.text" in mistral_common.deprecation._warned_keys
    mistral_common.deprecation._warned_keys.discard("Tokenized.text")
    yield
    if was_warned:
        mistral_common.deprecation._warned_keys.add("Tokenized.text")
    else:
        mistral_common.deprecation._warned_keys.discard("Tokenized.text")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0, SpecialTokenPolicy.IGNORE, id="ignore-int"),
        pytest.param(1, SpecialTokenPolicy.KEEP, id="keep-int"),
        pytest.param(2, SpecialTokenPolicy.RAISE, id="raise-int"),
        pytest.param("ignore", SpecialTokenPolicy.IGNORE, id="ignore-string"),
        pytest.param("keep", SpecialTokenPolicy.KEEP, id="keep-string"),
        pytest.param("raise", SpecialTokenPolicy.RAISE, id="raise-string"),
    ],
)
def test_special_token_policy_accepts_legacy_integer_values(value: int | str, expected: SpecialTokenPolicy) -> None:
    assert SpecialTokenPolicy(value) == expected


@pytest.mark.parametrize(
    "value",
    [pytest.param(3, id="unknown-int"), pytest.param("invalid", id="unknown-string")],
)
def test_special_token_policy_rejects_unknown_values(value: int | str) -> None:
    with pytest.raises(ValueError, match="not a valid SpecialTokenPolicy"):
        SpecialTokenPolicy(value)


@pytest.mark.parametrize(
    ("version", "expected"),
    [pytest.param(TokenizerVersion.v1, 1, id="v1"), pytest.param(TokenizerVersion.v15, 15, id="v15")],
)
def test_tokenizer_version_reports_numeric_value(version: TokenizerVersion, expected: int) -> None:
    assert version.version_num == expected


def test_tokenizer_version_numeric_value_is_deprecated_alias() -> None:
    with pytest.warns(UserWarning, match=r"TokenizerVersion\._version_num.*deprecated"):
        assert TokenizerVersion.v7._version_num == 7


@pytest.mark.parametrize(
    ("comparison", "left", "right", "expected"),
    [
        pytest.param(operator.lt, TokenizerVersion.v1, TokenizerVersion.v2, True, id="less-than"),
        pytest.param(operator.lt, TokenizerVersion.v1, "v2", True, id="less-than-string"),
        pytest.param(operator.lt, TokenizerVersion.v2, TokenizerVersion.v2, False, id="less-than-equal-boundary"),
        pytest.param(operator.le, TokenizerVersion.v2, "v2", True, id="less-than-or-equal"),
        pytest.param(operator.gt, TokenizerVersion.v15, TokenizerVersion.v13, True, id="greater-than"),
        pytest.param(operator.gt, TokenizerVersion.v15, "v13", True, id="greater-than-string"),
        pytest.param(operator.gt, TokenizerVersion.v2, "v2", False, id="greater-than-equal-boundary"),
        pytest.param(operator.ge, TokenizerVersion.v15, "v15", True, id="greater-than-or-equal"),
        pytest.param(operator.eq, TokenizerVersion.v2, TokenizerVersion.v2, True, id="enum-equality"),
        pytest.param(operator.eq, TokenizerVersion.v2, "v2", True, id="string-equality"),
        pytest.param(operator.eq, TokenizerVersion.v2, "v3", False, id="string-inequality"),
    ],
)
def test_tokenizer_version_comparison_uses_numeric_version(
    comparison: Callable[[TokenizerVersion, str | TokenizerVersion], bool],
    left: TokenizerVersion,
    right: str | TokenizerVersion,
    expected: bool,
) -> None:
    assert comparison(left, right) == expected


def test_tokenizer_version_comparison_rejects_unknown_string() -> None:
    with pytest.raises(ValueError, match="'unknown' is not a valid TokenizerVersion"):
        TokenizerVersion.v1 < "unknown"


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        pytest.param(TokenizerVersion.v13, False, id="before-model-settings"),
        pytest.param(TokenizerVersion.v15, True, id="model-settings"),
    ],
)
def test_tokenizer_version_reports_model_settings_support(version: TokenizerVersion, expected: bool) -> None:
    assert version.supports_model_settings == expected


def test_tokenized_text_returns_cached_value_and_warns_once() -> None:
    tokenized = Tokenized(tokens=[1])
    second_tokenized = Tokenized(tokens=[2])
    tokenized._text = "decoded"
    second_tokenized._text = "also decoded"

    with pytest.warns(DeprecationWarning, match="`text` property of `Tokenized`"):
        assert tokenized.text == "decoded"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert tokenized.text == "decoded"
        assert second_tokenized.text == "also decoded"

    assert [warning for warning in caught if issubclass(warning.category, DeprecationWarning)] == []


def test_tokenized_text_is_none_when_no_cached_value_exists() -> None:
    with pytest.warns(DeprecationWarning):
        assert Tokenized(tokens=[1]).text is None


def test_instruct_tokenizer_delegates_version_to_tokenizer() -> None:
    instruct_tokenizer = StubInstructTokenizer(tokenizer=StubTokenizer())

    assert instruct_tokenizer.version == TokenizerVersion.v15

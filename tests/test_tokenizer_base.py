import inspect
import pickle
import warnings
from typing import Any

import pytest

import mistral_common.deprecation
from mistral_common.protocol.fim.request import FIMRequest
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import InstructRequest, ModelSettings
from mistral_common.tokens.tokenizers.base import InstructTokenizer, SpecialTokenPolicy, Tokenized
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerBase,
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils import decode_keep

GENERIC_TOKENIZER_TYPES: tuple[type[Any], ...] = (
    InstructTokenizer,
    InstructTokenizerBase,
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
)


def test_special_token_policy_backward_compatibility() -> None:
    assert SpecialTokenPolicy(0) == SpecialTokenPolicy.IGNORE
    assert SpecialTokenPolicy(1) == SpecialTokenPolicy.KEEP
    assert SpecialTokenPolicy(2) == SpecialTokenPolicy.RAISE

    with pytest.raises(ValueError, match=r"3 is not a valid SpecialTokenPolicy"):
        SpecialTokenPolicy(3)
    with pytest.raises(ValueError, match=r"'invalid' is not a valid SpecialTokenPolicy"):
        SpecialTokenPolicy("invalid")


@pytest.mark.parametrize("tokenizer_type", GENERIC_TOKENIZER_TYPES)
def test_generic_tokenizer_types_have_three_parameters(tokenizer_type: type[Any]) -> None:
    assert len(getattr(tokenizer_type, "__parameters__", ())) == 3


def test_instruct_tokenizer_runtime_contract_has_no_extended_methods() -> None:
    parameters = inspect.signature(InstructTokenizer.encode_user_message).parameters

    assert "settings" not in parameters
    assert not hasattr(InstructTokenizer, "encode_system_message")


def _get_mistral_instruct_tokenizer(
    tokenizer: MistralTokenizer[UserMessage, AssistantMessage, ToolMessage, SystemMessage, Tokenized],
) -> InstructTokenizer[InstructRequest, FIMRequest, Tokenized]:
    return tokenizer.instruct_tokenizer


def _check_instruct_tokenizer_extended_static_contract(tokenizer: InstructTokenizerBase) -> None:
    tokenizer.encode_user_message(
        message=UserMessage(content=""),
        available_tools=None,
        is_last=True,
        is_first=True,
        system_prompt=None,
        force_img_first=False,
        settings=ModelSettings.none(),
    )
    tokenizer.encode_system_message(SystemMessage(content=""))


def test_mistral_tokenizer_wires_three_parameter_instruct_tokenizer() -> None:
    instruct_tokenizer = _get_mistral_instruct_tokenizer(MistralTokenizer.v3())

    assert isinstance(instruct_tokenizer, InstructTokenizer)


@pytest.fixture(autouse=True)
def _clear_text_tokenized_warning() -> None:
    mistral_common.deprecation._warned_keys.discard("Tokenized.text")


def _make_tokenized() -> tuple[InstructTokenizer, Tokenized]:
    tokenizer = MistralTokenizer.v3().instruct_tokenizer
    tokenized = tokenizer.encode_instruct(InstructRequest(messages=[UserMessage(content="a")]))
    assert isinstance(tokenized, Tokenized)
    return tokenizer, tokenized


def test_tokenized_text_property_returns_decoded_text() -> None:
    tokenizer, tokenized = _make_tokenized()
    expected = decode_keep(tokenizer, tokenized)

    with pytest.warns(DeprecationWarning, match="`text` property of `Tokenized`"):
        text = tokenized.text

    assert text == expected


def test_tokenized_text_property_returns_none_when_not_produced_by_a_tokenizer() -> None:
    tokenized = Tokenized(tokens=[1, 2, 3])

    with pytest.warns(DeprecationWarning, match="`text` property of `Tokenized`"):
        text = tokenized.text

    assert text is None


def test_tokenized_text_property_warns_only_once() -> None:
    _, tokenized = _make_tokenized()

    with pytest.warns(DeprecationWarning, match="`text` property of `Tokenized`"):
        tokenized.text

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Accessing `text` again on the same instance, and on a different one, must not warn again.
        tokenized.text
        _make_tokenized()[1].text

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings == []


def test_tokenized_equality_includes_cached_text() -> None:
    with_text = Tokenized(tokens=[1, 2, 3])
    with_text._text = "cached"
    without_text = Tokenized(tokens=[1, 2, 3])

    assert with_text != without_text


def test_tokenized_pickle_roundtrip_preserves_text() -> None:
    tokenizer, tokenized = _make_tokenized()
    expected_text = decode_keep(tokenizer, tokenized)

    restored = pickle.loads(pickle.dumps(tokenized))

    assert restored == tokenized
    with pytest.warns(DeprecationWarning):
        assert restored.text == expected_text

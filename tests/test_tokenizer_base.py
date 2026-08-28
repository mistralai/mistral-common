import pickle
import warnings

import pytest

import mistral_common.deprecation
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.base import InstructTokenizer, SpecialTokenPolicy, Tokenized
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils import decode_keep


def test_special_token_policy_backward_compatibility() -> None:
    assert SpecialTokenPolicy(0) == SpecialTokenPolicy.IGNORE
    assert SpecialTokenPolicy(1) == SpecialTokenPolicy.KEEP
    assert SpecialTokenPolicy(2) == SpecialTokenPolicy.RAISE

    with pytest.raises(ValueError, match=r"3 is not a valid SpecialTokenPolicy"):
        SpecialTokenPolicy(3)
    with pytest.raises(ValueError, match=r"'invalid' is not a valid SpecialTokenPolicy"):
        SpecialTokenPolicy("invalid")


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

import operator
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Callable

import pytest

import mistral_common
from mistral_common.tokens.tokenizers.base import SpecialTokens, Tokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import quick_vocab

ALL_VERSIONS = list(TokenizerVersion)

# Expected values of each externally-consumed property, keyed by version.
_EXPECTED: dict[str, dict[TokenizerVersion, bool]] = {
    "has_tools": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: True,
        TokenizerVersion.v3: True,
        TokenizerVersion.v7: True,
        TokenizerVersion.v11: True,
        TokenizerVersion.v13: True,
        TokenizerVersion.v15: True,
    },
    "uses_pre_v11_tool_format": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: True,
        TokenizerVersion.v3: True,
        TokenizerVersion.v7: True,
        TokenizerVersion.v11: False,
        TokenizerVersion.v13: False,
        TokenizerVersion.v15: False,
    },
    "supports_model_settings": {
        TokenizerVersion.v1: False,
        TokenizerVersion.v2: False,
        TokenizerVersion.v3: False,
        TokenizerVersion.v7: False,
        TokenizerVersion.v11: False,
        TokenizerVersion.v13: False,
        TokenizerVersion.v15: True,
    },
}

_VERSIONED_PROPERTIES = list(_EXPECTED)


class TestTokenizerVersionProperties:
    @pytest.mark.parametrize("version", ALL_VERSIONS)
    @pytest.mark.parametrize("prop", _VERSIONED_PROPERTIES)
    def test_property_matches_expected_value(self, version: TokenizerVersion, prop: str) -> None:
        assert getattr(version, prop) is _EXPECTED[prop][version]


class TestTokenizerVersionComparisons:
    @pytest.mark.parametrize("other", [TokenizerVersion.v11, "v11"])
    def test_orders_by_version_number_not_by_string(self, other: str | TokenizerVersion) -> None:
        # As plain strings "v2" > "v11"; versions must compare by number.
        assert TokenizerVersion.v2 < other
        assert TokenizerVersion.v2 <= other
        assert not TokenizerVersion.v2 > other
        assert not TokenizerVersion.v2 >= other

    @pytest.mark.parametrize("other", [TokenizerVersion.v7, "v7"])
    def test_equal_versions(self, other: str | TokenizerVersion) -> None:
        assert TokenizerVersion.v7 <= other
        assert TokenizerVersion.v7 >= other
        assert not TokenizerVersion.v7 < other
        assert not TokenizerVersion.v7 > other

    def test_str_on_the_left(self) -> None:
        assert "v2" < TokenizerVersion.v11
        assert "v11" >= TokenizerVersion.v2
        assert not "v11" <= TokenizerVersion.v2

    @pytest.mark.parametrize("op", [operator.lt, operator.le, operator.gt, operator.ge])
    def test_unknown_version_string_raises(self, op: Callable[[object, object], bool]) -> None:
        with pytest.raises(ValueError, match="'v99' is not a valid TokenizerVersion"):
            op(TokenizerVersion.v7, "v99")


_SP_MODEL = Path(mistral_common.__file__).parent / "data" / "tokenizer.model.v1"


def _tekkenizer() -> Tekkenizer:
    special_tokens = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)
    vocab = quick_vocab([b"hello"])
    return Tekkenizer(
        vocab=vocab,
        special_tokens=special_tokens,
        pattern=r".+",
        vocab_size=len(vocab) + len(special_tokens),
        num_special_tokens=len(special_tokens),
        version=TokenizerVersion.v3,
    )


@pytest.fixture(scope="module", params=["tekken", "sentencepiece"])
def tokenizer(request: pytest.FixtureRequest) -> Tokenizer:
    if request.param == "tekken":
        return _tekkenizer()
    return SentencePieceTokenizer(model_path=_SP_MODEL)


class TestTokenizerWarningFilters:
    @pytest.mark.parametrize(
        "module", ["mistral_common.tokens.tokenizers.tekken", "mistral_common.tokens.tokenizers.sentencepiece"]
    )
    def test_import_leaves_warning_filters_unchanged(self, module: str) -> None:
        code = (
            "import warnings\n"
            "import mistral_common.tokens.tokenizers.base\n"
            "before = list(warnings.filters)\n"
            f"import {module}\n"
            "added = [f for f in warnings.filters if f not in before]\n"
            "assert warnings.filters == before, added\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_get_control_token_honors_command_line_error_filter(self) -> None:
        code = (
            "import mistral_common.tokens.tokenizers.tekken\n"
            "from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer\n"
            f"SentencePieceTokenizer(model_path={str(_SP_MODEL)!r}).get_control_token('</s>')\n"
        )
        result = subprocess.run(
            [sys.executable, "-W", "error::FutureWarning", "-c", code], capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "FutureWarning: `get_control_token` is deprecated." in result.stderr


class TestTokenizerSharedMethods:
    def test_get_control_token_warns_once_at_caller(self, tokenizer: Tokenizer) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert tokenizer.get_control_token(SpecialTokens.eos.value) == tokenizer.eos_id
            assert tokenizer.get_control_token(SpecialTokens.bos.value) == tokenizer.bos_id

        assert len(record) == 1
        assert record[0].category is FutureWarning
        assert "`get_control_token` is deprecated. Use `get_special_token` instead." in str(record[0].message)
        assert record[0].filename == __file__

    def test_sentencepiece_control_tokens_warns_once_at_caller(self) -> None:
        sp_tokenizer = SentencePieceTokenizer(model_path=_SP_MODEL)
        other = SentencePieceTokenizer(model_path=_SP_MODEL)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert sp_tokenizer._control_tokens == sp_tokenizer.special_ids
            assert other._control_tokens == other.special_ids

        assert len(record) == 1
        assert record[0].category is FutureWarning
        assert "`_control_tokens` is deprecated." in str(record[0].message)
        assert record[0].filename == __file__

import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tests.utils import registry
from tests.utils.requests.fim import fim_request
from tests.utils.tokenizers import (
    SHIPPED_FIM_CAPABLE_KEYS,
    SPM_FIM_CAPABLE_KEYS,
    SPM_INFILLING_SENTINEL,
    SYNTHETIC_FIM_CAPABLE_KEYS,
    TEKKEN_FIM_CAPABLE_KEYS,
)

# Representative suffixes covering the shapes `_encode_infilling` must round-trip exactly:
# leading whitespace, a leading newline, non-ascii, punctuation, and digits. A single
# leading space is deliberately excluded: leading-space suffixes are backend-dependent
# (SPM drops one leading space, tekken preserves it exactly), and that split is pinned
# separately by `TestEncodeInfilling.test_encode_infilling_drops_one_leading_space_on_spm`
# and `test_encode_infilling_preserves_leading_space_on_tekken`.
_INFILLING_ROUND_TRIP_SUFFIXES: tuple[str, ...] = (
    "\treturn a + b",
    "\n    return a + b",
    "café",
    ".strip()",
    "1234",
)

# Leading-space suffixes exercising the SPM/tekken split pinned by
# `TestEncodeInfilling.test_encode_infilling_drops_one_leading_space_on_spm` and
# `test_encode_infilling_preserves_leading_space_on_tekken`.
_LEADING_SPACE_SUFFIXES: tuple[str, ...] = (" return", "  indented", "    deep")


class TestEncodeFim:
    @pytest.mark.parametrize("shipped_mistral_tokenizer", ["v1_spm"], indirect=True)
    def test_encode_fim_v1_raises(self, shipped_mistral_tokenizer: MistralTokenizer) -> None:
        with pytest.raises(TokenizerException, match="FIM not available for"):
            shipped_mistral_tokenizer.encode_fim(fim_request())

    @pytest.mark.parametrize("shipped_mistral_tokenizer", ["v2_spm"], indirect=True)
    def test_encode_fim_v2_emits_unk_for_fim_markers(self, shipped_mistral_tokenizer: MistralTokenizer) -> None:
        # Pins the pair documented in `registry.SILENT_UNSUPPORTED_PROTOCOLS`: v2's shipped
        # SentencePiece model has no `[SUFFIX]`/`[PREFIX]` pieces, so `piece_to_id` returns
        # `<unk>` instead of raising, and `encode_fim` succeeds silently rather than raising
        # like v1 (`test_encode_fim_v1_raises`).
        assert (TokenizerVersion.v2, "fim") in registry.SILENT_UNSUPPORTED_PROTOCOLS
        unk_id = shipped_mistral_tokenizer.instruct_tokenizer.tokenizer.unk_id
        tokenized = shipped_mistral_tokenizer.encode_fim(fim_request())
        assert tokenized.tokens.count(unk_id) == 2
        assert tokenized.tokens[1] == unk_id

    @pytest.mark.parametrize("shipped_mistral_tokenizer", SHIPPED_FIM_CAPABLE_KEYS, indirect=True)
    def test_encode_fim_starts_with_bos_once(self, shipped_mistral_tokenizer: MistralTokenizer) -> None:
        tokenized = shipped_mistral_tokenizer.encode_fim(fim_request())
        bos = shipped_mistral_tokenizer.instruct_tokenizer.tokenizer.bos_id
        assert tokenized.tokens[0] == bos
        assert tokenized.tokens.count(bos) == 1

    @pytest.mark.parametrize("shipped_mistral_tokenizer", SHIPPED_FIM_CAPABLE_KEYS, indirect=True)
    def test_encode_fim_empty_suffix_matches_none_suffix(self, shipped_mistral_tokenizer: MistralTokenizer) -> None:
        empty_suffix = shipped_mistral_tokenizer.encode_fim(fim_request(suffix=""))
        no_suffix = shipped_mistral_tokenizer.encode_fim(fim_request(suffix=None))
        assert empty_suffix == no_suffix

    @pytest.mark.parametrize("shipped_mistral_tokenizer", SHIPPED_FIM_CAPABLE_KEYS, indirect=True)
    def test_encode_fim_decode_strips_special_tokens(self, shipped_mistral_tokenizer: MistralTokenizer) -> None:
        tokenized = shipped_mistral_tokenizer.encode_fim(fim_request())
        decoded = shipped_mistral_tokenizer.decode(tokenized.tokens, special_token_policy=SpecialTokenPolicy.IGNORE)
        assert "[SUFFIX]" not in decoded
        assert "[PREFIX]" not in decoded


class TestEncodeInfilling:
    @pytest.mark.parametrize(
        "keyed_mistral_tokenizer", SHIPPED_FIM_CAPABLE_KEYS + SYNTHETIC_FIM_CAPABLE_KEYS, indirect=True
    )
    @pytest.mark.parametrize("suffix", _INFILLING_ROUND_TRIP_SUFFIXES)
    def test_encode_infilling_round_trips_suffix(self, keyed_mistral_tokenizer: MistralTokenizer, suffix: str) -> None:
        instruct_tokenizer = keyed_mistral_tokenizer.instruct_tokenizer
        encoded = instruct_tokenizer._encode_infilling(suffix)  # type: ignore[attr-defined]
        assert instruct_tokenizer.tokenizer.decode(encoded) == suffix

    @pytest.mark.parametrize("keyed_mistral_tokenizer", SPM_FIM_CAPABLE_KEYS, indirect=True)
    @pytest.mark.parametrize("suffix", _LEADING_SPACE_SUFFIXES)
    def test_encode_infilling_drops_one_leading_space_on_spm(
        self, keyed_mistral_tokenizer: MistralTokenizer, suffix: str
    ) -> None:
        # The sentinel absorbs its own implicit SentencePiece prefix space along with one
        # real leading space from the suffix (see `SPM_INFILLING_SENTINEL`).
        instruct_tokenizer = keyed_mistral_tokenizer.instruct_tokenizer
        encoded = instruct_tokenizer._encode_infilling(suffix)  # type: ignore[attr-defined]
        assert instruct_tokenizer.tokenizer.decode(encoded) == suffix[1:]

    @pytest.mark.parametrize("keyed_mistral_tokenizer", TEKKEN_FIM_CAPABLE_KEYS, indirect=True)
    @pytest.mark.parametrize("suffix", _LEADING_SPACE_SUFFIXES)
    def test_encode_infilling_preserves_leading_space_on_tekken(
        self, keyed_mistral_tokenizer: MistralTokenizer, suffix: str
    ) -> None:
        instruct_tokenizer = keyed_mistral_tokenizer.instruct_tokenizer
        encoded = instruct_tokenizer._encode_infilling(suffix)  # type: ignore[attr-defined]
        assert instruct_tokenizer.tokenizer.decode(encoded) == suffix

    @pytest.mark.parametrize(
        "keyed_mistral_tokenizer", SHIPPED_FIM_CAPABLE_KEYS + SYNTHETIC_FIM_CAPABLE_KEYS, indirect=True
    )
    def test_encode_infilling_sentinel_is_two_tokens(self, keyed_mistral_tokenizer: MistralTokenizer) -> None:
        # Secondary guard: the round-trip above is the invariant that matters. This pins the
        # incidental detail the hardcoded `[2:]` slice in `_encode_infilling` depends on.
        sentinel_tokens = keyed_mistral_tokenizer.instruct_tokenizer.tokenizer.encode(
            SPM_INFILLING_SENTINEL, bos=False, eos=False
        )
        assert len(sentinel_tokens) == 2

    @pytest.mark.parametrize("shipped_mistral_tokenizer", ["v3_tekken"], indirect=True)
    @pytest.mark.parametrize("suffix", ["return a + b", "[0]", ":"])
    def test_encode_infilling_is_noop_on_tekken_for_common_suffixes(
        self, shipped_mistral_tokenizer: MistralTokenizer, suffix: str
    ) -> None:
        instruct_tokenizer = shipped_mistral_tokenizer.instruct_tokenizer
        plain = instruct_tokenizer.tokenizer.encode(suffix, bos=False, eos=False)
        assert instruct_tokenizer._encode_infilling(suffix) == plain  # type: ignore[attr-defined]

    @pytest.mark.parametrize("shipped_mistral_tokenizer", ["v3_tekken"], indirect=True)
    @pytest.mark.parametrize("suffix", [".strip()", "(x, y):"])
    def test_encode_infilling_alters_tokens_on_tekken_but_still_round_trips(
        self, shipped_mistral_tokenizer: MistralTokenizer, suffix: str
    ) -> None:
        # Characterizes today's known degradation (see `SPM_INFILLING_SENTINEL`): on a real
        # tekken vocabulary the sentinel is not always a no-op, but it never corrupts the
        # decoded suffix. If `_encode_infilling` is later made SPM-only, this test's
        # assertions -- not just a golden file -- will force that change to be reviewed.
        instruct_tokenizer = shipped_mistral_tokenizer.instruct_tokenizer
        plain = instruct_tokenizer.tokenizer.encode(suffix, bos=False, eos=False)
        encoded = instruct_tokenizer._encode_infilling(suffix)  # type: ignore[attr-defined]
        assert encoded != plain
        assert instruct_tokenizer.tokenizer.decode(encoded) == suffix

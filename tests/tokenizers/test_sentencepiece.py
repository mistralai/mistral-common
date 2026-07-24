import numpy as np
import pytest

from mistral_common.exceptions import TokenizerException
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, TokenizerVersion
from mistral_common.tokens.tokenizers.sentencepiece import (
    SentencePieceTokenizer,
    get_image_config,
    get_spm_version,
    is_sentencepiece,
    is_sentencepiece_tokenizer,
)
from tests.utils.tokenizers import build_tekkenizer


class TestIsSentencepiece:
    def test_str_path_existing_file_returns_true(self, spm: SentencePieceTokenizer) -> None:
        assert is_sentencepiece(str(spm.file_path)) is True

    def test_path_object_existing_file_returns_true(self, spm: SentencePieceTokenizer) -> None:
        assert is_sentencepiece(spm.file_path) is True

    def test_nonexistent_path_returns_false(self) -> None:
        assert is_sentencepiece("/nonexistent/tokenizer.model.v7") is False


class TestGetSpmVersion:
    def test_matches_tokenizer_version(self, spm: SentencePieceTokenizer) -> None:
        assert get_spm_version(tokenizer_filename=spm.file_path) == spm.version

    def test_bare_model_extension_returns_v1(self) -> None:
        assert get_spm_version(tokenizer_filename="tokenizer.model") == TokenizerVersion.v1

    def test_bare_model_with_raise_deprecated_raises(self) -> None:
        with pytest.raises(TokenizerException, match="rename your tokenizer file"):
            get_spm_version(tokenizer_filename="tokenizer.model", raise_deprecated=True)

    def test_multimodal_suffix_strips_to_instruct_version(self) -> None:
        assert get_spm_version(tokenizer_filename="tokenizer.model.v7m1") == TokenizerVersion.v7

    def test_unrecognized_version_suffix_raises(self) -> None:
        with pytest.raises(TokenizerException, match="Unrecognized tokenizer filename"):
            get_spm_version(tokenizer_filename="tokenizer.model.vX")


class TestGetImageConfig:
    def test_instruct_only_path_returns_none(self, spm: SentencePieceTokenizer) -> None:
        assert get_image_config(tokenizer_filename=spm.file_path) is None

    def test_bare_model_extension_returns_none(self) -> None:
        assert get_image_config(tokenizer_filename="tokenizer.model") is None

    def test_known_multimodal_version_returns_config(self) -> None:
        assert get_image_config(tokenizer_filename="tokenizer.model.v7m1") is not None

    def test_unknown_multimodal_suffix_raises(self) -> None:
        with pytest.raises(TokenizerException, match="Unrecognized tokenizer filename"):
            get_image_config(tokenizer_filename="tokenizer.model.v7m99")


class TestIsSentencepieceTokenizer:
    def test_true_for_sentencepiece(self, spm: SentencePieceTokenizer) -> None:
        assert is_sentencepiece_tokenizer(spm) is True

    def test_false_for_tekkenizer(self) -> None:
        assert is_sentencepiece_tokenizer(build_tekkenizer(version=TokenizerVersion.v3)) is False


class TestSentencePieceTokenizer:
    # file_path
    def test_file_path_is_file(self, spm: SentencePieceTokenizer) -> None:
        assert spm.file_path.is_file()

    # n_words
    def test_n_words_positive(self, spm: SentencePieceTokenizer) -> None:
        assert spm.n_words > 0

    # vocab
    def test_vocab_length_matches_n_words(self, spm: SentencePieceTokenizer) -> None:
        assert len(spm.vocab()) == spm.n_words

    # bos_id / eos_id / pad_id / unk_id
    def test_bos_id_value(self, spm: SentencePieceTokenizer) -> None:
        assert spm.bos_id == 1

    def test_eos_id_value(self, spm: SentencePieceTokenizer) -> None:
        assert spm.eos_id == 2

    def test_pad_id_value(self, spm: SentencePieceTokenizer) -> None:
        assert spm.pad_id == -1

    def test_unk_id_value(self, spm: SentencePieceTokenizer) -> None:
        assert spm.unk_id == 0

    # model_settings_builder
    def test_model_settings_builder_returns_none(self, spm: SentencePieceTokenizer) -> None:
        assert spm.model_settings_builder is None

    def test_model_settings_builder_raises_for_version_with_settings(self, spm: SentencePieceTokenizer) -> None:
        tok = SentencePieceTokenizer(model_path=spm.file_path, tokenizer_version=TokenizerVersion.v15)
        with pytest.raises(ValueError, match="does not support model settings"):
            _ = tok.model_settings_builder

    # is_special
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            (1, True),
            (1001, False),
            (np.int64(1), True),
            (np.int64(1001), False),
            ("<s>", True),
            ("s", False),
        ],
    )
    def test_is_special_token_ids(self, spm: SentencePieceTokenizer, token: int | str, expected: bool) -> None:
        assert spm.is_special(token) is expected

    def test_is_special_invalid_type_raises_type_error(self, spm: SentencePieceTokenizer) -> None:
        with pytest.raises(TypeError, match="Expected int or str"):
            spm.is_special(3.14)  # type: ignore[arg-type]

    # special_ids
    def test_special_ids_is_nonempty_set(self, spm: SentencePieceTokenizer) -> None:
        special_ids = spm.special_ids
        assert isinstance(special_ids, set)
        assert len(special_ids) > 0

    # num_special_tokens
    def test_num_special_tokens_equals_len_special_ids(self, spm: SentencePieceTokenizer) -> None:
        assert spm.num_special_tokens == len(spm.special_ids)

    # get_special_token
    def test_get_special_token_returns_special_id(self, spm: SentencePieceTokenizer) -> None:
        token_id = spm.get_special_token(s="</s>")
        assert isinstance(token_id, int)
        assert spm.is_special(token_id)
        assert token_id == spm.eos_id

    # get_control_token (deprecated)
    def test_get_control_token_deprecated_emits_future_warning(self, spm: SentencePieceTokenizer) -> None:
        with pytest.warns(FutureWarning, match="get_control_token"):
            result = spm.get_control_token(s="</s>")
        assert isinstance(result, int)

    # _control_tokens (deprecated)
    def test_control_tokens_deprecated_property_emits_future_warning(self, spm: SentencePieceTokenizer) -> None:
        with pytest.warns(FutureWarning, match="_control_tokens"):
            control_tokens = spm._control_tokens
        assert control_tokens == spm.special_ids

    # id_to_piece
    def test_id_to_piece(self, spm: SentencePieceTokenizer) -> None:
        assert spm.id_to_piece(token_id=spm.bos_id) == "<s>"
        assert spm.id_to_piece(token_id=spm.eos_id) == "</s>"
        # roundtrip via id_to_piece: text -> ints -> pieces -> text
        text = "Hello world"
        ids = spm.encode(s=text, bos=False, eos=False)
        assert all(isinstance(token_id, int) for token_id in ids)
        reconstructed = "".join(spm.id_to_piece(token_id=token_id) for token_id in ids)
        assert reconstructed.replace("▁", " ").strip() == text

    # encode
    def test_encode_no_bos_no_eos_produces_only_normal_tokens(self, spm: SentencePieceTokenizer) -> None:
        ids = spm.encode(s="hello", bos=False, eos=False)
        assert all(not spm.is_special(t) for t in ids)

    @pytest.mark.parametrize("bos", [False, True])
    @pytest.mark.parametrize("eos", [False, True])
    def test_encode_bos_eos_markers(self, spm: SentencePieceTokenizer, bos: bool, eos: bool) -> None:
        ids = spm.encode(s="hello", bos=bos, eos=eos)
        assert (ids[0] == spm.bos_id) is bos
        assert (ids[-1] == spm.eos_id) is eos

    # decode
    @pytest.mark.parametrize("policy", [SpecialTokenPolicy.RAISE, SpecialTokenPolicy.IGNORE])
    def test_decode_normal_tokens_roundtrip(self, spm: SentencePieceTokenizer, policy: SpecialTokenPolicy) -> None:
        text = "Hello world, how are you?"
        ids = spm.encode(s=text, bos=False, eos=False)
        assert spm.decode(tokens=ids, special_token_policy=policy) == text

    def test_decode_raise_policy_raises_when_special_tokens_present(self, spm: SentencePieceTokenizer) -> None:
        ids = spm.encode(s="Hello world", bos=True, eos=True)
        assert any(spm.is_special(t) for t in ids)
        with pytest.raises(ValueError, match="special_token_policy=RAISE"):
            spm.decode(tokens=ids, special_token_policy=SpecialTokenPolicy.RAISE)

    @pytest.mark.parametrize("append_special", [False, True])
    def test_decode_keep_policy_matches_piecewise(self, spm: SentencePieceTokenizer, append_special: bool) -> None:
        ids = spm.encode(s="Hello world", bos=False, eos=False)
        if append_special:
            ids = ids + [spm.eos_id]
        expected = "".join(spm.id_to_piece(token_id=t) for t in ids)
        assert spm.decode(tokens=ids, special_token_policy=SpecialTokenPolicy.KEEP) == expected

    @pytest.mark.parametrize("policy", list(SpecialTokenPolicy))
    def test_decode_string_policy_matches_enum_policy(
        self, spm: SentencePieceTokenizer, policy: SpecialTokenPolicy
    ) -> None:
        include_special = policy is not SpecialTokenPolicy.RAISE
        ids = spm.encode(s="Hello world", bos=include_special, eos=include_special)
        assert spm.decode(tokens=ids, special_token_policy=policy.value) == spm.decode(  # type: ignore[arg-type]
            tokens=ids, special_token_policy=policy
        )

    def test_decode_invalid_policy_string_raises(self, spm: SentencePieceTokenizer) -> None:
        ids = spm.encode(s="Hello world", bos=True, eos=True)
        with pytest.raises(ValueError, match=r"Invalid `special_token_policy`"):
            spm.decode(tokens=ids, special_token_policy="keeep")  # type: ignore[arg-type]

    # _to_string
    def test_to_string(self, spm: SentencePieceTokenizer) -> None:
        ids = spm.encode(s="Hello world", bos=True, eos=True)
        assert spm._to_string(ids) == "<s>▁Hello▁world</s>"

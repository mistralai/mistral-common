from pathlib import Path

import pytest

from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.utils.tokenizers import (
    build_tekkenizer,
    build_tekkenizer_from_config,
    build_tekkenizer_from_file,
    deprecated_special_tokens,
    get_special_tokens,
    load_sentencepiece,
    quick_vocab,
    write_tekkenizer_model,
)
from tests.utils.versions import TestConfig


class TestQuickVocab:
    def test_quick_vocab_base_has_256_entries(self) -> None:
        assert len(quick_vocab()) == 256

    def test_quick_vocab_extra_toks_appended(self) -> None:
        vocab = quick_vocab(extra_toks=[b"hello", b"world"])
        assert len(vocab) == 258

    def test_quick_vocab_base_entries_are_single_bytes(self) -> None:
        vocab = quick_vocab()
        for i in range(256):
            assert vocab[i]["rank"] == i

    def test_quick_vocab_extra_tok_rank_starts_at_256(self) -> None:
        vocab = quick_vocab(extra_toks=[b"ab", b"cd"])
        assert vocab[256]["rank"] == 256
        assert vocab[257]["rank"] == 257

    def test_quick_vocab_extra_tok_token_str_decoded(self) -> None:
        vocab = quick_vocab(extra_toks=[b"hello"])
        assert vocab[256]["token_str"] == "hello"

    def test_quick_vocab_empty_extra_toks_is_pure_base(self) -> None:
        assert quick_vocab(extra_toks=()) == quick_vocab()


class TestDeprecatedSpecialTokens:
    def test_deprecated_special_tokens_has_20_entries(self) -> None:
        assert len(deprecated_special_tokens()) == 20

    def test_deprecated_special_tokens_returns_new_list_each_call(self) -> None:
        first = deprecated_special_tokens()
        second = deprecated_special_tokens()
        assert first is not second

    def test_deprecated_special_tokens_equals_source(self) -> None:
        assert deprecated_special_tokens() == list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)

    def test_deprecated_special_tokens_mutation_does_not_affect_source(self) -> None:
        tokens = deprecated_special_tokens()
        tokens.clear()
        assert len(deprecated_special_tokens()) == 20


class TestGetSpecialTokens:
    def test_get_special_tokens_audio_guard_v3_raises(self) -> None:
        with pytest.raises(ValueError, match="Audio tokens are only supported in v7 and above"):
            get_special_tokens(tokenizer_version=TokenizerVersion.v3, add_audio=True)

    def test_get_special_tokens_audio_guard_v1_raises(self) -> None:
        with pytest.raises(ValueError, match="Audio tokens are only supported in v7 and above"):
            get_special_tokens(tokenizer_version=TokenizerVersion.v1, add_audio=True)

    def test_get_special_tokens_v3_no_audio_returns_deprecated(self) -> None:
        tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v3)
        assert tokens == deprecated_special_tokens()

    def test_get_special_tokens_v7_no_audio_returns_deprecated(self) -> None:
        tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v7)
        assert tokens == deprecated_special_tokens()

    def test_get_special_tokens_v7_audio_contains_audio_tokens(self) -> None:
        tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v7, add_audio=True)
        token_strs = {t["token_str"] for t in tokens}
        assert SpecialTokens.audio in token_strs
        assert SpecialTokens.begin_audio in token_strs

    def test_get_special_tokens_v13_think_contains_think_tokens(self) -> None:
        tokens = get_special_tokens(tokenizer_version=TokenizerVersion.v13, add_think=True)
        token_strs = {t["token_str"] for t in tokens}
        assert SpecialTokens.begin_think in token_strs
        assert SpecialTokens.end_think in token_strs

    def test_get_special_tokens_returns_new_list_each_call(self) -> None:
        first = get_special_tokens(tokenizer_version=TokenizerVersion.v3)
        second = get_special_tokens(tokenizer_version=TokenizerVersion.v3)
        assert first is not second


class TestBuildTekkenizer:
    def test_build_tekkenizer_roundtrip(self) -> None:
        tkz = build_tekkenizer(version=TokenizerVersion.v3)
        encoded = tkz.encode("hello", bos=False, eos=False)
        decoded = tkz.decode(encoded)
        assert decoded == "hello"

    def test_build_tekkenizer_version_attribute(self) -> None:
        tkz = build_tekkenizer(version=TokenizerVersion.v3)
        assert tkz.version == TokenizerVersion.v3

    def test_build_tekkenizer_num_special_tokens_attribute(self) -> None:
        tkz = build_tekkenizer(version=TokenizerVersion.v3, num_special_tokens=50)
        assert tkz.num_special_tokens == 50

    def test_build_tekkenizer_audio_version(self) -> None:
        tkz = build_tekkenizer(version=TokenizerVersion.v7, add_audio=True)
        assert tkz.version == TokenizerVersion.v7
        token_strs = {t["token_str"] for t in tkz._all_special_tokens}
        assert SpecialTokens.audio in token_strs
        assert SpecialTokens.begin_audio in token_strs

    def test_build_tekkenizer_think(self) -> None:
        tkz = build_tekkenizer(version=TokenizerVersion.v13, add_think=True)
        token_strs = {t["token_str"] for t in tkz._all_special_tokens}
        assert SpecialTokens.begin_think in token_strs
        assert SpecialTokens.end_think in token_strs

    def test_build_tekkenizer_audio_guard_propagates(self) -> None:
        with pytest.raises(ValueError, match="Audio tokens are only supported in v7 and above"):
            build_tekkenizer(version=TokenizerVersion.v3, add_audio=True)

    def test_build_tekkenizer_custom_vocab_size(self) -> None:
        tkz = build_tekkenizer(version=TokenizerVersion.v3, num_special_tokens=50, vocab_size=306)
        assert tkz.num_special_tokens == 50
        assert tkz.n_words == 306

    def test_build_tekkenizer_default_vocab_size(self) -> None:
        tkz = build_tekkenizer(version=TokenizerVersion.v3, num_special_tokens=50)
        assert tkz.n_words == 256 + 50

    def test_build_tekkenizer_returns_tekkenizer_instance(self) -> None:
        tkz = build_tekkenizer(version=TokenizerVersion.v3)
        assert isinstance(tkz, Tekkenizer)

    def test_build_tekkenizer_v7_version(self) -> None:
        tkz = build_tekkenizer(version=TokenizerVersion.v7)
        assert tkz.version == TokenizerVersion.v7

    def test_build_tekkenizer_v13_version(self) -> None:
        tkz = build_tekkenizer(version=TokenizerVersion.v13)
        assert tkz.version == TokenizerVersion.v13

    def test_build_tekkenizer_model_settings_builder_supported_version(self) -> None:
        tkz = build_tekkenizer(
            version=TokenizerVersion.v15,
            model_settings_builder=ModelSettingsBuilder(reasoning_effort=None),
        )
        assert tkz.model_settings_builder == ModelSettingsBuilder(reasoning_effort=None)


class TestBuildTekkenizerFromFile:
    def test_default_vocab_loads_v3(self, tmp_path: Path) -> None:
        tkz = build_tekkenizer_from_file(tmp_path)
        assert isinstance(tkz, Tekkenizer)
        assert tkz.version == TokenizerVersion.v3

    def test_explicit_version(self, tmp_path: Path) -> None:
        tkz = build_tekkenizer_from_file(
            tmp_path,
            version="v7",
            special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v7),
        )
        assert tkz.version == TokenizerVersion.v7

    def test_explicit_vocab(self, tmp_path: Path) -> None:
        vocab = quick_vocab(extra_toks=[b"beau", b"My", b"unused"])
        tkz = build_tekkenizer_from_file(tmp_path, vocab=vocab, version="v3")
        assert tkz.version == TokenizerVersion.v3

    def test_file_path_matches_filename(self, tmp_path: Path) -> None:
        tkz = build_tekkenizer_from_file(tmp_path, version="v3", filename="custom.json")
        assert tkz.file_path == tmp_path / "custom.json"


class TestBuildTekkenizerFromConfig:
    def test_base_config_builds_in_memory(self) -> None:
        tkz = build_tekkenizer_from_config(TestConfig(version=TokenizerVersion.v3))
        assert isinstance(tkz, Tekkenizer)
        assert tkz.version == TokenizerVersion.v3

    def test_audio_config_has_audio_tokens_and_config(self, tmp_path: Path) -> None:
        tkz = build_tekkenizer_from_config(TestConfig(version=TokenizerVersion.v7, audio=True), output_dir=tmp_path)
        token_strs = {t["token_str"] for t in tkz._all_special_tokens}
        assert SpecialTokens.audio in token_strs
        assert tkz.audio is not None

    def test_image_config_builds_from_file(self, tmp_path: Path) -> None:
        tkz = build_tekkenizer_from_config(TestConfig(version=TokenizerVersion.v3, image=True), output_dir=tmp_path)
        assert tkz.image is not None

    def test_image_config_without_output_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="output_dir is required"):
            build_tekkenizer_from_config(TestConfig(version=TokenizerVersion.v3, image=True))

    def test_audio_config_without_output_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="output_dir is required"):
            build_tekkenizer_from_config(TestConfig(version=TokenizerVersion.v7, audio=True))


class TestLoadSentencepiece:
    def test_load_sentencepiece_v7_num_special_tokens(self) -> None:
        spm = load_sentencepiece(version="v7")
        assert spm.num_special_tokens == 748

    def test_load_sentencepiece_v7_returns_sentencepiece_instance(self) -> None:
        spm = load_sentencepiece(version="v7")
        assert isinstance(spm, SentencePieceTokenizer)

    def test_load_sentencepiece_missing_version_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_sentencepiece(version="v99")


class TestWriteTekkenizerModel:
    def test_write_then_from_file_returns_expected_version(self, tmp_path: Path) -> None:
        # write_tekkenizer_model hardcodes default_vocab_size = 256 + 3 + num_special_tokens,
        # so the vocab must have exactly 3 multi-byte extra tokens beyond the 256-byte base.
        vocab = quick_vocab(extra_toks=[b"beau", b"My", b"unused"])
        tokpath = tmp_path / "tekken.tokenizer.json"
        write_tekkenizer_model(tmp_path=tokpath, vocab=vocab, version="v3")
        loaded = Tekkenizer.from_file(tokpath)
        assert loaded.version == TokenizerVersion.v3

    def test_write_then_from_file_custom_version(self, tmp_path: Path) -> None:
        vocab = quick_vocab(extra_toks=[b"beau", b"My", b"unused"])
        tokpath = tmp_path / "tekken.tokenizer.json"
        write_tekkenizer_model(tmp_path=tokpath, vocab=vocab, version="v7")
        loaded = Tekkenizer.from_file(tokpath)
        assert loaded.version == TokenizerVersion.v7

    def test_written_pattern_preserves_newlines_tabs_and_unicode(self, tmp_path: Path) -> None:
        vocab = quick_vocab(extra_toks=[b"beau", b"My", b"unused"])
        tokpath = tmp_path / "tekken.tokenizer.json"
        write_tekkenizer_model(tmp_path=tokpath, vocab=vocab, version="v3")
        loaded = Tekkenizer.from_file(tokpath)
        text = "line1\nline2\ttab\u00e9"
        encoded = loaded.encode(text, bos=False, eos=False)
        assert loaded.decode(encoded) == text

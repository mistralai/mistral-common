from pathlib import Path

import numpy as np
import pytest

from mistral_common.tokens.tokenizers.audio import AudioConfig, AudioSpectrogramConfig
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.image import ImageConfig
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder
from mistral_common.tokens.tokenizers.tekken import Tekkenizer, _reload_mergeable_ranks, is_tekken, is_tekkenizer
from tests.utils.tokenizers import (
    build_tekkenizer,
    build_tekkenizer_from_file,
    deprecated_special_tokens,
    get_special_tokens,
    quick_vocab,
    write_tekkenizer_model,
)
from tests.utils.versions import (
    AUDIO_TEKKEN_CONFIGS,
    BASE_TEKKEN_CONFIGS,
    IMAGE_TEKKEN_CONFIGS,
    MODEL_SETTINGS_VERSIONS,
    TEKKEN_VERSIONS,
    THINK_TEKKEN_CONFIGS,
    config_id,
)

# Reusable narrowings of the `tekkenizer` fixture over subsets of the full matrix.
base_tekken = pytest.mark.parametrize("tekkenizer", BASE_TEKKEN_CONFIGS, indirect=True, ids=config_id)
audio_tekken = pytest.mark.parametrize("tekkenizer", AUDIO_TEKKEN_CONFIGS, indirect=True, ids=config_id)
think_tekken = pytest.mark.parametrize("tekkenizer", THINK_TEKKEN_CONFIGS, indirect=True, ids=config_id)
image_tekken = pytest.mark.parametrize("tekkenizer", IMAGE_TEKKEN_CONFIGS, indirect=True, ids=config_id)

_FROZEN_DEPRECATED_TOKENS = [  # DO NOT MODIFY
    "<unk>",
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
    "[TOOL_CALLS]",
    "[IMG]",
    "<pad>",
    "[IMG_BREAK]",
    "[IMG_END]",
    "[PREFIX]",
    "[MIDDLE]",
    "[SUFFIX]",
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "[TOOL_CONTENT]",
]
_N_DEPRECATED = len(_FROZEN_DEPRECATED_TOKENS)


@pytest.fixture(scope="module")
def tekkenizer_minimal() -> Tekkenizer:
    r"""Tekkenizer with `num_special_tokens` equal to the deprecated list length.

    This size makes token IDs for special and merged tokens easy to reason about in
    `id_to_byte_piece`, `is_byte`, and `special_ids` tests. Not a version/modality combo,
    so it is a dedicated fixture rather than part of the matrix.

    Returns:
        A v3 `Tekkenizer` with 20 special tokens and one merged token ``b"hello"``.
    """
    return build_tekkenizer(
        version=TokenizerVersion.v3,
        extra_toks=[b"hello"],
        num_special_tokens=_N_DEPRECATED,
        vocab_size=256 + 1 + _N_DEPRECATED,
    )


@pytest.fixture(scope="module")
def merge_tekkenizer() -> Tekkenizer:
    r"""Tekkenizer whose vocabulary admits no BPE merge, used for byte-level roundtrip tests.

    ``extra_toks=()`` leaves only the 256 single-byte tokens, so no merge into a compound
    token can ever occur and encoding ``"My very beautiful string"`` yields one token per
    character. Chunking uses the production `TEKKEN_PATTERN`, which `build_tekkenizer` does
    not let callers override — the empty vocabulary already rules merges out, so nothing here
    needs a narrower pattern.

    Returns:
        A v3 `Tekkenizer` with 100 special tokens and no mergeable multi-byte tokens.
    """
    return build_tekkenizer(
        version=TokenizerVersion.v3,
        extra_toks=(),
        num_special_tokens=100,
    )


class TestIsTekken:
    @pytest.mark.parametrize("name", ["tekken.tokenizer.json", "tekken_the_destroyer.tekken.json", "v4.tekken.json"])
    def test_valid_cases(self, tmp_path: Path, name: str) -> None:
        path = tmp_path / name
        write_tekkenizer_model(tmp_path=path)
        assert is_tekken(path)

    @pytest.mark.parametrize("name", ["sentencepiece.json", "tekken.model", "vocab.json"])
    def test_bad_name_cases(self, tmp_path: Path, name: str) -> None:
        path = tmp_path / name
        write_tekkenizer_model(tmp_path=path)
        assert not is_tekken(path)

    def test_nonexistent_returns_false(self, tmp_path: Path) -> None:
        assert not is_tekken(tmp_path / "nonexistent.tekken.json")

    def test_str_path_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "str_input.tekken.json"
        write_tekkenizer_model(tmp_path=path)
        assert is_tekken(str(path))


class TestIsTekkenizer:
    def test_true_for_tekkenizer(self, tekkenizer: Tekkenizer) -> None:
        assert is_tekkenizer(tekkenizer)

    def test_false_for_other(self, spm: object) -> None:
        assert not is_tekkenizer(spm)  # type: ignore[arg-type]


class TestReloadMergeableRanks:
    def test_reload_mergeable_ranks_without_max_vocab_keeps_every_entry(self) -> None:
        vocab = quick_vocab(extra_toks=[b"ab", b"cd"])
        ranks = _reload_mergeable_ranks(vocab=vocab, max_vocab=None)
        assert len(ranks) == len(vocab)
        assert set(ranks.values()) == set(range(len(vocab)))

    def test_reload_mergeable_ranks_with_max_vocab_truncates(self) -> None:
        vocab = quick_vocab(extra_toks=[b"ab", b"cd"])
        ranks = _reload_mergeable_ranks(vocab=vocab, max_vocab=256)
        assert len(ranks) == 256


class TestTekkenizer:
    # __init__ / special tokens by modality
    @audio_tekken
    def test_init_audio_special_tokens_present(self, tekkenizer: Tekkenizer) -> None:
        token_strs = {t["token_str"] for t in tekkenizer._all_special_tokens}
        assert SpecialTokens.audio in token_strs
        assert SpecialTokens.begin_audio in token_strs
        assert tekkenizer.audio is not None

    @think_tekken
    def test_init_think_special_tokens_present(self, tekkenizer: Tekkenizer) -> None:
        token_strs = {t["token_str"] for t in tekkenizer._all_special_tokens}
        assert SpecialTokens.begin_think in token_strs
        assert SpecialTokens.end_think in token_strs

    @image_tekken
    def test_init_image_config_present(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.image == ImageConfig(image_patch_size=2, max_image_size=10, spatial_merge_size=1)

    def test_init_model_settings_builder_wrong_version_raises(self) -> None:
        with pytest.raises(ValueError, match="model_settings_builder is not supported"):
            build_tekkenizer(
                version=TokenizerVersion.v3,
                model_settings_builder=ModelSettingsBuilder(reasoning_effort=None),
            )

    # from_file
    @pytest.mark.parametrize("version", TEKKEN_VERSIONS, ids=[v.value for v in TEKKEN_VERSIONS])
    def test_from_file_all_versions_load(self, tmp_path: Path, version: TokenizerVersion) -> None:
        loaded = build_tekkenizer_from_file(
            tmp_path,
            version=version.value,
            special_tokens=get_special_tokens(tokenizer_version=version),
        )
        assert loaded.version == version

    def test_from_file_none_version_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown version"):
            build_tekkenizer_from_file(
                tmp_path,
                version=None,
                special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v3),
            )

    def test_from_file_unknown_version_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown version"):
            build_tekkenizer_from_file(
                tmp_path,
                version="dummy-v",
                special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v3),
            )

    def test_from_file_roundtrip_encode(self, tmp_path: Path) -> None:
        loaded = build_tekkenizer_from_file(tmp_path, version="v3", special_tokens=None)
        # Same vocab/pattern as build_tekkenizer_from_file's defaults, built in memory instead
        # of via the file-writing path, so the two must encode identically.
        equivalent = build_tekkenizer(
            version=TokenizerVersion.v3,
            extra_toks=[b"beau", b"My", b"unused"],
            num_special_tokens=100,
            vocab_size=256 + 3 + 100,
        )
        text = "My very beautiful string"
        assert loaded.encode(text, bos=False, eos=False) == equivalent.encode(text, bos=False, eos=False)
        assert loaded.num_special_tokens == 100
        assert loaded.version == TokenizerVersion.v3

    def test_from_file_str_path_accepted(self, tmp_path: Path) -> None:
        # Write via the util, then load with a str path to exercise the str-path branch.
        build_tekkenizer_from_file(tmp_path, version="v3", special_tokens=None, filename="tekken_strpath.json")
        loaded = Tekkenizer.from_file(str(tmp_path / "tekken_strpath.json"))
        assert loaded.version == TokenizerVersion.v3

    def test_from_file_missing_special_tokens_v11_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Special tokens not found"):
            build_tekkenizer_from_file(tmp_path, version="v11", special_tokens=None)

    def test_from_file_with_audio_config(self, tmp_path: Path) -> None:
        loaded = build_tekkenizer_from_file(
            tmp_path,
            version="v7",
            special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v7, add_audio=True),
            audio={
                "sampling_rate": 16000,
                "frame_rate": 12.5,
                "audio_encoding_config": {"num_mel_bins": 80, "hop_length": 160, "window_size": 400},
            },
        )
        assert loaded.audio == AudioConfig(
            sampling_rate=16000,
            frame_rate=12.5,
            encoding_config=AudioSpectrogramConfig(num_mel_bins=80, hop_length=160, window_size=400),
        )

    def test_from_file_with_image_config(self, tmp_path: Path) -> None:
        loaded = build_tekkenizer_from_file(
            tmp_path,
            version="v13",
            special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v13),
            image={"image_patch_size": 16, "max_image_size": 1024},
        )
        assert loaded.image == ImageConfig(image_patch_size=16, max_image_size=1024)

    def test_from_file_with_multimodal_config_v11(self, tmp_path: Path) -> None:
        # The deprecated `multimodal` key is only allowed up to v11.
        loaded = build_tekkenizer_from_file(
            tmp_path,
            version="v11",
            special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v11),
            multimodal={"image_patch_size": 16, "max_image_size": 1024},
        )
        assert loaded.image == ImageConfig(image_patch_size=16, max_image_size=1024)

    def test_from_file_multimodal_v13_raises(self, tmp_path: Path) -> None:
        # Beyond v11 the deprecated `multimodal` key is rejected in favour of `image`.
        with pytest.raises(ValueError, match="has to be called 'image'"):
            build_tekkenizer_from_file(
                tmp_path,
                version="v13",
                special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v13),
                multimodal={"image_patch_size": 16, "max_image_size": 1024},
            )

    @pytest.mark.parametrize("version", MODEL_SETTINGS_VERSIONS, ids=[v.value for v in MODEL_SETTINGS_VERSIONS])
    def test_from_file_with_model_settings_builder(self, tmp_path: Path, version: TokenizerVersion) -> None:
        loaded = build_tekkenizer_from_file(
            tmp_path,
            version=version.value,
            special_tokens=get_special_tokens(tokenizer_version=version),
            model_settings_builder={"reasoning_effort": None},
        )
        assert loaded.model_settings_builder == ModelSettingsBuilder(reasoning_effort=None)

    def test_from_file_model_settings_builder_wrong_version_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="model_settings_builder is not supported"):
            build_tekkenizer_from_file(
                tmp_path,
                version="v13",
                special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v13),
                model_settings_builder={"reasoning_effort": None},
            )

    # decode
    def test_decode_roundtrip(self, merge_tekkenizer: Tekkenizer) -> None:
        text = "My very beautiful string"
        encoded = merge_tekkenizer.encode(text, bos=False, eos=False)
        assert merge_tekkenizer.decode(encoded) == text

    @base_tekken
    @pytest.mark.parametrize("policy", list(SpecialTokenPolicy))
    def test_decode_string_policy_matches_enum(self, tekkenizer: Tekkenizer, policy: SpecialTokenPolicy) -> None:
        include_special = policy is not SpecialTokenPolicy.RAISE
        ids = tekkenizer.encode("hello", bos=include_special, eos=include_special)
        assert tekkenizer.decode(ids, policy.value) == tekkenizer.decode(ids, policy)  # type: ignore[arg-type]

    @base_tekken
    def test_decode_invalid_policy_raises(self, tekkenizer: Tekkenizer) -> None:
        ids = tekkenizer.encode("hello", bos=True, eos=True)
        with pytest.raises(ValueError, match=r"Invalid \`special_token_policy\`"):
            tekkenizer.decode(ids, "keeep")  # type: ignore[arg-type]

    @base_tekken
    def test_decode_raise_policy_on_special_tokens(self, tekkenizer: Tekkenizer) -> None:
        ids = tekkenizer.encode("hello", bos=True, eos=True)
        with pytest.raises(ValueError, match="Decoding `tokens` that contain special tokens"):
            tekkenizer.decode(ids, special_token_policy=SpecialTokenPolicy.RAISE)

    @base_tekken
    def test_decode_all_invalid_policy_raises(self, tekkenizer: Tekkenizer) -> None:
        ids = tekkenizer.encode("hello", bos=True, eos=True)
        with pytest.raises(ValueError, match=r"Invalid `special_token_policy`"):
            tekkenizer._decode_all(ids, special_token_policy="keeep")  # type: ignore[arg-type]

    @base_tekken
    def test_decode_ignore_policy_on_special_tokens_between_normal_tokens(self, tekkenizer: Tekkenizer) -> None:
        head = tekkenizer.encode("hello", bos=False, eos=False)
        tail = tekkenizer.encode("world", bos=False, eos=False)
        ids = head + [tekkenizer.bos_id, tekkenizer.eos_id] + tail
        assert tekkenizer.decode(ids, special_token_policy=SpecialTokenPolicy.IGNORE) == "helloworld"

    # is_byte
    def test_is_byte_merged_token_not_byte(self, tekkenizer_minimal: Tekkenizer) -> None:
        tok = tekkenizer_minimal.encode("hello", bos=False, eos=False)
        assert len(tok) == 1
        assert not tekkenizer_minimal.is_byte(tok[0])

    def test_is_byte_byte_token_is_byte(self, tekkenizer_minimal: Tekkenizer) -> None:
        byte_tok = tekkenizer_minimal.encode(chr(0), bos=False, eos=False)
        assert len(byte_tok) == 1
        assert tekkenizer_minimal.is_byte(byte_tok[0])

    def test_is_byte_id_ordering(self, tekkenizer_minimal: Tekkenizer) -> None:
        merged_tok = tekkenizer_minimal.encode("hello", bos=False, eos=False)
        byte_tok = tekkenizer_minimal.encode(chr(0), bos=False, eos=False)
        assert byte_tok[0] < 256 + tekkenizer_minimal.num_special_tokens <= merged_tok[0]

    # id_to_byte_piece
    def test_id_to_byte_piece_raise_policy(self, tekkenizer_minimal: Tekkenizer) -> None:
        with pytest.raises(ValueError, match="5 is a special token"):
            tekkenizer_minimal.id_to_byte_piece(token_id=5, special_token_policy=SpecialTokenPolicy.RAISE)

    def test_id_to_byte_piece_ignore_policy(self, tekkenizer_minimal: Tekkenizer) -> None:
        assert tekkenizer_minimal.id_to_byte_piece(token_id=5, special_token_policy=SpecialTokenPolicy.IGNORE) == b""

    def test_id_to_byte_piece_keep_policy(self, tekkenizer_minimal: Tekkenizer) -> None:
        assert (
            tekkenizer_minimal.id_to_byte_piece(token_id=5, special_token_policy=SpecialTokenPolicy.KEEP)
            == b"[AVAILABLE_TOOLS]"
        )

    def test_id_to_byte_piece_merged_token_keep(self, tekkenizer_minimal: Tekkenizer) -> None:
        # Last non-filler token: internal rank 256, external ID = 256 + num_special_tokens.
        hello_id = 256 + tekkenizer_minimal.num_special_tokens
        assert (
            tekkenizer_minimal.id_to_byte_piece(token_id=hello_id, special_token_policy=SpecialTokenPolicy.KEEP)
            == b"hello"
        )

    def test_id_to_byte_piece_invalid_policy_raises(self, tekkenizer_minimal: Tekkenizer) -> None:
        # Message text is the pre-existing one: this arc was already reachable before the
        # refactor, so PR1 restructures it to `match` without changing what it raises.
        with pytest.raises(ValueError, match=r"Unknown special token policy"):
            tekkenizer_minimal.id_to_byte_piece(token_id=5, special_token_policy="keeep")  # type: ignore[arg-type]

    # is_special
    @base_tekken
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("</s>", True),
            ("a", False),
            (1, True),
            (1001, False),
            (np.int64(0), True),
            (np.int64(1), True),
            (np.int64(1001), False),
        ],
    )
    def test_is_special(self, tekkenizer: Tekkenizer, token: str | int, expected: bool) -> None:
        assert tekkenizer.is_special(token) is expected

    @base_tekken
    def test_is_special_type_error(self, tekkenizer: Tekkenizer) -> None:
        with pytest.raises(TypeError, match="Expected int or str"):
            tekkenizer.is_special(3.14)  # type: ignore[arg-type]

    # get_special_token
    @base_tekken
    def test_get_special_token_known(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.get_special_token("<s>") == tekkenizer.bos_id

    @base_tekken
    def test_get_special_token_unknown_raises(self, tekkenizer: Tekkenizer) -> None:
        with pytest.raises(ValueError, match="Unknown control token"):
            tekkenizer.get_special_token("NOT_A_REAL_TOKEN")

    # get_control_token (deprecated)
    @base_tekken
    def test_get_control_token_deprecated_warns_and_delegates(self, tekkenizer: Tekkenizer) -> None:
        with pytest.warns(FutureWarning, match="`get_control_token` is deprecated"):
            result = tekkenizer.get_control_token("<s>")
        assert result == tekkenizer.bos_id

    # _to_string
    @base_tekken
    def test_to_string(self, tekkenizer: Tekkenizer) -> None:
        result = tekkenizer._to_string([tekkenizer.bos_id, tekkenizer.eos_id])
        assert "<s>" in result
        assert "</s>" in result

    # special_ids
    def test_special_ids_maps_to_deprecated_ranks(self, tekkenizer_minimal: Tekkenizer) -> None:
        # tekkenizer_minimal has num_special_tokens == len(DEPRECATED_SPECIAL_TOKENS) == 20,
        # so special_ids maps exactly to the deprecated token ranks.
        special_ids = tekkenizer_minimal.special_ids
        assert isinstance(special_ids, set)
        assert special_ids == {t["rank"] for t in deprecated_special_tokens()}

    # num_special_tokens
    @base_tekken
    def test_num_special_tokens_matches_all_special_tokens(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.num_special_tokens == len(tekkenizer._all_special_tokens)

    # n_words
    @base_tekken
    def test_n_words_matches_vocab_size(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.n_words == tekkenizer._vocab_size

    # vocab
    @base_tekken
    def test_vocab_length_matches_n_words(self, tekkenizer: Tekkenizer) -> None:
        v = tekkenizer.vocab()
        assert isinstance(v, list)
        assert len(v) == tekkenizer.n_words

    # bos_id / eos_id / pad_id / unk_id
    @base_tekken
    def test_bos_id(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.bos_id == tekkenizer.get_special_token("<s>")

    @base_tekken
    def test_eos_id(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.eos_id == tekkenizer.get_special_token("</s>")

    @base_tekken
    def test_pad_id(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.pad_id == tekkenizer.get_special_token("<pad>")

    @base_tekken
    def test_unk_id(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.unk_id == tekkenizer.get_special_token("<unk>")

    # file_path
    @base_tekken
    def test_file_path_raises_when_not_from_file(self, tekkenizer: Tekkenizer) -> None:
        with pytest.raises(ValueError, match="not loaded from a file"):
            _ = tekkenizer.file_path

    def test_file_path_returns_path_when_from_file(self, tmp_path: Path) -> None:
        loaded = build_tekkenizer_from_file(tmp_path, version="v3", special_tokens=None, filename="tekken_fp.json")
        assert loaded.file_path == tmp_path / "tekken_fp.json"

    # model_settings_builder
    def test_model_settings_builder_getter(self, tmp_path: Path) -> None:
        loaded = build_tekkenizer_from_file(
            tmp_path,
            version="v15",
            special_tokens=get_special_tokens(tokenizer_version=TokenizerVersion.v15),
            model_settings_builder={"reasoning_effort": None},
        )
        assert loaded.model_settings_builder == ModelSettingsBuilder(reasoning_effort=None)

    # image
    @base_tekken
    def test_image_getter_none_by_default(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.image is None

    @base_tekken
    def test_image_setter_raises(self, tekkenizer: Tekkenizer) -> None:
        with pytest.raises(ValueError, match="Can only set Image config at init"):
            tekkenizer.image = ImageConfig(image_patch_size=16, max_image_size=1024)

    # audio
    @base_tekken
    def test_audio_getter_none_by_default(self, tekkenizer: Tekkenizer) -> None:
        assert tekkenizer.audio is None

    @base_tekken
    def test_audio_setter_raises(self, tekkenizer: Tekkenizer) -> None:
        audio_config = AudioConfig(
            sampling_rate=16000,
            frame_rate=12.5,
            encoding_config=AudioSpectrogramConfig(num_mel_bins=80, hop_length=160, window_size=400),
        )
        with pytest.raises(ValueError, match="Can only set Audio config at init"):
            tekkenizer.audio = audio_config

    # DEPRECATED_SPECIAL_TOKENS
    def test_deprecated_special_tokens_length(self) -> None:
        assert len(deprecated_special_tokens()) == _N_DEPRECATED

    def test_deprecated_special_tokens_exact_strings(self) -> None:
        assert _FROZEN_DEPRECATED_TOKENS == [t["token_str"] for t in deprecated_special_tokens()]

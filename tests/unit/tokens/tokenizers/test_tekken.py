import base64
import json
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import pytest

from mistral_common.protocol.instruct.request import ReasoningEffort
from mistral_common.tokens.tokenizers.audio import AudioConfig, AudioSpectrogramConfig
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy, SpecialTokens, Tokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.image import ImageConfig
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder, ReasoningEffortEnumBuilder
from mistral_common.tokens.tokenizers.tekken import (
    SpecialTokenInfo,
    Tekkenizer,
    TokenInfo,
    is_tekken,
    is_tekkenizer,
)


def quick_vocab(extra_toks: Sequence[bytes]) -> list[TokenInfo]:
    r"""Synthetic 256-byte vocab plus `extra_toks`; identical contract to the legacy shared builder."""
    vocab = [TokenInfo(rank=i, token_bytes=base64.b64encode(bytes([i])).decode(), token_str=chr(i)) for i in range(256)]
    vocab.extend(
        TokenInfo(rank=256 + i, token_bytes=base64.b64encode(token).decode(), token_str=token.decode())
        for i, token in enumerate(extra_toks)
    )
    return vocab


def deprecated_special_tokens() -> list[SpecialTokenInfo]:
    return list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)


def special_tokens_with_fillers(num_special_tokens: int) -> list[SpecialTokenInfo]:
    special_tokens = deprecated_special_tokens()
    special_tokens.extend(
        SpecialTokenInfo(rank=rank, token_str=f"<SPECIAL_{rank}>", is_control=True)
        for rank in range(len(special_tokens), num_special_tokens)
    )
    return special_tokens


def build_tekkenizer_v3(
    *,
    num_special_tokens: int,
    special_tokens: list[SpecialTokenInfo],
    vocab: list[TokenInfo],
    image_config: ImageConfig,
    audio_config: AudioConfig,
    model_settings_builder: ModelSettingsBuilder,
    path: str | Path,
) -> Tekkenizer:
    return Tekkenizer(
        vocab=vocab,
        special_tokens=special_tokens,
        pattern=r".+",
        vocab_size=len(vocab) + num_special_tokens,
        num_special_tokens=num_special_tokens,
        version=TokenizerVersion.v3,
        image_config=image_config,
        audio_config=audio_config,
        model_settings_builder=model_settings_builder,
        _path=path,
    )


def build_tekkenizer_with_vocab(*, vocab: list[TokenInfo], vocab_size: int) -> Tekkenizer:
    r"""Build a v3 tokenizer from a raw vocab with an explicit total vocab size."""
    return Tekkenizer(
        vocab=vocab,
        special_tokens=deprecated_special_tokens(),
        pattern=r".+",
        vocab_size=vocab_size,
        num_special_tokens=20,
        version=TokenizerVersion.v3,
    )


def build_none_only_reasoning_effort_model_settings_builder() -> ModelSettingsBuilder:
    return ModelSettingsBuilder(
        reasoning_effort=ReasoningEffortEnumBuilder(
            values=[ReasoningEffort.none],
            default=ReasoningEffort.none,
        )
    )


def audio_config() -> AudioConfig:
    return AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(num_mel_bins=128, hop_length=160, window_size=400),
    )


def write_tekkenizer_file(
    path: Path,
    *,
    version: str | None,
    special_tokens: list[SpecialTokenInfo] | None,
    num_special_tokens: int,
    vocab: list[TokenInfo],
    image: dict[str, int],
    multimodal: dict[str, int],
    audio: dict[str, object],
    model_settings: dict[str, object],
) -> None:
    config: dict[str, object] = {
        "pattern": r".+",
        "num_vocab_tokens": len(vocab),
        "default_vocab_size": len(vocab) + num_special_tokens,
        "default_num_special_tokens": num_special_tokens,
    }
    model: dict[str, object] = {
        "vocab": vocab,
        "special_tokens": special_tokens,
        "config": config,
        "version": 1,
        "type": "Tekken",
    }
    if version is not None:
        config["version"] = version
    if image:
        model["image"] = image
    if multimodal:
        model["multimodal"] = multimodal
    if audio:
        model["audio"] = audio
    if model_settings:
        model["model_settings_builder"] = model_settings
    path.write_text(json.dumps(model), encoding="utf-8")


def tekkenizer_v3() -> Tekkenizer:
    return build_tekkenizer_v3(
        num_special_tokens=20,
        special_tokens=deprecated_special_tokens(),
        vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
        image_config=ImageConfig(image_patch_size=16, max_image_size=1024),
        audio_config=audio_config(),
        model_settings_builder=cast(ModelSettingsBuilder, None),
        path="synthetic-v3.tekken.json",
    )


@pytest.fixture(scope="module")
def tekkenizer() -> Tekkenizer:
    return tekkenizer_v3()


@pytest.mark.parametrize(
    "filename",
    ["tekken.tokenizer.json", "v4.tekken.json", "tekken_the_destroyer.tekken.json", "model.tekken.json"],
    ids=["tekken-suffix", "versioned-tekken", "tekken-prefix", "plain-tekken"],
)
def test_is_tekken_accepts_existing_json_file_with_marker(tmp_path: Path, filename: str) -> None:
    path = tmp_path / filename
    path.write_text("{}", encoding="utf-8")

    assert is_tekken(path)
    assert is_tekken(str(path))


@pytest.mark.parametrize(
    "filename",
    ["model.json", "sentencepiece.json", "model.tekken.bin", "missing.tekken.json"],
    ids=["plain-json", "wrong-name", "wrong-suffix", "missing-file"],
)
def test_is_tekken_rejects_wrong_name_suffix_or_missing_file(tmp_path: Path, filename: str) -> None:
    assert not is_tekken(tmp_path / filename)


def test_is_tekken_rejects_directory_with_tekken_name(tmp_path: Path) -> None:
    path = tmp_path / "directory.tekken.json"
    path.mkdir()

    assert not is_tekken(path)


@pytest.mark.parametrize(
    ("version", "model_settings_builder", "expected_builder"),
    [
        (TokenizerVersion.v3, cast(ModelSettingsBuilder, None), None),
        (TokenizerVersion.v15, build_none_only_reasoning_effort_model_settings_builder(), True),
    ],
    ids=["legacy-v3", "modern-v15"],
)
def test_init_exposes_configured_values(
    version: TokenizerVersion, model_settings_builder: ModelSettingsBuilder, expected_builder: bool | None
) -> None:
    tokenizer = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
        special_tokens=special_tokens_with_fillers(22),
        pattern=r".+",
        vocab_size=258 + 22,
        num_special_tokens=22,
        version=version,
        image_config=ImageConfig(image_patch_size=16, max_image_size=1024),
        audio_config=audio_config(),
        model_settings_builder=model_settings_builder,
        _path="synthetic.tekken.json",
    )
    builder = tokenizer.model_settings_builder

    assert tokenizer.n_words == 280
    assert tokenizer.num_special_tokens == 22
    assert tokenizer.special_ids == set(range(22))
    assert tokenizer.version is version
    assert tokenizer.model_settings_builder is builder
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.pad_id == 11
    assert tokenizer.unk_id == 0
    assert tokenizer.vocab()[20:22] == ["<SPECIAL_20>", "<SPECIAL_21>"]


def test_init_rejects_model_settings_for_unsupported_version() -> None:
    builder = build_none_only_reasoning_effort_model_settings_builder()

    with pytest.raises(ValueError, match=r"model_settings_builder is not supported for version=<TokenizerVersion.v3:"):
        build_tekkenizer_v3(
            num_special_tokens=20,
            special_tokens=deprecated_special_tokens(),
            vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
            image_config=ImageConfig(image_patch_size=16, max_image_size=1024),
            audio_config=audio_config(),
            model_settings_builder=builder,
            path="synthetic-v3.tekken.json",
        )


def test_init_rejects_oversized_vocab() -> None:
    vocab = quick_vocab(extra_toks=())

    with pytest.raises(AssertionError, match=r"\(277, 256, 20\)"):
        Tekkenizer(
            vocab=vocab,
            special_tokens=deprecated_special_tokens(),
            pattern=r".+",
            vocab_size=277,
            num_special_tokens=20,
            version=TokenizerVersion.v3,
        )


def test_init_rejects_duplicate_special_tokens() -> None:
    special_tokens = deprecated_special_tokens()
    special_tokens[1] = special_tokens[0]

    with pytest.raises(AssertionError, match="Special tokens must be unique"):
        build_tekkenizer_v3(
            num_special_tokens=20,
            special_tokens=special_tokens,
            vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
            image_config=ImageConfig(image_patch_size=16, max_image_size=1024),
            audio_config=audio_config(),
            model_settings_builder=cast(ModelSettingsBuilder, None),
            path="synthetic-v3.tekken.json",
        )


def test_deprecated_special_tokens_are_frozen() -> None:
    expected = [
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

    assert len(Tekkenizer.DEPRECATED_SPECIAL_TOKENS) == 20
    assert [token["token_str"] for token in Tekkenizer.DEPRECATED_SPECIAL_TOKENS] == expected


def test_file_path_requires_file_loaded_instance(tmp_path: Path) -> None:
    path = tmp_path / "model.tekken.json"
    write_tekkenizer_file(
        path=path,
        version="v3",
        special_tokens=deprecated_special_tokens(),
        num_special_tokens=20,
        vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
        image={},
        multimodal={},
        audio={},
        model_settings={},
    )
    loaded = Tekkenizer.from_file(str(path))

    assert loaded.file_path == path
    without_path = Tekkenizer(
        vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
        special_tokens=deprecated_special_tokens(),
        pattern=r".+",
        vocab_size=258 + 20,
        num_special_tokens=20,
        version=TokenizerVersion.v3,
    )
    with pytest.raises(ValueError, match="The tokenizer was not loaded from a file"):
        without_path.file_path


def test_image_and_audio_properties_are_read_only(tekkenizer: Tekkenizer) -> None:
    image = ImageConfig(image_patch_size=16, max_image_size=1024)
    audio = audio_config()

    with pytest.raises(ValueError, match="Can only set Image config at init"):
        tekkenizer.image = image
    with pytest.raises(ValueError, match="Can only set Audio config at init"):
        tekkenizer.audio = audio


def test_from_file_loads_legacy_defaults_and_modern_configuration(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.tekken.json"
    write_tekkenizer_file(
        path=legacy_path,
        version="v3",
        special_tokens=None,
        num_special_tokens=20,
        vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
        image={},
        multimodal={},
        audio={},
        model_settings={},
    )
    legacy = Tekkenizer.from_file(legacy_path)
    assert legacy.version is TokenizerVersion.v3
    assert legacy.special_ids == set(range(20))

    modern_path = tmp_path / "modern.tekken.json"
    write_tekkenizer_file(
        path=modern_path,
        version="v15",
        special_tokens=special_tokens_with_fillers(22),
        num_special_tokens=22,
        vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
        image={"image_patch_size": 16, "max_image_size": 1024},
        multimodal={},
        audio={
            "sampling_rate": 24_000,
            "frame_rate": 12.5,
            "audio_encoding_config": {"num_mel_bins": 128, "hop_length": 160, "window_size": 400},
        },
        model_settings=build_none_only_reasoning_effort_model_settings_builder().model_dump(mode="json"),
    )
    modern = Tekkenizer.from_file(modern_path)

    assert modern.image == ImageConfig(image_patch_size=16, max_image_size=1024)
    assert modern.audio == audio_config()
    assert modern.model_settings_builder == build_none_only_reasoning_effort_model_settings_builder()


def test_from_file_loaded_tokenizer_encodes_like_the_synthetic_source(tmp_path: Path) -> None:
    path = tmp_path / "loaded.tekken.json"
    write_tekkenizer_file(
        path=path,
        version="v3",
        special_tokens=deprecated_special_tokens(),
        num_special_tokens=20,
        vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
        image={},
        multimodal={},
        audio={},
        model_settings={},
    )
    loaded = Tekkenizer.from_file(path)
    source = tekkenizer_v3()

    inputs = "My very beautiful string"

    assert loaded.encode(s=inputs, bos=False, eos=False) == source.encode(s=inputs, bos=False, eos=False)
    assert loaded.decode(tokens=loaded.encode(s=inputs, bos=True, eos=True)) == source.decode(
        tokens=source.encode(s=inputs, bos=True, eos=True)
    )
    assert loaded.num_special_tokens == source.num_special_tokens
    assert loaded.version == source.version


@pytest.mark.parametrize("version", list(TokenizerVersion), ids=[f"synthetic-{v.value}" for v in TokenizerVersion])
def test_from_file_loads_each_known_version_with_synthetic_file(tmp_path: Path, version: TokenizerVersion) -> None:
    path = tmp_path / f"{version.value}.tekken.json"
    model_settings = (
        build_none_only_reasoning_effort_model_settings_builder().model_dump(mode="json")
        if version.supports_model_settings
        else {}
    )
    write_tekkenizer_file(
        path=path,
        version=version.value,
        special_tokens=deprecated_special_tokens(),
        num_special_tokens=20,
        vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
        image={},
        multimodal={},
        audio={},
        model_settings=model_settings,
    )

    loaded = Tekkenizer.from_file(path)

    assert loaded.version is version


def test_from_file_loads_deprecated_multimodal_configuration_for_legacy_version(tmp_path: Path) -> None:
    path = tmp_path / "legacy-mm.tekken.json"
    write_tekkenizer_file(
        path=path,
        version="v3",
        special_tokens=deprecated_special_tokens(),
        num_special_tokens=20,
        vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
        multimodal={"image_patch_size": 16, "max_image_size": 1024},
        image={},
        audio={},
        model_settings={},
    )

    assert Tekkenizer.from_file(path).image == ImageConfig(image_patch_size=16, max_image_size=1024)


def test_from_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="missing.tekken.json"):
        Tekkenizer.from_file(tmp_path / "missing.tekken.json")


@pytest.mark.parametrize(
    ("version", "special_tokens", "multimodal", "audio", "model_settings", "exception", "message"),
    [
        ("v999", None, None, None, None, ValueError, "Unknown version: v999"),
        (None, None, None, None, None, ValueError, "Unknown version: None"),
        ("v15", None, None, None, None, ValueError, "Special tokens not found"),
        (
            "v15",
            deprecated_special_tokens(),
            {"image_patch_size": 16, "max_image_size": 1024},
            None,
            None,
            ValueError,
            "has to be called 'image'",
        ),
        (
            "v3",
            deprecated_special_tokens(),
            None,
            {"sampling_rate": 24_000},
            None,
            KeyError,
            "audio_encoding_config",
        ),
        (
            "v3",
            deprecated_special_tokens(),
            None,
            None,
            build_none_only_reasoning_effort_model_settings_builder().model_dump(mode="json"),
            ValueError,
            "model_settings_builder is not supported",
        ),
    ],
    ids=[
        "unknown-version",
        "absent-version",
        "missing-specials",
        "invalid-mm-version",
        "malformed-audio",
        "unsupported-model-settings",
    ],
)
def test_from_file_rejects_defined_invalid_inputs(
    tmp_path: Path,
    version: str | None,
    special_tokens: list[SpecialTokenInfo] | None,
    multimodal: dict[str, int] | None,
    audio: dict[str, object] | None,
    model_settings: dict[str, object] | None,
    exception: type[Exception],
    message: str,
) -> None:
    path = tmp_path / "invalid.tekken.json"
    write_tekkenizer_file(
        path=path,
        version=version,
        special_tokens=special_tokens,
        num_special_tokens=20,
        vocab=quick_vocab(extra_toks=[b"hello", b"world"]),
        image={},
        multimodal=multimodal if multimodal is not None else {},
        audio=audio if audio is not None else {},
        model_settings=model_settings if model_settings is not None else {},
    )
    with pytest.raises(exception, match=message):
        Tekkenizer.from_file(path)


@pytest.mark.parametrize(
    ("bos", "eos", "expected"),
    [
        (False, False, [276]),
        (True, False, [1, 276]),
        (False, True, [276, 2]),
        (True, True, [1, 276, 2]),
    ],
    ids=["no-boundaries", "bos-only", "eos-only", "both-boundaries"],
)
def test_encode_applies_special_token_offsets_and_boundaries(
    tekkenizer: Tekkenizer, bos: bool, eos: bool, expected: list[int]
) -> None:
    assert tekkenizer.encode(s="hello", bos=bos, eos=eos) == expected


def test_arbitrary_text_roundtrip_uses_byte_fallback(tekkenizer: Tekkenizer) -> None:
    text = "A curious byte fallback: café 🦊"

    encoded = tekkenizer.encode(s=text, bos=False, eos=False)

    assert any(tekkenizer.is_byte(token) for token in encoded)
    assert tekkenizer.decode(tokens=encoded) == text


@pytest.mark.parametrize(
    ("policy", "policy_arg"),
    [(policy, policy_arg) for policy in SpecialTokenPolicy for policy_arg in (policy, policy.value)],
    ids=[f"{policy.value}-{form}" for policy in SpecialTokenPolicy for form in ("enum", "string")],
)
def test_decode_accepts_policy_values_and_enums(
    tekkenizer: Tekkenizer, policy: SpecialTokenPolicy, policy_arg: SpecialTokenPolicy | str
) -> None:
    tokens = [1, 276, 2]
    expected = {SpecialTokenPolicy.IGNORE: "hello", SpecialTokenPolicy.KEEP: "<s>hello</s>"}

    if policy is SpecialTokenPolicy.RAISE:
        with pytest.raises(ValueError, match="Decoding `tokens` that contain special tokens"):
            tekkenizer.decode(tokens=tokens, special_token_policy=policy_arg)  # type: ignore[arg-type]
    else:
        assert tekkenizer.decode(tokens=tokens, special_token_policy=policy_arg) == expected[policy]  # type: ignore[arg-type]


def test_decode_rejects_invalid_policy(tekkenizer: Tekkenizer) -> None:
    with pytest.raises(ValueError, match=r"Invalid `special_token_policy` 'keeep'"):
        tekkenizer.decode(tokens=[276], special_token_policy="keeep")  # type: ignore[arg-type]


def test_string_and_piece_helpers_keep_special_token_text(tekkenizer: Tekkenizer) -> None:
    assert tekkenizer._to_string(tokens=[1, 276, 2]) == "<s>hello</s>"
    assert tekkenizer.id_to_piece(token_id=1) == "<s>"
    assert tekkenizer.id_to_piece(token_id=276) == "hello"
    assert tekkenizer.vocab()[1] == "<s>"
    assert tekkenizer.vocab()[20] == "\x00"
    assert tekkenizer.vocab()[276] == "hello"


@pytest.mark.parametrize(
    ("token_id", "expected"),
    [(20, True), (275, True), (276, False), (0, False)],
    ids=["first-byte", "last-byte", "merge-token", "special-token"],
)
def test_is_byte_classifies_byte_range_boundaries(tekkenizer: Tekkenizer, token_id: int, expected: bool) -> None:
    assert tekkenizer.is_byte(token_id=token_id) == expected


def test_special_token_lookup_and_deprecated_alias(tekkenizer: Tekkenizer) -> None:
    assert tekkenizer.get_special_token(s=SpecialTokens.bos.value) == 1
    with pytest.raises(ValueError, match="Unknown control token <unknown>"):
        tekkenizer.get_special_token(s="<unknown>")

    with pytest.warns(FutureWarning, match="get_control_token.*deprecated"):
        assert tekkenizer.get_control_token(s=SpecialTokens.eos.value) == 2


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("</s>", True),
        ("hello", False),
        (1, True),
        (276, False),
        (np.int64(0), True),
        (np.int64(1), True),
        (np.int64(276), False),
    ],
    ids=[
        "special-string",
        "ordinary-string",
        "special-int",
        "ordinary-int",
        "special-numpy-id-0",
        "special-numpy-int",
        "ordinary-numpy-int",
    ],
)
def test_is_special_classifies_supported_input_types(tekkenizer: Tekkenizer, token: str | int, expected: bool) -> None:
    assert tekkenizer.is_special(token=token) == expected


def test_is_special_rejects_unsupported_input_type(tekkenizer: Tekkenizer) -> None:
    with pytest.raises(TypeError, match="Expected int or str, got float"):
        tekkenizer.is_special(token=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [(SpecialTokenPolicy.IGNORE, b""), (SpecialTokenPolicy.KEEP, b"<s>")],
    ids=["ignore-special", "keep-special"],
)
def test_id_to_byte_piece_handles_special_token_policies(
    tekkenizer: Tekkenizer, policy: SpecialTokenPolicy, expected: bytes
) -> None:
    assert tekkenizer.id_to_byte_piece(token_id=1, special_token_policy=policy) == expected


def test_id_to_byte_piece_returns_ordinary_bytes_and_raises_for_special_tokens(tekkenizer: Tekkenizer) -> None:
    assert tekkenizer.id_to_byte_piece(token_id=276) == b"hello"
    with pytest.raises(ValueError, match="1 is a special token"):
        tekkenizer.id_to_byte_piece(token_id=1, special_token_policy=SpecialTokenPolicy.RAISE)
    with pytest.raises(ValueError, match=r"Invalid `special_token_policy` 'invalid'"):
        tekkenizer.id_to_byte_piece(token_id=1, special_token_policy="invalid")  # type: ignore[arg-type]


def test_init_truncates_vocab_to_inner_vocab_size() -> None:
    vocab = quick_vocab(extra_toks=[b"hello", b"world"])
    truncated = build_tekkenizer_with_vocab(vocab=vocab, vocab_size=256 + 20)
    full = build_tekkenizer_with_vocab(vocab=vocab, vocab_size=258 + 20)

    # with only the 256 byte tokens kept, "hello"/"world" fall back to identical byte sequences offset by 20
    assert truncated.encode(s="hello", bos=False, eos=False) == [20 + ord(c) for c in "hello"]
    assert full.encode(s="hello", bos=False, eos=False) == [276]
    assert full.encode(s="world", bos=False, eos=False) == [277]


def test_init_rejects_malformed_vocab_entry_missing_key() -> None:
    malformed_vocab = [cast(TokenInfo, {"rank": 0, "token_bytes": "AA=="})]

    with pytest.raises(AssertionError):
        build_tekkenizer_with_vocab(vocab=malformed_vocab, vocab_size=len(malformed_vocab) + 20)


def test_init_rejects_non_contiguous_rank() -> None:
    malformed_vocab = [TokenInfo(rank=1, token_bytes="AA==", token_str=None)]

    with pytest.raises(AssertionError):
        build_tekkenizer_with_vocab(vocab=malformed_vocab, vocab_size=len(malformed_vocab) + 20)


def test_init_rejects_invalid_byte_invariant() -> None:
    malformed_vocab = [TokenInfo(rank=0, token_bytes="%%%", token_str=None)]

    with pytest.raises(AssertionError, match=r"\(0, b''\)"):
        build_tekkenizer_with_vocab(vocab=malformed_vocab, vocab_size=len(malformed_vocab) + 20)


def test_init_rejects_duplicate_mergeable_ranks() -> None:
    vocab = quick_vocab(extra_toks=())
    vocab.extend(
        [
            TokenInfo(rank=256, token_bytes="AA==", token_str=None),
            TokenInfo(rank=257, token_bytes="AA==", token_str=None),
        ]
    )

    with pytest.raises(AssertionError):
        build_tekkenizer_with_vocab(vocab=vocab, vocab_size=258 + 20)


def test_init_rejects_vocab_smaller_than_inner_vocab_size() -> None:
    # 256 byte tokens cannot fill an inner vocab of 257 mergeable ranks
    with pytest.raises(AssertionError, match=r"\(277, 256, 20\)"):
        build_tekkenizer_with_vocab(vocab=quick_vocab(extra_toks=()), vocab_size=257 + 20)


def test_is_tekkenizer_distinguishes_tekken_from_other_tokenizers(tekkenizer: Tekkenizer) -> None:
    assert is_tekkenizer(tekkenizer)
    assert not is_tekkenizer(cast(Tokenizer, object()))

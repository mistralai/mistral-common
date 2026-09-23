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
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.tekken import (
    SpecialTokenInfo,
    Tekkenizer,
    TokenInfo,
    _reload_mergeable_ranks,
    is_tekken,
    is_tekkenizer,
)


def quick_vocab(extra_tokens: Sequence[bytes] = ()) -> list[TokenInfo]:
    vocab = [TokenInfo(rank=i, token_bytes=base64.b64encode(bytes([i])).decode(), token_str=chr(i)) for i in range(256)]
    vocab.extend(
        TokenInfo(rank=256 + i, token_bytes=base64.b64encode(token).decode(), token_str=token.decode())
        for i, token in enumerate(extra_tokens)
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


def build_tokenizer(
    *,
    version: TokenizerVersion = TokenizerVersion.v3,
    num_special_tokens: int = 20,
    special_tokens: list[SpecialTokenInfo] | None = None,
    image_config: ImageConfig | None = None,
    audio_config: AudioConfig | None = None,
    model_settings_builder: ModelSettingsBuilder | None = None,
    path: str | Path | None = None,
) -> Tekkenizer:
    vocab = quick_vocab(extra_tokens=[b"hello", b"world"])
    if special_tokens is None:
        special_tokens = deprecated_special_tokens()
    return Tekkenizer(
        vocab=vocab,
        special_tokens=special_tokens,
        pattern=r".+",
        vocab_size=len(vocab) + num_special_tokens,
        num_special_tokens=num_special_tokens,
        version=version,
        image_config=image_config,
        audio_config=audio_config,
        model_settings_builder=model_settings_builder,
        _path=path,
    )


def model_settings_builder() -> ModelSettingsBuilder:
    return ModelSettingsBuilder(
        reasoning_effort=EnumBuilder[ReasoningEffort](
            values=[ReasoningEffort.none],
            accepts_none=True,
            default=ReasoningEffort.none,
        )
    )


def audio_config() -> AudioConfig:
    return AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(num_mel_bins=128, hop_length=160, window_size=400),
    )


def write_tokenizer_file(
    path: Path,
    *,
    version: str | None = "v3",
    special_tokens: list[SpecialTokenInfo] | None = None,
    num_special_tokens: int = 20,
    image: dict[str, int] | None = None,
    multimodal: dict[str, int] | None = None,
    audio: dict[str, object] | None = None,
    model_settings: dict[str, object] | None = None,
    vocab: list[TokenInfo] | None = None,
    default_vocab_size: int | None = None,
) -> None:
    if vocab is None:
        vocab = quick_vocab(extra_tokens=[b"hello", b"world"])
    config: dict[str, object] = {
        "pattern": r".+",
        "num_vocab_tokens": len(vocab),
        "default_vocab_size": default_vocab_size or len(vocab) + num_special_tokens,
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
    if image is not None:
        model["image"] = image
    if multimodal is not None:
        model["multimodal"] = multimodal
    if audio is not None:
        model["audio"] = audio
    if model_settings is not None:
        model["model_settings_builder"] = model_settings
    path.write_text(json.dumps(model), encoding="utf-8")


@pytest.fixture(scope="module")
def tekkenizer() -> Tekkenizer:
    return build_tokenizer()


def test_is_tekken_requires_existing_json_file_with_marker(tmp_path: Path) -> None:
    tekken_path = tmp_path / "model.tekken.json"
    tekken_path.write_text("{}", encoding="utf-8")

    assert is_tekken(tekken_path)
    assert is_tekken(str(tekken_path))

    for path in (
        tmp_path / "model.json",
        tmp_path / "model.tekken.bin",
        tmp_path / "missing.tekken.json",
        tmp_path / "directory.tekken.json",
    ):
        if path.name.startswith("directory"):
            path.mkdir()
        assert not is_tekken(path)


@pytest.mark.parametrize(
    ("version", "has_builder"),
    [(TokenizerVersion.v3, False), (TokenizerVersion.v15, True)],
    ids=["legacy-v3", "modern-v15"],
)
def test_init_exposes_configured_values(version: TokenizerVersion, has_builder: bool) -> None:
    builder = model_settings_builder() if has_builder else None
    tokenizer = build_tokenizer(version=version, num_special_tokens=22, model_settings_builder=builder)

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
    builder = model_settings_builder()

    with pytest.raises(ValueError, match=r"model_settings_builder is not supported for version=<TokenizerVersion.v3:"):
        build_tokenizer(model_settings_builder=builder)


def test_init_rejects_oversized_vocab() -> None:
    vocab = quick_vocab()

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
        build_tokenizer(special_tokens=special_tokens)


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
    write_tokenizer_file(path)
    loaded = Tekkenizer.from_file(str(path))

    assert loaded.file_path == path
    with pytest.raises(ValueError, match="The tokenizer was not loaded from a file"):
        build_tokenizer().file_path


def test_image_and_audio_properties_are_read_only(tekkenizer: Tekkenizer) -> None:
    image = ImageConfig(image_patch_size=16, max_image_size=1024)
    audio = audio_config()

    with pytest.raises(ValueError, match="Can only set Image config at init"):
        tekkenizer.image = image
    with pytest.raises(ValueError, match="Can only set Audio config at init"):
        tekkenizer.audio = audio


def test_from_file_loads_legacy_defaults_and_modern_configuration(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.tekken.json"
    write_tokenizer_file(legacy_path, special_tokens=None)
    legacy = Tekkenizer.from_file(legacy_path)
    assert legacy.version is TokenizerVersion.v3
    assert legacy.special_ids == set(range(20))

    modern_path = tmp_path / "modern.tekken.json"
    write_tokenizer_file(
        modern_path,
        version="v15",
        special_tokens=special_tokens_with_fillers(22),
        num_special_tokens=22,
        image={"image_patch_size": 16, "max_image_size": 1024},
        audio={
            "sampling_rate": 24_000,
            "frame_rate": 12.5,
            "audio_encoding_config": {"num_mel_bins": 128, "hop_length": 160, "window_size": 400},
        },
        model_settings=model_settings_builder().model_dump(mode="json"),
    )
    modern = Tekkenizer.from_file(modern_path)

    assert modern.image == ImageConfig(image_patch_size=16, max_image_size=1024)
    assert modern.audio == audio_config()
    assert modern.model_settings_builder == model_settings_builder()


@pytest.mark.parametrize(
    "version",
    [
        TokenizerVersion.v3,
        TokenizerVersion.v7,
        TokenizerVersion.v11,
        TokenizerVersion.v13,
        TokenizerVersion.v15,
    ],
    ids=["tekken-v3", "tekken-v7", "tekken-v11", "tekken-v13", "tekken-v15"],
)
def test_from_file_loads_every_released_tekken_version(tmp_path: Path, version: TokenizerVersion) -> None:
    path = tmp_path / f"{version.value}.tekken.json"
    write_tokenizer_file(
        path,
        version=version.value,
        special_tokens=deprecated_special_tokens(),
        num_special_tokens=20,
        model_settings=model_settings_builder().model_dump(mode="json") if version is TokenizerVersion.v15 else None,
    )

    loaded = Tekkenizer.from_file(path)

    assert loaded.version is version


def test_from_file_loads_deprecated_multimodal_configuration_for_legacy_version(tmp_path: Path) -> None:
    path = tmp_path / "legacy-mm.tekken.json"
    write_tokenizer_file(path, multimodal={"image_patch_size": 16, "max_image_size": 1024})

    assert Tekkenizer.from_file(path).image == ImageConfig(image_patch_size=16, max_image_size=1024)


@pytest.mark.parametrize(
    ("case", "version", "special_tokens", "multimodal", "audio", "model_settings", "exception", "message"),
    [
        ("missing", "v3", None, None, None, None, AssertionError, "missing.tekken.json"),
        ("unknown-version", "v999", None, None, None, None, ValueError, "Unknown version: v999"),
        ("missing-specials", "v15", None, None, None, None, ValueError, "Special tokens not found"),
        (
            "invalid-mm-version",
            "v15",
            deprecated_special_tokens(),
            {"image_patch_size": 16, "max_image_size": 1024},
            None,
            None,
            ValueError,
            "has to be called 'image'",
        ),
        (
            "malformed-audio",
            "v3",
            deprecated_special_tokens(),
            None,
            {"sampling_rate": 24_000},
            None,
            KeyError,
            "audio_encoding_config",
        ),
        (
            "unsupported-model-settings",
            "v3",
            deprecated_special_tokens(),
            None,
            None,
            model_settings_builder().model_dump(mode="json"),
            ValueError,
            "model_settings_builder is not supported",
        ),
    ],
    ids=[
        "missing-file",
        "unknown-version",
        "missing-specials",
        "invalid-mm-version",
        "malformed-audio",
        "unsupported-model-settings",
    ],
)
def test_from_file_rejects_defined_invalid_inputs(
    tmp_path: Path,
    case: str,
    version: str,
    special_tokens: list[SpecialTokenInfo] | None,
    multimodal: dict[str, int] | None,
    audio: dict[str, object] | None,
    model_settings: dict[str, object] | None,
    exception: type[Exception],
    message: str,
) -> None:
    path = tmp_path / f"{case}.tekken.json"
    if case != "missing":
        write_tokenizer_file(
            path,
            version=version,
            special_tokens=special_tokens,
            multimodal=multimodal,
            audio=audio,
            model_settings=model_settings,
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
    assert tekkenizer.encode("hello", bos=bos, eos=eos) == expected


def test_arbitrary_text_roundtrip_uses_byte_fallback(tekkenizer: Tekkenizer) -> None:
    text = "A curious byte fallback: café 🦊"

    encoded = tekkenizer.encode(text, bos=False, eos=False)

    assert any(tekkenizer.is_byte(token) for token in encoded)
    assert tekkenizer.decode(encoded) == text


@pytest.mark.parametrize("policy", list(SpecialTokenPolicy), ids=[policy.value for policy in SpecialTokenPolicy])
def test_decode_accepts_policy_values_and_enums(tekkenizer: Tekkenizer, policy: SpecialTokenPolicy) -> None:
    tokens = [1, 276, 2]
    expected = {SpecialTokenPolicy.IGNORE: "hello", SpecialTokenPolicy.KEEP: "<s>hello</s>"}

    if policy is SpecialTokenPolicy.RAISE:
        with pytest.raises(ValueError, match="Decoding `tokens` that contain special tokens"):
            tekkenizer.decode(tokens, special_token_policy=policy)
    else:
        assert tekkenizer.decode(tokens, special_token_policy=policy.value) == expected[policy]


def test_decode_rejects_invalid_policy(tekkenizer: Tekkenizer) -> None:
    with pytest.raises(ValueError, match=r"Invalid `special_token_policy` 'keeep'"):
        tekkenizer.decode([276], special_token_policy="keeep")  # type: ignore[arg-type]


def test_decode_all_handles_contiguous_special_and_ordinary_groups(tekkenizer: Tekkenizer) -> None:
    tokens = [1, 276, 2, 277]

    assert tekkenizer._decode_all(tokens, SpecialTokenPolicy.IGNORE) == ["hello", "world"]
    assert tekkenizer._decode_all(tokens, SpecialTokenPolicy.KEEP) == ["<s>", "hello", "</s>", "world"]


def test_string_and_piece_helpers_keep_special_token_text(tekkenizer: Tekkenizer) -> None:
    assert tekkenizer._to_string([1, 276, 2]) == "<s>hello</s>"
    assert tekkenizer.id_to_piece(1) == "<s>"
    assert tekkenizer.id_to_piece(276) == "hello"
    assert tekkenizer.vocab()[1] == "<s>"
    assert tekkenizer.vocab()[20] == "\x00"
    assert tekkenizer.vocab()[276] == "hello"


@pytest.mark.parametrize(
    ("token_id", "expected"),
    [(20, True), (275, True), (276, False), (0, False)],
    ids=["first-byte", "last-byte", "merge-token", "special-token"],
)
def test_is_byte_classifies_byte_range_boundaries(tekkenizer: Tekkenizer, token_id: int, expected: bool) -> None:
    assert tekkenizer.is_byte(token_id) == expected


def test_special_token_lookup_and_deprecated_alias(tekkenizer: Tekkenizer) -> None:
    assert tekkenizer.get_special_token(SpecialTokens.bos.value) == 1
    with pytest.raises(ValueError, match="Unknown control token <unknown>"):
        tekkenizer.get_special_token("<unknown>")

    with pytest.warns(FutureWarning, match="get_control_token.*deprecated"):
        assert tekkenizer.get_control_token(SpecialTokens.eos.value) == 2


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("</s>", True),
        ("hello", False),
        (1, True),
        (276, False),
        (np.int64(1), True),
        (np.int64(276), False),
    ],
    ids=["special-string", "ordinary-string", "special-int", "ordinary-int", "special-numpy-int", "ordinary-numpy-int"],
)
def test_is_special_classifies_supported_input_types(tekkenizer: Tekkenizer, token: str | int, expected: bool) -> None:
    assert tekkenizer.is_special(token) == expected


def test_is_special_rejects_unsupported_input_type(tekkenizer: Tekkenizer) -> None:
    with pytest.raises(TypeError, match="Expected int or str, got float"):
        tekkenizer.is_special(1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [(SpecialTokenPolicy.IGNORE, b""), (SpecialTokenPolicy.KEEP, b"<s>")],
    ids=["ignore-special", "keep-special"],
)
def test_id_to_byte_piece_handles_special_token_policies(
    tekkenizer: Tekkenizer, policy: SpecialTokenPolicy, expected: bytes
) -> None:
    assert tekkenizer.id_to_byte_piece(1, special_token_policy=policy) == expected


def test_id_to_byte_piece_returns_ordinary_bytes_and_raises_for_special_tokens(tekkenizer: Tekkenizer) -> None:
    assert tekkenizer.id_to_byte_piece(276) == b"hello"
    with pytest.raises(ValueError, match="1 is a special token"):
        tekkenizer.id_to_byte_piece(1, special_token_policy=SpecialTokenPolicy.RAISE)
    with pytest.raises(ValueError, match="Unknown special token policy invalid"):
        tekkenizer.id_to_byte_piece(1, special_token_policy="invalid")  # type: ignore[arg-type]


def test_reload_mergeable_ranks_decodes_and_truncates_vocab() -> None:
    vocab = quick_vocab(extra_tokens=[b"hello"])

    assert _reload_mergeable_ranks(vocab=vocab, max_vocab=256) == {bytes([i]): i for i in range(256)}
    assert _reload_mergeable_ranks(vocab=vocab, max_vocab=257)[b"hello"] == 256


def test_reload_mergeable_ranks_rejects_missing_token_info_key() -> None:
    malformed_vocab = [cast(TokenInfo, {"rank": 0, "token_bytes": "AA=="})]

    with pytest.raises(AssertionError):
        _reload_mergeable_ranks(malformed_vocab)


def test_reload_mergeable_ranks_rejects_non_contiguous_rank() -> None:
    malformed_vocab = [TokenInfo(rank=1, token_bytes="AA==", token_str=None)]

    with pytest.raises(AssertionError):
        _reload_mergeable_ranks(malformed_vocab)


def test_reload_mergeable_ranks_rejects_invalid_byte_invariant() -> None:
    malformed_vocab = [TokenInfo(rank=0, token_bytes="%%%", token_str=None)]

    with pytest.raises(AssertionError, match=r"\(0, b''\)"):
        _reload_mergeable_ranks(malformed_vocab)


def test_reload_mergeable_ranks_rejects_duplicate_merges() -> None:
    vocab = quick_vocab()
    vocab.extend(
        [
            TokenInfo(rank=256, token_bytes="AA==", token_str=None),
            TokenInfo(rank=257, token_bytes="AA==", token_str=None),
        ]
    )

    with pytest.raises(AssertionError):
        _reload_mergeable_ranks(vocab=vocab, max_vocab=258)


def test_reload_mergeable_ranks_requires_enough_vocab() -> None:
    with pytest.raises(AssertionError, match=r"\(256, 257\)"):
        _reload_mergeable_ranks(quick_vocab(), max_vocab=257)


def test_is_tekkenizer_distinguishes_tekken_from_other_tokenizers(tekkenizer: Tekkenizer) -> None:
    assert is_tekkenizer(tekkenizer)
    assert not is_tekkenizer(cast(Tokenizer, object()))

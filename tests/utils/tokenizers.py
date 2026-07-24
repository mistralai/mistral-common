import base64
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ReasoningEffort
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.audio import AudioConfig, AudioEncoder, AudioSpectrogramConfig, SpecialAudioIDs
from mistral_common.tokens.tokenizers.base import InstructTokenizer, SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.image import ImageConfig, ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerV1,
    InstructTokenizerV2,
    InstructTokenizerV3,
    InstructTokenizerV7,
    InstructTokenizerV11,
    InstructTokenizerV13,
    InstructTokenizerV15,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer, TokenInfo
from tests.utils.versions import TestConfig

# Patch size used by image tokenization tests so expected token counts stay small
# and hand-checkable.
TEST_IMAGE_PATCH_SIZE = 2

# `InstructTokenizerV2._encode_infilling` hardcodes this exact sentinel character and
# slices off its first two encoded tokens (`[2:]`) to strip a SentencePiece leading space
# without touching the FIM suffix -- see that method's docstring. It is purely an SPM
# workaround: SentencePiece always prepends `▁` to the first piece of anything it encodes,
# and the sentinel absorbs that prefix so it can be sliced away. Tekken has no such prefix
# and no native concept of this sentinel; `InstructTokenizerV2` (and every tekken subclass
# that inherits `_encode_infilling` from it) merely runs the same SPM-shaped code path
# regardless of backend. There is no public production constant for it, so it is
# duplicated here, once, to build a synthetic tekken vocab that satisfies the same
# incidental token-count invariant real tekken vocabularies happen to satisfy.
SPM_INFILLING_SENTINEL = "\u263a"

# First two UTF-8 bytes of `SPM_INFILLING_SENTINEL`. Adding this pair as a single vocab
# entry makes a synthetic `Tekkenizer` merge them into one token, so the sentinel encodes
# to exactly two tokens there -- replicating, in a 256-byte synthetic vocab, the two-token
# encoding real tekken vocabularies produce by accident. Without this, the sentinel would
# encode into three tokens and the hardcoded `[2:]` slice in `_encode_infilling` would
# corrupt the FIM suffix by dropping one of its real leading tokens.
SPM_INFILLING_SENTINEL_MERGE: bytes = SPM_INFILLING_SENTINEL.encode("utf-8")[:2]


def _load_tekken_pattern() -> str:
    r"""Load the production tekken regex pattern from a bundled tekken tokenizer file.

    Every real tekken tokenizer file (bundled or shipped by any recent Mistral model)
    carries this exact pattern, so reading it here instead of retyping the regex as a
    string literal guarantees the constant can never drift from what production
    tokenizers actually use.

    Returns:
        The regex pattern tiktoken uses to split text into chunks before BPE merging.
    """
    data_path = MistralTokenizer._data_path() / "tekken_240911.json"
    with open(data_path, encoding="utf-8") as f:
        config = json.load(f)["config"]
    pattern: str = config["pattern"]
    return pattern


# The real tekken chunking pattern. Synthetic tokenizers use it so they chunk text the same
# way a production tokenizer does; a pattern without DOTALL silently discards newlines.
TEKKEN_PATTERN: str = _load_tekken_pattern()


def quick_vocab(extra_toks: Sequence[bytes] = ()) -> list[TokenInfo]:
    r"""Build a minimal vocabulary for unit tests.

    Creates a 256-byte base vocabulary where each entry corresponds to the
    byte value at that index, plus one entry per extra token appended after.

    Args:
        extra_toks: Additional raw byte strings to append after the 256-byte base.

    Returns:
        List of `TokenInfo` dicts forming the complete vocabulary.
    """
    vocab = [TokenInfo(rank=i, token_bytes=base64.b64encode(bytes([i])).decode(), token_str=chr(i)) for i in range(256)]
    for i, tok in enumerate(extra_toks):
        vocab.append(
            TokenInfo(
                rank=256 + i,
                token_bytes=base64.b64encode(tok).decode(),
                # SPM_INFILLING_SENTINEL_MERGE is half of a UTF-8 codepoint, and Tekkenizer only
                # reads token_bytes to build the rank map, so token_str must tolerate the split.
                token_str=tok.decode(errors="replace"),
            )
        )
    return vocab


def deprecated_special_tokens() -> list[SpecialTokenInfo]:
    r"""Return a fresh copy of the deprecated special tokens list.

    Returns:
        A new list containing all entries from `Tekkenizer.DEPRECATED_SPECIAL_TOKENS`.
    """
    return list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)


def get_special_tokens(
    tokenizer_version: TokenizerVersion, add_audio: bool = False, add_think: bool = False
) -> list[SpecialTokenInfo]:
    r"""Build the special token list for a given tokenizer version.

    Args:
        tokenizer_version: The version of the tokenizer.
        add_audio: Whether to include audio-specific special tokens. Requires v7 or above.
        add_think: Whether to include thinking special tokens. Cannot be combined with `add_audio`.

    Returns:
        The list of special token infos appropriate for the given version and flags.

    Raises:
        ValueError: If `add_audio` is True and `tokenizer_version` is below v7.
    """
    special_tokens = deprecated_special_tokens()
    if tokenizer_version < TokenizerVersion.v7 and add_audio:
        raise ValueError("Audio tokens are only supported in v7 and above")

    if tokenizer_version <= TokenizerVersion.v7 and not add_audio:
        return special_tokens

    # fill special tokens until 24
    special_tokens += [
        SpecialTokenInfo(rank=i, token_str=f"<SPECIAL_{i}>", is_control=True) for i in range(len(special_tokens), 24)
    ]

    if add_audio:
        # add audio tokes
        special_tokens += [
            SpecialTokenInfo(rank=24, token_str=SpecialTokens.audio, is_control=True),
            SpecialTokenInfo(rank=25, token_str=SpecialTokens.begin_audio, is_control=True),
        ]

    # fill special tokens until 32
    special_tokens += [
        SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True) for i in range(len(special_tokens), 32)
    ]

    if tokenizer_version > TokenizerVersion.v7:
        special_tokens += [
            SpecialTokenInfo(rank=32, token_str=SpecialTokens.args, is_control=True),
            SpecialTokenInfo(rank=33, token_str=SpecialTokens.call_id, is_control=True),
        ]

    # fill special tokens until 34
    special_tokens += [
        SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True) for i in range(len(special_tokens), 34)
    ]

    if add_audio:
        assert not add_think, f"Audio and think tokens are mutually exclusive, got {add_audio} and {add_think}"
        special_tokens += [
            SpecialTokenInfo(rank=34, token_str=SpecialTokens.transcribe, is_control=True),
            SpecialTokenInfo(rank=35, token_str=SpecialTokens.text_to_audio, is_control=True),
            SpecialTokenInfo(rank=36, token_str=SpecialTokens.audio_to_text, is_control=True),
        ]

    if tokenizer_version < TokenizerVersion.v13:
        return special_tokens

    if not add_audio and add_think:
        special_tokens += [SpecialTokenInfo(rank=34, token_str=f"<SPCECIAL_{34}>", is_control=True)]

    if add_think:
        special_tokens += [
            SpecialTokenInfo(rank=35, token_str="[THINK]", is_control=True),
            SpecialTokenInfo(rank=36, token_str="[/THINK]", is_control=True),
        ]

    if tokenizer_version >= TokenizerVersion.v15:
        # fill until rank 37
        special_tokens += [
            SpecialTokenInfo(rank=i, token_str=f"<SPCECIAL_{i}>", is_control=True)
            for i in range(len(special_tokens), 37)
        ]
        special_tokens += [
            SpecialTokenInfo(rank=37, token_str=SpecialTokens.begin_model_settings, is_control=True),
            SpecialTokenInfo(rank=38, token_str=SpecialTokens.end_model_settings, is_control=True),
        ]

    return special_tokens


def build_tekkenizer(
    version: TokenizerVersion,
    *,
    add_audio: bool = False,
    add_think: bool = False,
    extra_toks: Sequence[bytes] = (b"a", b"b", b"c", b"f", b"de"),
    pattern: str = TEKKEN_PATTERN,
    num_special_tokens: int = 100,
    vocab_size: int | None = None,
    model_settings_builder: ModelSettingsBuilder | None = None,
) -> Tekkenizer:
    r"""Build an in-memory `Tekkenizer` for use in unit tests.

    Constructs a `Tekkenizer` from a minimal in-memory vocabulary, avoiding
    the need for a tokenizer file on disk. The vocabulary is built from
    `quick_vocab` and the special tokens from `get_special_tokens`.

    Args:
        version: The tokenizer version to build.
        add_audio: Whether to include audio special tokens (v7 and above only).
        add_think: Whether to include thinking special tokens.
        extra_toks: Additional raw byte strings appended to the 256-byte base vocabulary.
        pattern: The regex pattern used by tiktoken for tokenization. Defaults to the
            production tekken pattern, so synthetic tokenizers chunk text the same way a
            shipped one does. A pattern without `DOTALL` silently discards newlines.
        num_special_tokens: The number of special token slots to reserve.
        vocab_size: Total vocabulary size. Defaults to `256 + num_special_tokens` when `None`.
        model_settings_builder: Optional model settings builder passed to the constructor.

    Returns:
        A `Tekkenizer` instance built from the given parameters.

    Raises:
        ValueError: If `add_audio` is True and `version` is below v7, or if
            `model_settings_builder` is set for a version that does not support it.
    """
    if vocab_size is None:
        vocab_size = 256 + num_special_tokens
    vocab = quick_vocab(extra_toks=list(extra_toks))
    special_tokens = get_special_tokens(tokenizer_version=version, add_audio=add_audio, add_think=add_think)
    return Tekkenizer(
        vocab=vocab,
        special_tokens=special_tokens,
        pattern=pattern,
        vocab_size=vocab_size,
        num_special_tokens=num_special_tokens,
        version=version,
        model_settings_builder=model_settings_builder,
    )


def write_tekkenizer_model(
    tmp_path: Path,
    vocab: list[TokenInfo] | None = None,
    special_tokens: list[SpecialTokenInfo] | None = None,
    version: str | None = "v3",
    *,
    image: dict | None = None,
    audio: dict | None = None,
    multimodal: dict | None = None,
    model_settings_builder: dict | None = None,
) -> None:
    r"""Write a minimal tekken tokenizer JSON file for use in file-loading tests.

    Supports optional `image`, `audio`, `multimodal`, and `model_settings_builder`
    keys for tests that exercise those `from_file` code paths.

    Args:
        tmp_path: Destination file path to write.
        vocab: Token list to embed. Defaults to `quick_vocab()` (256-byte vocabulary).
        special_tokens: Special token list. `None` omits the key from the JSON,
            which `Tekkenizer.from_file` interprets as the deprecated special token set
            for tokenizers up to v7.
        version: Version string written to the config block. `None` omits the key,
            which causes `Tekkenizer.from_file` to raise `ValueError`.
        image: Optional dict written as the `image` key in the JSON payload.
        audio: Optional dict written as the `audio` key in the JSON payload.
        multimodal: Optional dict written as the `multimodal` key in the JSON payload.
        model_settings_builder: Optional dict written as the `model_settings_builder`
            key in the JSON payload.
    """
    if vocab is None:
        vocab = quick_vocab()

    num_special_tokens = 100
    config: dict[str, Any] = {
        "pattern": ".",
        "default_num_special_tokens": num_special_tokens,
        "default_vocab_size": 256 + 3 + num_special_tokens,
    }

    if version is not None:
        config["version"] = version

    model: dict[str, Any] = {
        "vocab": vocab,
        "config": config,
        "special_tokens": special_tokens,
        "version": 1,
        "type": "Tekken",
    }

    if image is not None:
        model["image"] = image
    if audio is not None:
        model["audio"] = audio
    if multimodal is not None:
        model["multimodal"] = multimodal
    if model_settings_builder is not None:
        model["model_settings_builder"] = model_settings_builder

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)


def build_tekkenizer_from_file(
    output_dir: Path,
    *,
    version: str | None = "v3",
    vocab: list[TokenInfo] | None = None,
    special_tokens: list[SpecialTokenInfo] | None = None,
    image: dict | None = None,
    audio: dict | None = None,
    multimodal: dict | None = None,
    model_settings_builder: dict | None = None,
    filename: str = "tekken.json",
) -> Tekkenizer:
    r"""Write a tekken tokenizer JSON file and load it through `Tekkenizer.from_file`.

    Wraps `write_tekkenizer_model` and `Tekkenizer.from_file` so file-loading tests do
    not repeat the write-then-load boilerplate. The default vocabulary carries the three
    extra merged tokens that match the `default_vocab_size` written by `write_tekkenizer_model`.

    Args:
        output_dir: Directory to write the JSON file into.
        version: Version string written to the config block. `None` omits the key.
        vocab: Token list to embed. Defaults to a 256-byte base plus ``beau``, ``My``, ``unused``.
        special_tokens: Special token list. `None` omits the key from the JSON.
        image: Optional dict written as the `image` key.
        audio: Optional dict written as the `audio` key.
        multimodal: Optional dict written as the `multimodal` key.
        model_settings_builder: Optional dict written as the `model_settings_builder` key.
        filename: Name of the JSON file written under `output_dir`.

    Returns:
        The `Tekkenizer` loaded from the written file.
    """
    if vocab is None:
        vocab = quick_vocab(extra_toks=[b"beau", b"My", b"unused"])
    path = output_dir / filename
    write_tekkenizer_model(
        tmp_path=path,
        vocab=vocab,
        special_tokens=special_tokens,
        version=version,
        image=image,
        audio=audio,
        multimodal=multimodal,
        model_settings_builder=model_settings_builder,
    )
    return Tekkenizer.from_file(path)


def build_tekkenizer_from_config(config: TestConfig, output_dir: Path | None = None) -> Tekkenizer:
    r"""Build a `Tekkenizer` from a `TestConfig` (version + modality flags).

    Base and think configs are built in memory via `build_tekkenizer`. Image and audio
    configs carry a real `ImageConfig` / `AudioConfig`, which only `from_file` can attach,
    so they are written and loaded via `build_tekkenizer_from_file` and require `output_dir`.

    Args:
        config: The version + modality configuration to build.
        output_dir: Directory used to write the tokenizer file. Required when `config.image`
            or `config.audio`.

    Returns:
        A `Tekkenizer` matching the configuration.

    Raises:
        ValueError: If `config.image` or `config.audio` is set but `output_dir` is `None`.
    """
    if config.image or config.audio:
        if output_dir is None:
            raise ValueError("output_dir is required to build an image or audio config from file")
        return build_tekkenizer_from_file(
            output_dir,
            version=config.version.value,
            special_tokens=get_special_tokens(
                tokenizer_version=config.version, add_audio=config.audio, add_think=config.think
            ),
            image={"image_patch_size": 2, "max_image_size": 10, "spatial_merge_size": 1} if config.image else None,
            audio={
                "sampling_rate": 16000,
                "frame_rate": 12.5,
                "audio_encoding_config": {"num_mel_bins": 80, "hop_length": 160, "window_size": 400},
            }
            if config.audio
            else None,
        )
    return build_tekkenizer(version=config.version, add_audio=False, add_think=config.think)


def load_sentencepiece(version: str) -> SentencePieceTokenizer:
    r"""Load a shipped SentencePiece tokenizer by version string.

    Args:
        version: The tokenizer version string, e.g. `"v7"`.

    Returns:
        A `SentencePieceTokenizer` loaded from the packaged data directory.

    Raises:
        FileNotFoundError: If no model file matching the given version exists in the data directory.
    """
    data_path = MistralTokenizer._data_path()
    matches = sorted(data_path.glob(f"*.model.{version}"))
    if not matches:
        raise FileNotFoundError(f"No SentencePiece model found for version {version!r} in {data_path}")
    return SentencePieceTokenizer(model_path=matches[0])


def build_audio_encoder(
    tokenizer: Tekkenizer,
    *,
    sampling_rate: int = 24_000,
    frame_rate: float = 12.5,
    num_mel_bins: int = 128,
    window_size: int = 400,
    hop_length: int = 160,
) -> AudioEncoder:
    r"""Build an `AudioEncoder` wired to a tokenizer's audio special tokens.

    Defaults match the audio configuration used across the instruct tokenizer tests,
    replacing the `AudioConfig` / `SpecialAudioIDs` block that was copied per version.

    Args:
        tokenizer: Tokenizer providing the audio special token ids.
        sampling_rate: Audio sampling rate in Hz.
        frame_rate: Audio frame rate in Hz.
        num_mel_bins: Number of mel bins in the spectrogram.
        window_size: Spectrogram window size.
        hop_length: Spectrogram hop length.

    Returns:
        An `AudioEncoder` for the tokenizer.
    """
    audio_config = AudioConfig(
        sampling_rate=sampling_rate,
        frame_rate=frame_rate,
        encoding_config=AudioSpectrogramConfig(
            num_mel_bins=num_mel_bins,
            window_size=window_size,
            hop_length=hop_length,
        ),
    )
    special_audio_ids = SpecialAudioIDs(
        audio=tokenizer.get_special_token(SpecialTokens.audio.value),
        begin_audio=tokenizer.get_special_token(SpecialTokens.begin_audio.value),
        streaming_pad=None,
        text_to_audio=None,
        audio_to_text=None,
    )
    return AudioEncoder(audio_config, special_audio_ids)


def special_image_ids(tokenizer: Tekkenizer) -> SpecialImageIDs:
    r"""Return the `SpecialImageIDs` of a tokenizer.

    Args:
        tokenizer: Tokenizer providing the image special token ids.

    Returns:
        The image special token ids of the tokenizer.
    """
    return SpecialImageIDs(
        img=tokenizer.get_special_token(SpecialTokens.img.value),
        img_break=tokenizer.get_special_token(SpecialTokens.img_break.value),
        img_end=tokenizer.get_special_token(SpecialTokens.img_end.value),
    )


def build_image_encoder(
    tokenizer: Tekkenizer,
    *,
    image_patch_size: int = 16,
    max_image_size: int = 1024,
) -> ImageEncoder:
    r"""Build an `ImageEncoder` wired to a tokenizer's image special tokens.

    Args:
        tokenizer: Tokenizer providing the image special token ids.
        image_patch_size: Patch size of the image config.
        max_image_size: Maximum image size of the image config.

    Returns:
        An `ImageEncoder` for the tokenizer.
    """
    image_config = ImageConfig(image_patch_size=image_patch_size, max_image_size=max_image_size)
    return ImageEncoder(image_config, special_image_ids(tokenizer))


def set_test_image_patch_size(
    instruct_tokenizer: InstructTokenizer, patch_size: int = TEST_IMAGE_PATCH_SIZE
) -> ImageEncoder:
    r"""Shrink an instruct tokenizer's image patch size so token counts stay checkable.

    Args:
        instruct_tokenizer: The instruct tokenizer whose image encoder to adjust.
        patch_size: The patch size to set.

    Returns:
        The adjusted `ImageEncoder`.

    Raises:
        ValueError: If the instruct tokenizer has no image encoder.
    """
    image_encoder = instruct_tokenizer.image_encoder
    if not isinstance(image_encoder, ImageEncoder):
        raise ValueError("Instruct tokenizer has no image encoder")
    image_encoder.image_config.image_patch_size = patch_size
    return image_encoder


def image_token_ids(width: int, height: int, special_ids: SpecialImageIDs) -> list[int]:
    r"""Build the expected token sequence for an image of a given patch grid.

    Args:
        width: Number of patches per row.
        height: Number of patch rows.
        special_ids: The image special token ids.

    Returns:
        The expected image token sequence, ending with the image-end token.
    """
    tokens = ([special_ids.img] * width + [special_ids.img_break]) * height
    tokens[-1] = special_ids.img_end
    return tokens


def image_token_spans(tokens: list[int], special_ids: SpecialImageIDs) -> list[list[int]]:
    r"""Extract each image token span from a token sequence.

    A span starts at an image token and ends at the following image-end token.

    Args:
        tokens: The full token sequence to scan.
        special_ids: The image special token ids.

    Returns:
        One sub-sequence per image, in order of appearance.
    """
    spans: list[list[int]] = []
    start_idx: int | None = None
    for idx, token in enumerate(tokens):
        if start_idx is None:
            if token == special_ids.img:
                start_idx = idx
        elif token == special_ids.img_end:
            spans.append(tokens[start_idx : idx + 1])
            start_idx = None
    return spans


def build_mistral_tokenizer(
    instruct_tokenizer: InstructTokenizer,
    version: TokenizerVersion,
    *,
    mode: ValidationMode = ValidationMode.test,
    model_settings_builder: ModelSettingsBuilder | None = None,
) -> MistralTokenizer:
    r"""Wrap an instruct tokenizer in a `MistralTokenizer` with matching normalizer and validator.

    Replaces the repeated `get_normalizer` / `get_validator` / `MistralTokenizer(...)`
    assembly in the version-specific instruct tokenizer tests.

    Args:
        instruct_tokenizer: The instruct tokenizer to wrap.
        version: The tokenizer version driving normalizer and validator selection.
        mode: The validation mode.
        model_settings_builder: Model settings builder passed to the normalizer.

    Returns:
        A `MistralTokenizer` around the instruct tokenizer.
    """
    return MistralTokenizer(
        instruct_tokenizer=instruct_tokenizer,
        validator=get_validator(version, mode=mode),
        request_normalizer=get_normalizer(version, model_settings_builder),
    )


_INSTRUCT_TOKENIZER_CLASSES: dict[TokenizerVersion, type[InstructTokenizer]] = {
    TokenizerVersion.v1: InstructTokenizerV1,
    TokenizerVersion.v2: InstructTokenizerV2,
    TokenizerVersion.v3: InstructTokenizerV3,
    TokenizerVersion.v7: InstructTokenizerV7,
    TokenizerVersion.v11: InstructTokenizerV11,
    TokenizerVersion.v13: InstructTokenizerV13,
    TokenizerVersion.v15: InstructTokenizerV15,
}


def instruct_tokenizer_class(version: TokenizerVersion) -> type[InstructTokenizer]:
    r"""Return the `InstructTokenizer` subclass matching a tokenizer version.

    Args:
        version: The tokenizer version.

    Returns:
        The `InstructTokenizer` subclass for the version.

    Raises:
        ValueError: If no instruct tokenizer class is registered for the version.
    """
    if version not in _INSTRUCT_TOKENIZER_CLASSES:
        raise ValueError(f"No instruct tokenizer class registered for version {version}")
    return _INSTRUCT_TOKENIZER_CLASSES[version]


def build_instruct_tokenizer_from_config(
    config: TestConfig,
    *,
    model_settings_builder: ModelSettingsBuilder | None = None,
) -> InstructTokenizer:
    r"""Build an in-memory `InstructTokenizer` from a `TestConfig`.

    The underlying `Tekkenizer` comes from `build_tekkenizer`, and image/audio encoders
    are attached from `build_image_encoder` / `build_audio_encoder` when the config
    enables those modalities. This is the single construction path behind the shared
    `instruct_tokenizer` and `mistral_tokenizer` fixtures.

    Args:
        config: The version + modality configuration to build.
        model_settings_builder: Optional model settings builder, for versions that support it.

    Returns:
        An `InstructTokenizer` of the subclass matching `config.version`.
    """
    tokenizer = build_tekkenizer(
        version=config.version,
        add_audio=config.audio,
        add_think=config.think,
        model_settings_builder=model_settings_builder,
    )
    return instruct_tokenizer_class(config.version)(
        tokenizer,
        image_encoder=build_image_encoder(tokenizer) if config.image else None,
        audio_encoder=build_audio_encoder(tokenizer) if config.audio else None,
    )


def build_mistral_tokenizer_from_config(
    config: TestConfig,
    *,
    mode: ValidationMode = ValidationMode.test,
    model_settings_builder: ModelSettingsBuilder | None = None,
) -> MistralTokenizer:
    r"""Build an in-memory `MistralTokenizer` from a `TestConfig`.

    Args:
        config: The version + modality configuration to build.
        mode: The validation mode.
        model_settings_builder: Optional model settings builder, for versions that support it.

    Returns:
        A `MistralTokenizer` wrapping the configured instruct tokenizer.
    """
    instruct_tokenizer = build_instruct_tokenizer_from_config(config, model_settings_builder=model_settings_builder)
    return build_mistral_tokenizer(
        instruct_tokenizer,
        config.version,
        mode=mode,
        model_settings_builder=model_settings_builder,
    )


def build_model_settings_builder(
    reasoning_efforts: tuple[ReasoningEffort, ...] | None,
    *,
    use_default: bool = True,
) -> ModelSettingsBuilder:
    r"""Build a `ModelSettingsBuilder` accepting the given reasoning effort values.

    Args:
        reasoning_efforts: The accepted reasoning effort values. `None` yields a builder
            that ignores every field, matching a tokenizer file without a
            `model_settings_builder` key.
        use_default: Whether the first accepted value becomes the default. When `False`
            the builder has no default, so an unset reasoning effort encodes nothing.

    Returns:
        The model settings builder.
    """
    if reasoning_efforts is None:
        return ModelSettingsBuilder.none()
    default = reasoning_efforts[0] if (use_default and reasoning_efforts) else None
    return ModelSettingsBuilder(
        reasoning_effort=EnumBuilder[ReasoningEffort](
            values=list(reasoning_efforts),
            accepts_none=True,
            default=default,
        )
    )


def build_spm_sentinel_compatible_tekkenizer(version: TokenizerVersion) -> Tekkenizer:
    r"""Build a synthetic `Tekkenizer` whose vocab satisfies the SPM sentinel invariant.

    `InstructTokenizerV2._encode_infilling` (inherited by every tekken instruct tokenizer
    subclass) is an SPM-shaped workaround, not a tekken concept -- see
    `SPM_INFILLING_SENTINEL`. The default `build_tekkenizer` vocabulary is truncated below
    its extra tokens, so the sentinel encodes into three tokens there, which breaks the
    `[2:]` slice `_encode_infilling` relies on. This vocabulary instead adds
    `SPM_INFILLING_SENTINEL_MERGE` as an extra token wide enough to be kept, and uses the
    real tekken pattern (`TEKKEN_PATTERN`) so newlines in a FIM suffix are chunked the same
    way a production tokenizer would chunk them.

    Args:
        version: The tokenizer version to build.

    Returns:
        A `Tekkenizer` whose vocab encodes `SPM_INFILLING_SENTINEL` in exactly two tokens.
    """
    extra_toks = (SPM_INFILLING_SENTINEL_MERGE,)
    return build_tekkenizer(
        version=version,
        extra_toks=extra_toks,
        pattern=TEKKEN_PATTERN,
        vocab_size=256 + len(extra_toks) + 100,
    )


def build_spm_sentinel_compatible_mistral_tokenizer(version: TokenizerVersion) -> MistralTokenizer:
    r"""Build a synthetic `MistralTokenizer` whose vocab satisfies the SPM sentinel invariant.

    Args:
        version: The tokenizer version to build.

    Returns:
        A `MistralTokenizer` wrapping a synthetic instruct tokenizer whose vocab is safe
        for `_encode_infilling`, so it can encode FIM requests.
    """
    tokenizer = build_spm_sentinel_compatible_tekkenizer(version=version)
    instruct_tokenizer = instruct_tokenizer_class(version)(tokenizer, image_encoder=None, audio_encoder=None)
    return build_mistral_tokenizer(instruct_tokenizer, version)


# Every reasoning effort value, used by the `v15_*` synthetic factories below so their
# model settings builder accepts (and defaults to) the full range.
_ALL_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = tuple(ReasoningEffort)

# Single source of truth for every `MistralTokenizer` the test suite builds, keyed by a
# backend-qualified name. Shipped tokenizers exist only for v1 (spm), v2 (spm), v3
# (spm + tekken, + mm) and v7 (spm + mm) -- tekken v1/v2 were never released. v11/v13/v15
# are tekken-only released versions with no shipped file bundled for tests, so their
# factories fall back to a synthetic tokenizer via
# `build_spm_sentinel_compatible_mistral_tokenizer`. There is no separate synthetic `v2`
# key: the shipped v2 SentencePiece model has no `[SUFFIX]`/`[PREFIX]` pieces, so it cannot
# exercise `encode_fim` at all, and no synthetic stand-in would faithfully represent that
# gap (see `SUPPORTED_PROTOCOLS` in `tests.utils.registry`).
# `v7_tekken`/`v7_tekken_aud` are synthetic in-memory tekken tokenizers for v7 built
# directly via `build_mistral_tokenizer_from_config`, unlike the shipped `v7_spm*` keys
# above: no tekken v7 file is bundled for tests, and some tests need the exact synthetic
# tekken vocab the `instruct_tokenizer`/`mistral_tokenizer` fixtures build for `TestConfig`
# rather than the shipped SentencePiece data.
# `v13_think`/`v13_aud`/`v15_think`/`v15_aud`/`v15_img_think` are synthetic instruct-only
# tokenizers (not FIM-capable) carrying the modality/model-settings combinations the golden
# registry's instruct scenarios need but the FIM-capable factories above cannot build.
TOKENIZER_FACTORIES: dict[str, Callable[[], MistralTokenizer]] = {
    "v1_spm": MistralTokenizer.v1,
    "v2_spm": MistralTokenizer.v2,
    "v3_spm": MistralTokenizer.v3,
    "v3_tekken": lambda: MistralTokenizer.v3(is_tekken=True),
    "v3_tekken_mm": lambda: MistralTokenizer.v3(is_tekken=True, is_mm=True),
    "v7_spm": MistralTokenizer.v7,
    "v7_spm_mm": lambda: MistralTokenizer.v7(is_mm=True),
    "v7_tekken": lambda: build_mistral_tokenizer_from_config(TestConfig(version=TokenizerVersion.v7)),
    "v7_tekken_aud": lambda: build_mistral_tokenizer_from_config(TestConfig(version=TokenizerVersion.v7, audio=True)),
    "v11": lambda: build_spm_sentinel_compatible_mistral_tokenizer(TokenizerVersion.v11),
    "v13": lambda: build_spm_sentinel_compatible_mistral_tokenizer(TokenizerVersion.v13),
    "v15": lambda: build_spm_sentinel_compatible_mistral_tokenizer(TokenizerVersion.v15),
    "v13_think": lambda: build_mistral_tokenizer_from_config(TestConfig(version=TokenizerVersion.v13, think=True)),
    "v13_aud": lambda: build_mistral_tokenizer_from_config(TestConfig(version=TokenizerVersion.v13, audio=True)),
    "v15_think": lambda: build_mistral_tokenizer_from_config(
        TestConfig(version=TokenizerVersion.v15, think=True),
        model_settings_builder=build_model_settings_builder(_ALL_REASONING_EFFORTS),
    ),
    "v15_aud": lambda: build_mistral_tokenizer_from_config(
        TestConfig(version=TokenizerVersion.v15, audio=True),
        model_settings_builder=build_model_settings_builder(_ALL_REASONING_EFFORTS),
    ),
    "v15_img_think": lambda: build_mistral_tokenizer_from_config(
        TestConfig(version=TokenizerVersion.v15, image=True, think=True),
        model_settings_builder=build_model_settings_builder(_ALL_REASONING_EFFORTS),
    ),
}


# Real, shipped tokenizer keys (a subset of `TOKENIZER_FACTORIES`); `v11`, `v13`, `v15`
# are synthetic-only and excluded here.
_SHIPPED_TOKENIZER_BASE_KEYS: tuple[str, ...] = (
    "v1_spm",
    "v2_spm",
    "v3_spm",
    "v3_tekken",
    "v3_tekken_mm",
    "v7_spm",
    "v7_spm_mm",
)

# Shipped keys that support `encode_fim` (every shipped key except `v1_spm` and `v2_spm` --
# the shipped v2 SentencePiece model lacks FIM marker pieces -- and excluding the multimodal
# variants, which encode FIM identically to their text counterpart).
SHIPPED_FIM_CAPABLE_KEYS: tuple[str, ...] = ("v3_spm", "v3_tekken", "v7_spm")

# Synthetic keys that support `encode_fim`, built via
# `build_spm_sentinel_compatible_mistral_tokenizer`; v11/v13/v15 are tekken-only released
# versions with no shipped file to test against. There is deliberately no `v2` entry here:
# the shipped v2 SentencePiece model cannot encode FIM at all (see `TOKENIZER_FACTORIES`
# above), so a synthetic FIM-capable stand-in for it would test fictional behavior.
SYNTHETIC_FIM_CAPABLE_KEYS: tuple[str, ...] = ("v11", "v13", "v15")

# FIM-capable keys split by backend, derived from the two tuples above. `SPM_INFILLING_SENTINEL`
# only strips a real SentencePiece prefix space, so `_encode_infilling` behaves differently on
# each side of this split -- see `tests.tokenizers.test_fim`.
SPM_FIM_CAPABLE_KEYS: tuple[str, ...] = tuple(key for key in SHIPPED_FIM_CAPABLE_KEYS if key.endswith("_spm"))
TEKKEN_FIM_CAPABLE_KEYS: tuple[str, ...] = (
    tuple(key for key in SHIPPED_FIM_CAPABLE_KEYS if key not in SPM_FIM_CAPABLE_KEYS) + SYNTHETIC_FIM_CAPABLE_KEYS
)

# Multimodal keys suffixed with `_small_patch` load a separate instance whose image patch
# size is shrunk to `TEST_IMAGE_PATCH_SIZE`, so tests that need small, hand-checkable token
# counts never mutate the tokenizer other tests (and the goldens) rely on.
SMALL_PATCH_SUFFIX = "_small_patch"


@dataclass(frozen=True)
class KeyCapabilities:
    r"""Explicit, hand-authored capabilities of one `TOKENIZER_FACTORIES` key.

    This is the single source of truth `tests.utils.registry` filters the golden scenario
    matrix against: a (scenario, key) pair is only generated when the scenario's derived
    requirements (see `tests.utils.registry._derive_requirements`) are met by the key's
    capabilities here. Capabilities are never inferred from a key's name (e.g. `"_aud" in
    key`) -- that both name and the modality it implies come from this one table.

    Attributes:
        version: The tokenizer version.
        backend: The tokenizer backend, `"spm"` or `"tekken"`.
        image: Whether the tokenizer carries an image encoder.
        audio: Whether the tokenizer carries an audio encoder.
        think: Whether the tokenizer's special tokens include think tokens.
        model_settings: Whether the tokenizer has a real (non-`None`) model settings
            builder attached to both its normalizer and its underlying tokenizer. A v15+
            key without one (e.g. `"v15"`) cannot encode any instruct request at all, not
            only ones that set model settings explicitly.
    """

    version: TokenizerVersion
    backend: Literal["spm", "tekken"]
    image: bool = False
    audio: bool = False
    think: bool = False
    model_settings: bool = False


# Every `TOKENIZER_FACTORIES` key's capabilities, plus the two `SMALL_PATCH_SUFFIX`
# variants `load_mistral_tokenizer` builds on top of their base key (same capabilities,
# only the image patch size differs). `"v15"` deliberately has no `model_settings`
# builder: it exists only to exercise `encode_fim`, which never touches model settings.
KEY_CAPABILITIES: dict[str, KeyCapabilities] = {
    "v1_spm": KeyCapabilities(version=TokenizerVersion.v1, backend="spm"),
    "v2_spm": KeyCapabilities(version=TokenizerVersion.v2, backend="spm"),
    "v3_spm": KeyCapabilities(version=TokenizerVersion.v3, backend="spm"),
    "v3_tekken": KeyCapabilities(version=TokenizerVersion.v3, backend="tekken"),
    "v3_tekken_mm": KeyCapabilities(version=TokenizerVersion.v3, backend="tekken", image=True),
    "v3_tekken_mm" + SMALL_PATCH_SUFFIX: KeyCapabilities(version=TokenizerVersion.v3, backend="tekken", image=True),
    "v7_spm": KeyCapabilities(version=TokenizerVersion.v7, backend="spm"),
    "v7_spm_mm": KeyCapabilities(version=TokenizerVersion.v7, backend="spm", image=True),
    "v7_spm_mm" + SMALL_PATCH_SUFFIX: KeyCapabilities(version=TokenizerVersion.v7, backend="spm", image=True),
    "v7_tekken": KeyCapabilities(version=TokenizerVersion.v7, backend="tekken"),
    "v7_tekken_aud": KeyCapabilities(version=TokenizerVersion.v7, backend="tekken", audio=True),
    "v11": KeyCapabilities(version=TokenizerVersion.v11, backend="tekken"),
    "v13": KeyCapabilities(version=TokenizerVersion.v13, backend="tekken"),
    "v13_think": KeyCapabilities(version=TokenizerVersion.v13, backend="tekken", think=True),
    "v13_aud": KeyCapabilities(version=TokenizerVersion.v13, backend="tekken", audio=True),
    "v15": KeyCapabilities(version=TokenizerVersion.v15, backend="tekken"),
    "v15_think": KeyCapabilities(version=TokenizerVersion.v15, backend="tekken", think=True, model_settings=True),
    "v15_aud": KeyCapabilities(version=TokenizerVersion.v15, backend="tekken", audio=True, model_settings=True),
    "v15_img_think": KeyCapabilities(
        version=TokenizerVersion.v15, backend="tekken", image=True, think=True, model_settings=True
    ),
}

SHIPPED_TOKENIZER_KEYS: tuple[str, ...] = _SHIPPED_TOKENIZER_BASE_KEYS + (
    "v3_tekken_mm" + SMALL_PATCH_SUFFIX,
    "v7_spm_mm" + SMALL_PATCH_SUFFIX,
)


@lru_cache(maxsize=None)
def load_mistral_tokenizer(key: str) -> MistralTokenizer:
    r"""Load (and cache) a `MistralTokenizer` by key.

    Keys ending in `SMALL_PATCH_SUFFIX` return a separate instance whose image patch size
    is shrunk to `TEST_IMAGE_PATCH_SIZE`; every other key returns the tokenizer exactly as
    built by `TOKENIZER_FACTORIES`.

    Args:
        key: A key from `TOKENIZER_FACTORIES`, optionally suffixed with `SMALL_PATCH_SUFFIX`.

    Returns:
        The `MistralTokenizer` for the key.

    Raises:
        KeyError: If the key is unknown.
    """
    small_patch = key.endswith(SMALL_PATCH_SUFFIX)
    base_key = key[: -len(SMALL_PATCH_SUFFIX)] if small_patch else key
    tokenizer = TOKENIZER_FACTORIES[base_key]()
    if small_patch:
        set_test_image_patch_size(tokenizer.instruct_tokenizer)
    return tokenizer

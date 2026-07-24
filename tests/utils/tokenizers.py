import base64
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mistral_common.tokens.tokenizers.base import SpecialTokens, TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import ModelSettingsBuilder
from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenInfo, Tekkenizer, TokenInfo
from tests.utils.versions import TestConfig


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
                token_str=tok.decode(),
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

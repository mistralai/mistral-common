r"""Reusable helper functions for chat template tests.

This module contains utility functions used by test files across the
`unit/` and `transformers/` subdirectories.  It intentionally has no
fixtures or test data -- those live in `conftest.py` and `fixtures_data.py`.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from jinja2 import BaseLoader
from jinja2.sandbox import ImmutableSandboxedEnvironment
from PIL import Image

try:
    from transformers import PreTrainedTokenizerFast
    from transformers.utils.chat_template_utils import render_jinja_template

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from mistral_common.audio import Audio
from mistral_common.integrations.chat_templates.template_generator import TemplateConfig
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.instruct.normalize import get_normalizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.protocol.instruct.validator import ValidationMode, get_validator
from mistral_common.tokens.tokenizers.audio import (
    AudioConfig,
    AudioEncoder,
    AudioSpectrogramConfig,
    SpecialAudioIDs,
)
from mistral_common.tokens.tokenizers.base import InstructTokenizer, Tokenizer, TokenizerVersion
from mistral_common.tokens.tokenizers.image import ImageConfig, ImageEncoder, SpecialImageIDs
from mistral_common.tokens.tokenizers.instruct import (
    InstructTokenizerBase,
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
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from tests.test_tekken import get_special_tokens

# Golden template files live in the data/ tree (outside src/).
_GOLDEN_DIR = Path(__file__).parent.parent.parent / "data" / "chat_templates"

SPM_WHITESPACE = "▁"


@dataclass(frozen=True)
class TestConfig:
    r"""Test configuration for chat template parametrization."""

    __test__ = False

    version: TokenizerVersion
    spm: bool = False
    image: bool = False
    audio: bool = False
    think: bool = False
    plain_think: bool = False


def _make_config(c: TestConfig) -> TemplateConfig:
    r"""Create a `TemplateConfig` from a test config."""
    return TemplateConfig(
        version=c.version,
        spm=c.spm,
        image_support=c.image,
        audio_support=c.audio,
        thinking_support=c.think,
        plain_thinking_support=c.plain_think,
        use_special_token_variables=True,
    )


def _load_golden_template(config: TemplateConfig) -> str:
    r"""Load the static golden template for a config."""
    parts = [config.version.value]
    if config.image_support and config.any_thinking_support:
        parts.append("image_think")
    elif config.image_support:
        parts.append("image")
    elif config.audio_support:
        parts.append("audio")
    elif config.any_thinking_support:
        parts.append("think")
    if config.spm:
        parts.append("spm")
    filename = "_".join(parts) + ".jinja"
    path = _GOLDEN_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Golden template not found: {path}")
    return path.read_text()


def render_template(
    template: str, messages: list[Any], tools: list[Any] | None = None, reasoning_effort: str | None = None
) -> str:
    r"""Render a Jinja2 template with the given messages using a pure Jinja2 sandbox.

    This function tests template rendering independently of HuggingFace transformers.
    It uses ImmutableSandboxedEnvironment directly, while `encode_transformers`
    tests integration with the HuggingFace `render_jinja_template` function.
    Both should produce identical output for the same input.
    """

    def raise_exception(msg: str) -> None:
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(loader=BaseLoader())
    env.globals["raise_exception"] = raise_exception
    jinja_template = env.from_string(template)

    render_kwargs: dict[str, Any] = {
        "messages": messages,
        "tools": tools,
        "bos_token": "<s>",
        "eos_token": "</s>",
    }

    # Only add reasoning_effort for v15+ templates that support it
    if reasoning_effort is not None:
        render_kwargs["reasoning_effort"] = reasoning_effort

    return jinja_template.render(**render_kwargs)


def encode_mistral_common(mistral_tokenizer: MistralTokenizer, chat_request: ChatCompletionRequest, spm: bool) -> str:
    r"""Encode a chat request using mistral-common tokenizer.

    Returns the text representation (not token IDs) for parity comparison
    with the Jinja template rendering. Token ID comparison is handled
    separately via `encode_hf_tokens`.
    """
    mistral_encoded = str(mistral_tokenizer.encode_chat_completion(chat_request).text)
    # Collapse each image token sequence ([IMG]...[IMG_BREAK]...[IMG_END]) into a single [IMG].
    # Each sequence has exactly one [IMG_END], so replacing inner tokens and converting [IMG_END]
    # preserves one [IMG] per image (important for multi-image inputs).
    mistral_encoded = mistral_encoded.replace("[IMG]", "").replace("[IMG_BREAK]", "").replace("[IMG_END]", "[IMG]")
    # Remove audio tokens except one per audio
    mistral_encoded = mistral_encoded.replace("[AUDIO]", "").replace("[BEGIN_AUDIO]", "[AUDIO]")
    if spm:
        mistral_encoded = mistral_encoded.replace(SPM_WHITESPACE, " ").replace("<0x0A>", "\n")
    return mistral_encoded


def _to_openai_request(chat_request: ChatCompletionRequest, keep_name_for_tools: bool = False) -> dict[str, Any]:
    r"""Convert a ChatCompletionRequest to OpenAI format.

    Args:
        chat_request: The chat completion request to convert.
        keep_name_for_tools: If True, preserve tool message `name` fields
            that `to_openai` strips by default.

    Returns:
        OpenAI-format request dictionary.
    """
    openai_request = chat_request.to_openai()
    if keep_name_for_tools:
        for openai_message, chat_message in zip(openai_request["messages"], chat_request.messages):
            if chat_message.role == "tool":
                openai_message["name"] = chat_message.name
    return openai_request


def encode_transformers(
    chat_template: str, chat_request: ChatCompletionRequest, keep_name_for_tools: bool = False
) -> str:
    r"""Encode a chat request using the transformers render_jinja_template.

    Converts a ChatCompletionRequest to OpenAI format and renders through
    the HuggingFace transformers rendering pipeline.
    """
    assert _HAS_TRANSFORMERS, "transformers is required"
    openai_request = _to_openai_request(chat_request, keep_name_for_tools)
    return _render_via_transformers(chat_template, openai_request)


def encode_transformers_from_openai(chat_template: str, openai_request: dict[str, Any]) -> str:
    r"""Encode a raw OpenAI-format dict using the transformers render_jinja_template.

    Use this for tests with hand-crafted dicts (e.g., invalid input tests)
    that cannot be represented as a ChatCompletionRequest.
    """
    assert _HAS_TRANSFORMERS, "transformers is required"
    return _render_via_transformers(chat_template, openai_request)


def _build_hf_tokenizer(tekken_path: Path, chat_template: str) -> "PreTrainedTokenizerFast":
    r"""Build a `PreTrainedTokenizerFast` from a tekken.json file.

    Converts the Tekken BPE vocabulary to a HuggingFace tokenizers format,
    preserving the Tekkenizer's ID scheme: special tokens at IDs 0 to
    `num_special - 1`, regular BPE tokens at IDs `num_special` and above.

    All special tokens in the file are unconditionally registered as
    `AddedToken` objects, matching Tekkenizer behavior where every token
    in the special range is treated as special.

    .. note::

        TODO: The ID-offset conversion done here should be fixed upstream
        in transformers' `MistralConverter` so that it produces IDs matching
        the Tekkenizer natively. Until then, we apply the offset manually to
        ensure a fair token-level comparison.

    Args:
        tekken_path: Path to the tekken.json file.
        chat_template: Jinja chat template string to set on the tokenizer.

    Returns:
        A configured `PreTrainedTokenizerFast` with matching token IDs.
    """
    import base64
    from functools import lru_cache

    from tokenizers import Regex, decoders, pre_tokenizers, processors
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from transformers import AddedToken
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    with open(tekken_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pattern: str = data["config"]["pattern"]
    special_tokens_list: list[dict[str, Any]] = data["special_tokens"]
    num_special: int = data["config"]["default_num_special_tokens"]
    vocab_size: int = data["config"]["default_vocab_size"]
    inner_vocab_size = vocab_size - num_special
    vocab_entries: list[dict[str, Any]] = data["vocab"][:inner_vocab_size]

    byte_encoder = bytes_to_unicode()

    @lru_cache
    def token_bytes_to_string(b: bytes) -> str:
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    # Decode raw bytes for each vocab entry
    raw_tokens = [base64.b64decode(entry["token_bytes"]) for entry in vocab_entries]
    rank_set = set(raw_tokens)
    token_to_rank: dict[bytes, int] = {token: rank for rank, token in enumerate(raw_tokens)}

    # Build BPE vocab with regular tokens at IDs num_special+ and special tokens at 0..num_special-1.
    # All special token slots (active and filler) must appear in the BPE vocab so the
    # HuggingFace tokenizer recognizes IDs 0 through num_special-1.
    bpe_vocab: dict[str, int] = {}
    defined_ids: set[int] = set()
    for st in special_tokens_list:
        bpe_vocab[st["token_str"]] = st["rank"]
        defined_ids.add(st["rank"])
    for i in range(num_special):
        if i not in defined_ids:
            bpe_vocab[f"<SPECIAL_{i}>"] = i
    for rank, token in enumerate(raw_tokens):
        bpe_vocab[token_bytes_to_string(token)] = rank + num_special

    # Extract BPE merges (same algorithm as MistralConverter.extract_vocab_merges_from_model)
    merges: list[tuple[bytes, bytes, int]] = []
    for rank, token in enumerate(raw_tokens):
        if len(token) == 1:
            continue
        local: list[tuple[bytes, bytes, int]] = []
        for index in range(1, len(token)):
            piece_l, piece_r = token[:index], token[index:]
            if piece_l in rank_set and piece_r in rank_set and (piece_l + piece_r) in rank_set:
                local.append((piece_l, piece_r, rank))
        local = sorted(local, key=lambda x: (token_to_rank[x[0]], token_to_rank[x[1]]))
        merges.extend(local)
    merges = sorted(merges, key=lambda val: val[2])
    bpe_merges = [(token_bytes_to_string(m[0]), token_bytes_to_string(m[1])) for m in merges]

    # Create tokenizer with BPE model
    tokenizer = HFTokenizer(BPE(bpe_vocab, bpe_merges, fuse_unk=False))
    if hasattr(tokenizer.model, "ignore_merges"):
        tokenizer.model.ignore_merges = True

    # Set pre-tokenizer, decoder, and post-processor
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Register all special tokens in added_tokens_decoder, matching Tekkenizer
    # behavior where every token in the special range is treated as special.
    added_tokens_decoder: dict[int, AddedToken] = {}
    for st in special_tokens_list:
        added_tokens_decoder[st["rank"]] = AddedToken(st["token_str"], special=True, normalized=False)
    for i in range(num_special):
        if i not in defined_ids:
            added_tokens_decoder[i] = AddedToken(f"<SPECIAL_{i}>", special=True, normalized=False)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        added_tokens_decoder=added_tokens_decoder,
        bos_token="<s>",
        eos_token="</s>",
        chat_template=chat_template,
    )


def encode_hf_tokens(
    hf_tokenizer: "PreTrainedTokenizerFast", chat_request: ChatCompletionRequest, keep_name_for_tools: bool = False
) -> list[int]:
    r"""Get token IDs from a standard HuggingFace tokenizer.

    Converts the `ChatCompletionRequest` to OpenAI format and encodes via
    `apply_chat_template(tokenize=True)`.

    Args:
        hf_tokenizer: A `PreTrainedTokenizerFast` instance.
        chat_request: The chat completion request to encode.
        keep_name_for_tools: If True, preserve tool message `name` fields.

    Returns:
        Token IDs produced by the HuggingFace tokenizer.
    """
    openai_request = _to_openai_request(chat_request, keep_name_for_tools)
    messages = openai_request["messages"]

    tools = openai_request.get("tools", None)
    if tools is not None:
        for tool in tools:
            tool["function"].pop("strict", None)

    template_kwargs: dict[str, Any] = {}
    reasoning_effort = openai_request.get("reasoning_effort")
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    result = hf_tokenizer.apply_chat_template(
        conversation=messages,
        tools=tools,
        tokenize=True,
        return_dict=False,
        padding=False,
        truncation=False,
        **template_kwargs,
    )
    return cast(list[int], result)


def _render_via_transformers(chat_template: str, openai_request: dict[str, Any]) -> str:
    r"""Shared rendering logic for transformers-based encoding."""
    for tool in openai_request.get("tools", []):
        tool["function"].pop("strict", False)

    reasoning_effort = openai_request.get("reasoning_effort")
    template_kwargs: dict[str, Any] = {}
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    encoded = render_jinja_template(
        [openai_request["messages"]],
        tools=openai_request.get("tools", None),
        chat_template=chat_template,
        bos_token="<s>",
        eos_token="</s>",
        **template_kwargs,
    )[0][0]
    assert isinstance(encoded, str), type(encoded)
    return encoded


def _create_dummy_image() -> Image.Image:
    r"""Create a simple dummy 2x2 red square image for testing."""
    return Image.new("RGB", (2, 2), color="red")


def _sin_wave(sampling_rate: int, duration: float) -> np.ndarray:
    r"""Generate a sine wave numpy array."""
    return np.sin(np.ones([int(duration * sampling_rate)]))


def _sample_audio() -> Audio:
    r"""Create a sample `Audio` instance for testing."""
    sampling_rate = 44100
    original_array = _sin_wave(sampling_rate, 1)
    return Audio(
        audio_array=original_array,
        sampling_rate=sampling_rate,
        format="wav",
    )


_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/7/78/Red_Square_%282x2_Pixel%29.png"
_IMAGE = _create_dummy_image()
_AUDIO_URL = _sample_audio().to_base64("wav")
_AUDIO = RawAudio(data=_AUDIO_URL, format="wav")


def _get_image_encoder(tokenizer: Tokenizer) -> ImageEncoder:
    r"""Create an `ImageEncoder` for testing."""
    image_config = ImageConfig(image_patch_size=2, max_image_size=10, spatial_merge_size=1)
    return ImageEncoder(
        image_config=image_config,
        special_ids=SpecialImageIDs(
            img=tokenizer.get_special_token("[IMG]"),
            img_break=tokenizer.get_special_token("[IMG_BREAK]"),
            img_end=tokenizer.get_special_token("[IMG_END]"),
        ),
    )


def _get_audio_encoder() -> AudioEncoder:
    r"""Create an `AudioEncoder` for testing."""
    audio_config = AudioConfig(
        sampling_rate=24_000,
        frame_rate=12.5,
        encoding_config=AudioSpectrogramConfig(
            num_mel_bins=128,
            window_size=400,
            hop_length=160,
        ),
    )
    return AudioEncoder(
        audio_config=audio_config,
        special_ids=SpecialAudioIDs(audio=24, begin_audio=25, streaming_pad=26, text_to_audio=27, audio_to_text=28),
    )


def _build_tekken_json(config: TestConfig, output_dir: Path) -> Path:
    r"""Build a tekken.json file for the given test config.

    Constructs a complete Tekken tokenizer JSON file by combining the base
    vocabulary from the shipped tokenizer with version-specific special tokens
    and optional image/audio/model_settings configuration. The file is written
    to `output_dir / "tekken.json"` and can be loaded by
    `MistralTokenizer.from_file`.

    Args:
        config: Test configuration specifying version and feature flags.
        output_dir: Directory to write the JSON file into.

    Returns:
        Path to the written tekken.json file.
    """
    with open(MistralTokenizer._data_path() / "tekken_240911.json", "r", encoding="utf-8") as f:
        base_data = json.load(f)

    special_tokens = get_special_tokens(
        tokenizer_version=config.version, add_audio=config.audio, add_think=config.think
    )

    tekken_data: dict[str, Any] = {
        "config": {
            "pattern": base_data["config"]["pattern"],
            "default_vocab_size": base_data["config"]["default_vocab_size"],
            "default_num_special_tokens": 100,
            "version": config.version.value,
        },
        "vocab": base_data["vocab"],
        "special_tokens": special_tokens,
    }

    if config.image:
        tekken_data["image"] = {
            "image_patch_size": 2,
            "max_image_size": 10,
            "spatial_merge_size": 1,
        }

    if config.audio:
        tekken_data["audio"] = {
            "sampling_rate": 24_000,
            "frame_rate": 12.5,
            "audio_encoding_config": {
                "num_mel_bins": 128,
                "window_size": 400,
                "hop_length": 160,
            },
        }

    if config.version.supports_model_settings:
        model_settings_builder = ModelSettingsBuilder(
            reasoning_effort=EnumBuilder(
                accepts_none=True,
                default=ReasoningEffort.none,
                values=[ReasoningEffort.none, ReasoningEffort.high],
            )
        )
        tekken_data["model_settings_builder"] = model_settings_builder.model_dump(mode="json")

    tekken_path = output_dir / "tekken.json"
    with open(tekken_path, "w", encoding="utf-8") as f:
        json.dump(tekken_data, f, ensure_ascii=False)

    return tekken_path


def _build_spm_path(config: TestConfig, output_dir: Path) -> Path:
    r"""Copy the SPM model file with the correct version suffix.

    The SPM tokenizer version is determined by the filename suffix (e.g.,
    `.model.v3` for v3). Image support is indicated by an `m1` suffix
    (e.g., `.model.v3m1`). The shipped v7m1 model file is copied with the
    appropriate suffix for the requested config.

    Args:
        config: Test configuration specifying version and image flag.
        output_dir: Directory to copy the file into.

    Returns:
        Path to the copied model file.
    """
    source = MistralTokenizer._data_path() / "mistral_instruct_tokenizer_241114.model.v7m1"
    suffix = f".model.{config.version.value}"
    if config.image:
        suffix += "m1"
    dest = output_dir / f"tokenizer{suffix}"
    shutil.copy2(source, dest)
    return dest


def _get_instruct_tokenizer_class(tokenizer_version: TokenizerVersion) -> type[InstructTokenizerBase]:
    r"""Get the instruct tokenizer class for a given tokenizer version.

    Args:
        tokenizer_version: Tokenizer version to look up.

    Returns:
        The instruct tokenizer class for the given version.
    """
    match tokenizer_version:
        case TokenizerVersion.v1:
            return InstructTokenizerV1
        case TokenizerVersion.v2:
            return InstructTokenizerV2
        case TokenizerVersion.v3:
            return InstructTokenizerV3
        case TokenizerVersion.v7:
            return InstructTokenizerV7
        case TokenizerVersion.v11:
            return InstructTokenizerV11
        case TokenizerVersion.v13:
            return InstructTokenizerV13
        case TokenizerVersion.v15:
            return InstructTokenizerV15
        case _:
            raise ValueError(f"Unknown tokenizer version: {tokenizer_version}")


def _get_mistral_tekkenizer(
    tokenizer_version: TokenizerVersion, validation_mode: ValidationMode, image: bool, audio: bool, think: bool
) -> MistralTokenizer:
    r"""Build a `MistralTokenizer` with Tekken backend.

    We construct the tokenizer manually instead of using `MistralTokenizer.from_file`
    because `from_file` reads version, special tokens, and image/audio config from the
    JSON file with no override mechanism. Tests need to pair a single base tokenizer file
    (`tekken_240911.json`, which is v3) with arbitrary versions (v1-v15) and feature
    combinations (image, audio, thinking) that aren't present in any shipped JSON file.
    """
    special_tokens = get_special_tokens(tokenizer_version=tokenizer_version, add_audio=audio, add_think=think)
    with open(MistralTokenizer._data_path() / "tekken_240911.json", "r", encoding="utf-8") as f:
        json_tekkenizer = json.load(f)
    vocab = json_tekkenizer["vocab"]
    vocab_size = json_tekkenizer["config"]["default_vocab_size"]
    pattern = json_tekkenizer["config"]["pattern"]
    # ModelSettingsBuilder is constructed manually because from_file reads it from the
    # JSON file's model_settings_builder key, which is absent in fixture tokenizer files.
    # This mirrors what from_file would produce for a v15 tokenizer config.
    model_settings_builder = (
        ModelSettingsBuilder(
            reasoning_effort=EnumBuilder(
                accepts_none=True, default=ReasoningEffort.none, values=[ReasoningEffort.none, ReasoningEffort.high]
            )
        )
        if tokenizer_version.supports_model_settings
        else None
    )
    tokenizer = Tekkenizer(
        vocab,
        special_tokens,
        pattern=pattern,
        vocab_size=vocab_size,
        num_special_tokens=100,
        version=tokenizer_version,
        model_settings_builder=model_settings_builder,
    )

    audio_encoder = _get_audio_encoder() if audio else None
    image_encoder = _get_image_encoder(tokenizer) if image else None

    instruct_tokenizer: InstructTokenizer = _get_instruct_tokenizer_class(tokenizer_version)(
        tokenizer=tokenizer,
        image_encoder=image_encoder,
        audio_encoder=audio_encoder,
    )
    model_settings_builder = tokenizer.model_settings_builder if isinstance(tokenizer, Tekkenizer) else None
    return MistralTokenizer(
        instruct_tokenizer,
        validator=get_validator(mode=validation_mode, version=tokenizer_version),
        request_normalizer=get_normalizer(version=tokenizer_version, model_settings_builder=model_settings_builder),
    )


def _get_mistral_sentencepiece(
    tokenizer_version: TokenizerVersion, validation_mode: ValidationMode, image: bool
) -> MistralTokenizer:
    r"""Build a `MistralTokenizer` with SentencePiece backend."""
    tokenizer = SentencePieceTokenizer(
        MistralTokenizer._data_path() / "mistral_instruct_tokenizer_241114.model.v7m1",
        tokenizer_version,
    )

    image_encoder = _get_image_encoder(tokenizer) if image else None

    instruct_tokenizer: InstructTokenizer = _get_instruct_tokenizer_class(tokenizer_version)(
        tokenizer=tokenizer,
        image_encoder=image_encoder,
    )
    return MistralTokenizer(
        instruct_tokenizer,
        validator=get_validator(mode=validation_mode, version=tokenizer_version),
        request_normalizer=get_normalizer(version=tokenizer_version),
    )


def _get_mistral_tokenizer(
    spm: bool,
    tokenizer_version: TokenizerVersion,
    validation_mode: ValidationMode,
    image: bool,
    audio: bool,
    think: bool,
) -> MistralTokenizer:
    r"""Get a `MistralTokenizer` instance for testing.

    Args:
        spm: Whether to use SentencePiece backend instead of Tekken.
        tokenizer_version: Version of the tokenizer to build.
        validation_mode: Validation mode (test or finetuning).
        image: Whether to attach an image encoder.
        audio: Whether to attach an audio encoder (Tekken only).
        think: Whether to enable thinking special tokens (Tekken only).

    Returns:
        A configured `MistralTokenizer` instance.
    """
    if spm:
        return _get_mistral_sentencepiece(tokenizer_version, validation_mode, image)
    else:
        return _get_mistral_tekkenizer(tokenizer_version, validation_mode, image, audio, think)

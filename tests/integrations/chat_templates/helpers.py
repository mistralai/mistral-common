r"""Reusable helper functions for chat template tests.

This module contains utility functions used by test files across the
`unit/` and `transformers/` subdirectories.  It intentionally has no
fixtures or test data -- those live in `conftest.py` and `fixtures_data.py`.

HuggingFace-specific helpers live in `hf_utils.py`.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import BaseLoader
from jinja2.sandbox import ImmutableSandboxedEnvironment
from PIL import Image

from mistral_common.integrations.chat_templates.template_generator import TemplateConfig
from mistral_common.protocol.instruct.request import ChatCompletionRequest, ReasoningEffort
from mistral_common.protocol.instruct.validator import ValidationMode
from mistral_common.tokens.tokenizers.audio import Audio
from mistral_common.tokens.tokenizers.base import TokenizerVersion
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.model_settings_builder import EnumBuilder, ModelSettingsBuilder
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
_AUDIO = _AUDIO_URL


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


def _get_mistral_tekkenizer(config: TestConfig, output_dir: Path, validation_mode: ValidationMode) -> MistralTokenizer:
    r"""Build a `MistralTokenizer` with Tekken backend via `from_file`.

    Writes a tekken.json with the desired version and features via
    `_build_tekken_json`, then loads it through the production
    `MistralTokenizer.from_file` path.

    Args:
        config: Test configuration specifying version, image, audio, think.
        output_dir: Directory to write the temporary tekken.json file.
        validation_mode: Validation mode (test or finetuning).

    Returns:
        A configured `MistralTokenizer` instance.
    """
    tekken_path = _build_tekken_json(config, output_dir)
    return MistralTokenizer.from_file(str(tekken_path), mode=validation_mode)


def _get_mistral_sentencepiece(
    config: TestConfig, output_dir: Path, validation_mode: ValidationMode
) -> MistralTokenizer:
    r"""Build a `MistralTokenizer` with SentencePiece backend via `from_file`.

    Copies the shipped SPM model with the correct version suffix via
    `_build_spm_path`, then loads it through the production
    `MistralTokenizer.from_file` path.

    Args:
        config: Test configuration specifying version and image flag.
        output_dir: Directory to copy the SPM model file into.
        validation_mode: Validation mode (test or finetuning).

    Returns:
        A configured `MistralTokenizer` instance.
    """
    spm_path = _build_spm_path(config, output_dir)
    return MistralTokenizer.from_file(str(spm_path), mode=validation_mode)


def _get_mistral_tokenizer(
    spm: bool,
    tokenizer_version: TokenizerVersion,
    validation_mode: ValidationMode,
    image: bool,
    audio: bool,
    think: bool,
    output_dir: Path,
) -> MistralTokenizer:
    r"""Get a `MistralTokenizer` instance for testing.

    Args:
        spm: Whether to use SentencePiece backend instead of Tekken.
        tokenizer_version: Version of the tokenizer to build.
        validation_mode: Validation mode (test or finetuning).
        image: Whether to attach an image encoder.
        audio: Whether to attach an audio encoder (Tekken only).
        think: Whether to enable thinking special tokens (Tekken only).
        output_dir: Temporary directory for tokenizer files.

    Returns:
        A configured `MistralTokenizer` instance.
    """
    config = TestConfig(
        version=tokenizer_version,
        spm=spm,
        image=image,
        audio=audio,
        think=think,
        plain_think=False,
    )
    if spm:
        return _get_mistral_sentencepiece(config, output_dir, validation_mode)

    return _get_mistral_tekkenizer(config, output_dir, validation_mode)
